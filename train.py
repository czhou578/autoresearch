import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import os

class ResNeXtModule(nn.Module):
    def __init__(self, in_channel, out_channel, reduce_channel, cardinality, stride=1):
        super().__init__()

        assert reduce_channel % cardinality == 0, \
            f"bottleneck_channels ({reduce_channel}) must be divisible by cardinality ({cardinality})"

        self.layers = nn.Sequential(
            nn.Conv2d(in_channel, reduce_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduce_channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(reduce_channel, reduce_channel, kernel_size=3, stride=stride,
                     padding=1, groups=cardinality, bias=False),
            nn.BatchNorm2d(reduce_channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(reduce_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        # Shortcut connection for dimension matching
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.layers(x)
        out += identity
        out = self.relu(out)
        return out

class ResNeXtCIFAR(nn.Module):
    def __init__(self, cardinality=8, width=64, num_classes=100):
        super().__init__()
        
        # First layer: 3×3 conv with 64 filters (no stride, no pooling for CIFAR)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1: 3 blocks, output map size 32×32, width=64
        self.stage1 = self._make_stage(64, 64, width, cardinality, 3, stride=1)
        
        # Stage 2: 3 blocks, output map size 16×16, width=128 (2× increase)
        self.stage2 = self._make_stage(64, 128, width*2, cardinality, 3, stride=2)
        
        # Stage 3: 3 blocks, output map size 8×8, width=256 (2× increase)
        self.stage3 = self._make_stage(128, 256, width*4, cardinality, 3, stride=2)
        
        # Global average pooling + fully connected
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def _make_stage(self, in_channels, out_channels, reduce_channels, cardinality, num_blocks, stride):
        """Create a stage with multiple ResNeXt blocks"""
        layers = []
        
        # First block handles downsampling and channel increase
        layers.append(ResNeXtModule(in_channels, out_channels, reduce_channels, 
                                   cardinality, stride=stride))
        
        # Remaining blocks keep same dimensions
        for _ in range(num_blocks - 1):
            layers.append(ResNeXtModule(out_channels, out_channels, reduce_channels, 
                                       cardinality, stride=1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)      # 32×32×64
        x = self.stage1(x)     # 32×32×64
        x = self.stage2(x)     # 16×16×128
        x = self.stage3(x)     # 8×8×256
        x = self.avgpool(x)    # 1×1×256
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.cuda.set_per_process_memory_fraction(0.45, 0)
    torch.cuda.empty_cache()



train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Mild to avoid over-distortion
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761]),
    transforms.RandomErasing(p=0.25)  # Apply after normalization for consistency
])

test_transform = transforms.Compose([
    transforms.ToTensor(), # Moved ToTensor before Normalize (good practice)
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

# Load raw datasets
cifar_train_raw = datasets.CIFAR100(root="./data", train=True, download=True, transform=None)

train_size = int(0.9 * len(cifar_train_raw))  # 48,000

train_indices = list(range(0, train_size))
val_indices = list(range(train_size, len(cifar_train_raw)))

# Create datasets with appropriate transforms
cifar_train = Subset(
    datasets.CIFAR100(root="./data", train=True, transform=train_transform),
    train_indices
)
cifar_val = Subset(
    datasets.CIFAR100(root="./data", train=True, transform=test_transform),
    val_indices
)
# Use original test set (10,000 samples) - close to 10% of 60,000
cifar_test = datasets.CIFAR100(root="./data", train=False, transform=test_transform)

train_loader = DataLoader(
    cifar_train,
    batch_size=512,  # Changed from 512
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=6
)

val_loader = DataLoader(
    cifar_val,  # Use directly
    batch_size=512,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=6
)

test_loader = DataLoader(
    cifar_test,
    batch_size=512,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=6
)

num_classes = 100

model = ResNeXtCIFAR(cardinality=8, width=64, num_classes=100).to(device)

num_epochs = 40
loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)
base_lr = 4e-3

batch_scale = 1024 / 256  # 4x larger batches
scaled_lr = base_lr * batch_scale**0.5  # Square root scaling

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-3,  # Changed from 1e-3
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-2,  # Changed from 3e-3
    epochs=num_epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,
    anneal_strategy='cos',
    div_factor=25.0,
    final_div_factor=1000.0
)

train_losses = []
val_losses = []
epochs_recorded = []

best_val_loss = float('inf')

for epoch in range(num_epochs):
    print(f'Starting Epoch {epoch+1}')
    model.train()

    current_loss = 0.0
    num_batches = 0

    for i, data in enumerate(train_loader):
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        try:
            # OPTIMISTIC PASS: Try full batch
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()

        except torch.cuda.OutOfMemoryError:
            # RECOVERY: VRAM spike! Clear cache and use micro-batches
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            print(f"⚠️ VRAM Spike detected at batch {i}. Recovering via Gradient Accumulation...")

            # Process 4 smaller chunks
            chunks = 4
            micro_batch_size = max(1, len(inputs) // chunks)
            micro_loss_sum = 0.0

            for start_idx in range(0, len(inputs), micro_batch_size):
                end_idx = start_idx + micro_batch_size
                m_inputs = inputs[start_idx:end_idx]
                m_targets = targets[start_idx:end_idx]

                m_outputs = model(m_inputs)
                
                # Scale loss so the accumulated gradient matches the full batch
                m_loss = loss_function(m_outputs, m_targets) / chunks
                m_loss.backward()

                micro_loss_sum += m_loss.item() * chunks # Un-scale for reporting

            # Create a detached tensor with gradient requirement for the logging below
            loss = torch.tensor(micro_loss_sum)

        optimizer.step()
        scheduler.step()  # Step per batch for OneCycleLR

        current_loss += loss.item()
        num_batches += 1

        if i % 50 == 0:
            torch.cuda.empty_cache()
            print(f'Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}')

    avg_train_loss = current_loss / num_batches

    # Validate EVERY epoch (removed the if condition)
    model.eval()
    val_loss = 0.0
    val_batches = 0

    with torch.no_grad():
        for val_data in val_loader:
            val_inputs, val_targets = val_data
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)

            val_outputs = model(val_inputs)
            val_batch_loss = loss_function(val_outputs, val_targets)

            val_loss += val_batch_loss.item()
            val_batches += 1

    avg_val_loss = val_loss / val_batches
    best_val_loss = min(best_val_loss, avg_val_loss)

    # Record metrics every epoch
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    epochs_recorded.append(epoch + 1)

    print(f'Epoch {epoch+1} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

if torch.cuda.is_available():
    torch.cuda.empty_cache()
