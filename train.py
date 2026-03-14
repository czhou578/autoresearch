import time
import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, reduction_factor = 4, stride = 1):
        super().__init__()

        expanded_channels = in_channels * expansion_factor

        self.use_skip = (in_channels == out_channels and stride == 1)

        self.expansion_layers = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, kernel_size=(1, 1), stride=1),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(),
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size=(3, 3), groups=expanded_channels, padding=1, stride=stride),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(),        
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.excitation_layers = nn.Sequential(
            nn.Linear(expanded_channels, expanded_channels // reduction_factor),
            nn.SiLU(),
            nn.Linear(expanded_channels // reduction_factor, expanded_channels),
            nn.Sigmoid(),
        )

        self.contract_layers = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )

    
    def forward(self, x):
        features = self.expansion_layers(x)
        pooled_features = self.global_avg_pool(features)
        squeezed_features = pooled_features.squeeze(-1).squeeze(-1)
        excited_features = self.excitation_layers(squeezed_features)
        unsqueezed_features = excited_features.unsqueeze(-1).unsqueeze(-1)
        attention_features = unsqueezed_features * features
        final_features = self.contract_layers(attention_features)

        return final_features + x if self.use_skip else final_features
    
class EfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),

            MBConvBlock(32, 64, 2, 4, 2),
            MBConvBlock(64, 64, 2, 4),
            MBConvBlock(64, 64, 2, 4),

            MBConvBlock(64, 128, 3, 4, 2),
            MBConvBlock(128, 128, 3, 4, 1),
            MBConvBlock(128, 128, 3, 4, 1),
            MBConvBlock(128, 128, 3, 4, 1),

            MBConvBlock(128, 256, 3, 4, 2),
            MBConvBlock(256, 256, 3, 4, 1),
            MBConvBlock(256, 256, 3, 4, 1),
            MBConvBlock(256, 256, 3, 4, 1),


            nn.Conv2d(256, 512, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(512, 100)
    
    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.dropout(self.linear(x))

        return x

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.cuda.empty_cache()

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Mild to avoid over-distortion
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761]),
    transforms.RandomErasing(p=0.5)  # Apply after normalization for consistency
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
    cifar_train,  # Use directly
    batch_size=1024,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=6
)

val_loader = DataLoader(
    cifar_val,  # Use directly
    batch_size=1024,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=6
)

test_loader = DataLoader(
    cifar_test,
    batch_size=1024,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=6
)

num_classes = 100

model = EfficientNet().to(device)

num_epochs = 40
loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)
base_lr = 4e-3

batch_scale = 1024 / 256  # 4x larger batches
scaled_lr = base_lr * batch_scale**0.5  # Square root scaling

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-3,  # Keep this for now, let OneCycleLR handle it
    weight_decay=5e-4
)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=5e-3,                # Slightly lower peak LR for stability
    epochs=num_epochs,          # Keep 45 epochs
    steps_per_epoch=len(train_loader),
    pct_start=0.4,              # Increase warmup to 40% (18 epochs)
    anneal_strategy='cos',
    div_factor=12.0,            # Start LR = 5e-3 / 12 = 4.2e-4
    final_div_factor=400.0      # Final LR = 5e-3 / 400 = 1.25e-5
)

best_val_loss = float('inf')

start_time = time.time()
training_seconds = 0
num_steps = 0
total_start_time = time.time()

num_params_M = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

for epoch in range(num_epochs):
    print(f'Starting Epoch {epoch+1}')
    model.train()

    current_loss = 0.0
    num_batches = 0

    for i, data in enumerate(train_loader):
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)
            
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        current_loss += loss.item()
        num_batches += 1
        num_steps += 1

        print(f'Batch {num_batches}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        training_seconds = time.time() - start_time
        if training_seconds > 300:
            print("Time limit reached. Stopping training.")
            break
            
    if training_seconds > 300:
        break


    avg_train_loss = current_loss / num_batches
    print(f'Epoch {epoch+1} finished')
    print(f'Training - Loss: {avg_train_loss:.4f}')

    if (epoch + 1) % 2 == 0:
        model.eval()
        val_loss = 0.0
        val_batches = 0

        print(f'Epoch {epoch+1} finished')
        print(f'average training loss is {avg_train_loss:.4f}')

        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_targets = val_data
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)  # Convert inputs to FP16

                val_outputs = model(val_inputs)
                val_batch_loss = loss_function(val_outputs, val_targets)

                val_loss += val_batch_loss.item()
                val_batches += 1


        avg_val_loss = val_loss / val_batches
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        print(f'Epoch {epoch+1} finished')
        print(f'Training - Loss: {avg_train_loss:.4f}')
        print(f'Validation - Loss: {avg_val_loss:.4f}')

if torch.cuda.is_available():
    torch.cuda.empty_cache()

peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
total_seconds = time.time() - total_start_time

print("---")
print(f"loss:          {best_val_loss:.6f}")
print(f"training_seconds: {training_seconds:.1f}")
print(f"total_seconds:    {total_seconds:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {num_steps}")
print(f"num_params_M:     {num_params_M:.1f}")