import torch
from torch.utils.data import DataLoader
from torch import nn
from collections import OrderedDict
from torchvision import transforms, datasets
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import time

start_total_time = time.time()

class BottleNeckBlock(nn.Module):
    def __init__(self, in_channels, out_channel, stride=1):
        super().__init__()
        self.layers = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv2d(in_channels, out_channel // 4, kernel_size=(1, 1))),
            ('bn1_1', nn.BatchNorm2d(out_channel // 4)),
            ('relu1_1', nn.ReLU()),
            ('conv1_2',  nn.Conv2d(out_channel // 4, out_channel // 4, kernel_size=(3, 3), stride=stride, padding=1)),
            ('bn1_2', nn.BatchNorm2d(out_channel // 4)),
            ('relu1_2', nn.ReLU()),
            ('conv1_3', nn.Conv2d(out_channel // 4, out_channel, kernel_size=(1, 1))),
            ('bn1_3', nn.BatchNorm2d(out_channel)),

        ]))
        self.last_relu = nn.ReLU()
        if stride != 1 or in_channels != out_channel:
            self.shortcut = nn.Sequential(OrderedDict([
                ('sho_conv_1', nn.Conv2d(in_channels, out_channel, kernel_size=(1, 1), stride=stride)),
                ('sho_bn1', nn.BatchNorm2d(out_channel))
            ]))
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        x = self.layers(x) + self.shortcut(x)
        x = self.last_relu(x)
        return x

class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv2d(3, 64, kernel_size=(3, 3), stride=2)),
            ('bn1_1', nn.BatchNorm2d(64)),
            ('relu1_1', nn.ReLU()),
            ('block1', BottleNeckBlock(in_channels=64, out_channel=256)),
            ('block2', BottleNeckBlock(in_channels=256, out_channel=256)),
            ('block3', BottleNeckBlock(in_channels=256, out_channel=256)),
            ('block4', BottleNeckBlock(in_channels=256, out_channel=512, stride=2)),

            ('block5', BottleNeckBlock(in_channels=512, out_channel=512)),
            ('block6', BottleNeckBlock(in_channels=512, out_channel=512)),
            ('block7', BottleNeckBlock(in_channels=512, out_channel=512)),
            ('block8', BottleNeckBlock(in_channels=512, out_channel=1024, stride=2)),
            
            ('block9', BottleNeckBlock(in_channels=1024, out_channel=1024)),
            ('block10', BottleNeckBlock(in_channels=1024, out_channel=1024)),
            ('block11', BottleNeckBlock(in_channels=1024, out_channel=1024)),
            ('block12', BottleNeckBlock(in_channels=1024, out_channel=1024)),
            ('block13', BottleNeckBlock(in_channels=1024, out_channel=1024, stride=2)),

            ('block14', BottleNeckBlock(in_channels=1024, out_channel=2048)),
            ('block15', BottleNeckBlock(in_channels=2048, out_channel=2048)),
            ('block16', BottleNeckBlock(in_channels=2048, out_channel=2048)),

        ]))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(2048, 100)
    
    def forward(self, x):
        x = self.layers(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear(x)
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
    transforms.RandomErasing(p=0.7)  # Apply after normalization for consistency
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
    batch_size=512,
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

model = ResNet50().to(device)

num_epochs = 45
loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)
base_lr = 4e-3

batch_scale = 1024 / 256  # 4x larger batches
scaled_lr = base_lr * batch_scale**0.5  # Square root scaling

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,  # Changed from 1e-3
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

start_train_time = time.time()
train_time_budget = 300.0  # 5 minutes
total_steps = 0
elapsed_train_time = 0.0

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

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Changed from 1.0

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()  # Step per batch for OneCycleLR

        current_loss += loss.item()
        num_batches += 1
        total_steps += 1

        print(f'Batch {i}/{len(train_loader)}, Step Loss: {loss.item():.4f}')
        if i % 50 == 0:
            torch.cuda.empty_cache()

        elapsed_train_time = time.time() - start_train_time
        if elapsed_train_time >= train_time_budget:
            print("Time budget reached during epoch, stopping training loop.")
            break

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

    if elapsed_train_time >= train_time_budget:
        break

total_seconds = time.time() - start_total_time
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
else:
    peak_vram_mb = 0.0
num_params_m = sum(p.numel() for p in model.parameters()) / 1e6

print("\n---")
print(f"loss:          {best_val_loss:.6f}")
print(f"training_seconds: {elapsed_train_time:.1f}")
print(f"total_seconds:    {total_seconds:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {total_steps}")
print(f"num_params_M:     {num_params_m:.1f}")