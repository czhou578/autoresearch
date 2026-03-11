import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import Subset

class FireModule(nn.Module):
    def __init__(self, in_channels, sqz_out_channels, expand_filters_one, expand_filters_three):
        super().__init__()

        self.squeeze_layers = nn.Conv2d(in_channels, sqz_out_channels, kernel_size=1)

        self.expand_ones = nn.Sequential(
            nn.Conv2d(sqz_out_channels, expand_filters_one, kernel_size=1),
            nn.ReLU()            
        )

        self.expand_threes = nn.Sequential(
            nn.Conv2d(sqz_out_channels, expand_filters_three, kernel_size=3, padding=1),
            nn.ReLU()            
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.squeeze_layers(x)

        return torch.cat([self.expand_ones(x), self.expand_threes(x)], dim=1)


class SqueezeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            # nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2),
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1),

            nn.ReLU(), # not specified clearly within paper
            # nn.MaxPool2d(kernel_size=3, stride=2),
            FireModule(96, sqz_out_channels=16, expand_filters_one=64, expand_filters_three=64),
            FireModule(128, sqz_out_channels=16, expand_filters_one=64, expand_filters_three=64),
            FireModule(128, sqz_out_channels=32, expand_filters_one=128, expand_filters_three=128),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            FireModule(256, sqz_out_channels=32, expand_filters_one=128, expand_filters_three=128),
            FireModule(256, sqz_out_channels=48, expand_filters_one=192, expand_filters_three=192),
            FireModule(384, sqz_out_channels=48, expand_filters_one=192, expand_filters_three=192),
            FireModule(384, sqz_out_channels=64, expand_filters_one=256, expand_filters_three=256),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            FireModule(512, sqz_out_channels=64, expand_filters_one=256, expand_filters_three=256),
            # nn.Dropout(0.5),
            nn.Conv2d(in_channels=512, out_channels=100, kernel_size=1, stride=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    
    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
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

model = SqueezeNet().to(device)

lr = 0.001
batch_size = 256
epochs = 10

train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=6)
val_loader = DataLoader(cifar_val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=6)
loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr * 3, epochs=epochs, steps_per_epoch=len(train_loader), pct_start=0.3, anneal_strategy='cos', div_factor=25.0, final_div_factor=1000.0)
import time

start_time = time.time()
best_val_loss = float('inf')
num_steps = 0

for epoch in range(epochs):
    model.train()
    for inputs, targets in train_loader:
        if time.time() - start_time > 300:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        num_steps += 1
    
    model.eval()
    val_loss = 0.0
    val_batches = 0
    with torch.no_grad():
        for val_inputs, val_targets in val_loader:
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
            val_outputs = model(val_inputs)
            val_batch_loss = loss_function(val_outputs, val_targets)
            val_loss += val_batch_loss.item()
            val_batches += 1
    avg_val_loss = val_loss / val_batches
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_squeezenet.pth')
    
    if time.time() - start_time > 300:
        break

total_seconds = time.time() - start_time
peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
num_params_M = sum(p.numel() for p in model.parameters()) / 1e6

print("---")
print(f"loss:             {best_val_loss:.6f}")
print(f"training_seconds: {total_seconds:.1f}")
print(f"total_seconds:    {total_seconds:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {num_steps}")
print(f"num_params_M:     {num_params_M:.2f}")
