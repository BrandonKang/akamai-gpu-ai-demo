# ai_training.py

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
import os
from tqdm import tqdm

# Define CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Set device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(weights=None, num_classes=10)

# Enable multi-GPU if available
if torch.cuda.device_count() > 1:
    print(f"✅ Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with progress bar
num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        avg_loss = running_loss / (i + 1)
        progress_bar.set_postfix(loss=loss.item(), avg_loss=avg_loss)

# Save trained model and class labels
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/resnet18_cifar10.pth")

with open("model/class_names.txt", "w") as f:
    for name in class_names:
        f.write(f"{name}\n")

print("✅ Training complete. Model saved to model/resnet18_cifar10.pth")

