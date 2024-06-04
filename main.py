import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim):
        super(ViT, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2  # RGB channels

        self.patch_size = patch_size
        self.patch_embedding = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        patches = self.patch_embedding(x)
        batch_size, _, height, width = patches.shape
        patches = patches.permute(0, 2, 3, 1).view(batch_size, height*width, -1)
        embeddings = patches + self.positional_embedding[:, :height*width]
        features = self.transformer_encoder(embeddings)
        cls_token = features[:, 0, :]
        output = self.fc(cls_token)
        return output

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

classes_to_keep = ['airplane', 'automobile', 'bird', 'cat']

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_indices = [idx for idx, label in enumerate(train_dataset.targets) if train_dataset.classes[label] in classes_to_keep]
test_indices = [idx for idx, label in enumerate(test_dataset.targets) if test_dataset.classes[label] in classes_to_keep]

train_dataset = Subset(train_dataset, train_indices)
test_dataset = Subset(test_dataset, test_indices)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(len(train_loader.dataset))
print(len(val_loader.dataset))
print(len(test_loader.dataset))

model = ViT(
    image_size=224,
    patch_size=16,
    num_classes=len(classes_to_keep),
    dim=768,
    depth=12,
    heads=12,
    mlp_dim=3072
)

epochs = 10

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= 0.001 , betas=(0.9, 0.999), weight_decay=0.1)

train_losses = []
eval_accuracies = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    eval_accuracies.append(accuracy)

    print(f"Epoch {epoch+1}, Loss: {train_losses[-1]}, Accuracy on val set: {accuracy}%")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test set: {100 * correct / total}%")

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color=color)
ax1.plot(range(1, epochs+1), train_losses, color=color, marker='o')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Accuracy (%)', color=color)
ax2.plot(range(1, epochs+1), eval_accuracies, color=color, marker='o')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Training Loss and Validation Set Accuracy')
plt.show()