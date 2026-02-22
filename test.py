import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image

# CNN Model
class FruitCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Load classes
dataset = datasets.ImageFolder("data/train")
classes = dataset.classes

# Load model
model = FruitCNN(len(classes))
model.load_state_dict(torch.load("model/fruit_model.pth"))
model.eval()

# Calorie dictionary
calorie_dict = {
    "apple": 52,
    "banana": 96,
    "orange": 47
}

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ---- IMAGE INPUT FROM USER ----
image_path = input("Enter image path: ")

image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)

# Prediction
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)

fruit = classes[predicted.item()]
calories = calorie_dict[fruit]

print(f"\nDetected Fruit: {fruit.capitalize()}")
print(f"Estimated Calories: {calories} kcal")
