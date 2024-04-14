import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
import pandas as pd
from PIL import Image
import numpy as np

# Define your CNN model
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()
        # Define your model architecture, you can use pre-trained models as well
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

# Load your labeled data
class FundusDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        image = Image.open(img_name).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Load labeled image and corresponding labels
labeled_image_path = "./lab.jpg"
labels_csv_path = "./_classes.csv"

# Load labeled classes
classes_df = pd.read_csv(labels_csv_path)

# Assuming classes.csv has columns: image_path, label
image_paths = classes_df['image_path'].tolist()
labels = classes_df['label'].tolist()

# Assuming you have more unlabeled image paths
unlabeled_image_paths = ["./image1.jpg", "./image2.jpg", ...]

# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create labeled dataset
labeled_dataset = FundusDataset([labeled_image_path], labels, transform=transform)

# Create unlabeled dataset
unlabeled_dataset = FundusDataset(unlabeled_image_paths, [0]*len(unlabeled_image_paths), transform=transform)

# Combine datasets for training
combined_dataset = torch.utils.data.ConcatDataset([labeled_dataset, unlabeled_dataset])

# Define data loaders
data_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=32, shuffle=True)

# Initialize model
model = MyModel(num_classes=len(classes_df))

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model using self-training
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, num_epochs, running_loss))

# Predict labels for unlabeled images
model.eval()
predicted_labels = []
for images, _ in unlabeled_dataset:
    images = images.unsqueeze(0)  # Add batch dimension
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    predicted_labels.append(predicted.item())

# Output labeled images
for i in range(len(unlabeled_image_paths)):
    print("Image:", unlabeled_image_paths[i])
    print("Predicted Label:", predicted_labels[i])
