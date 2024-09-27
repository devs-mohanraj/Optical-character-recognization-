import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F  # Import functional module

# Check if GPU is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NumberCNN(nn.Module):
    def __init__(self):
        super(NumberCNN, self).__init__()
        self.con1 = nn.Conv2d(1, 32, kernel_size=3)  # Output: 26x26
        self.con2 = nn.Conv2d(32, 64, kernel_size=3)  # Output: 11x11
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # Update to 1600
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.relu(self.con1(x))        # Apply ReLU after convolution
        x = F.max_pool2d(x, 2)          # Call pooling as a function
        x = F.relu(self.con2(x))        # Apply ReLU after convolution
        x = F.max_pool2d(x, 2)          # Call pooling as a function
        x = x.view(x.size(0), -1)       # Flatten the tensor
        x = F.relu(self.fc1(x))         # Apply ReLU after the first fully connected layer
        x = self.fc2(x)                  # Call the final layer
        return x
    
# Transformations for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', download=True, train=True, transform=transform)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Model, loss function, and optimizer
model = NumberCNN().to(device)  # Move the model to the appropriate device
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
epochs = 50
for epoch in range(epochs):
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)  # Move data to device
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)  # Use the criterion defined
        loss.backward()
        optimizer.step()
        
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    
# Saving the model
torch.save(model.state_dict(), "./models/numbers_model_path.pth")
