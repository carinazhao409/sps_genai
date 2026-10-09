import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from cnn_model import CNN


# --------------------------------------------------
# 1. Choose device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# 2. Image preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


# --------------------------------------------------
# 3. Load CIFAR-10 datasets
# --------------------------------------------------

train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# --------------------------------------------------
# 4. Create DataLoaders
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# --------------------------------------------------
# 5. CIFAR-10 class names
# --------------------------------------------------

classes = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
)


# --------------------------------------------------
# 6. Create the CNN model
# --------------------------------------------------

model = CNN().to(device)


# --------------------------------------------------
# 7. Define loss function and optimizer
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# --------------------------------------------------
# 8. Train the model
# --------------------------------------------------

num_epochs = 5

for epoch in range(num_epochs):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear old gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Loss: {average_loss:.4f}"
    )


# --------------------------------------------------
# 9. Evaluate the model
# --------------------------------------------------

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


accuracy = 100 * correct / total

print(
    f"Test Accuracy: {accuracy:.2f}%"
)


# --------------------------------------------------
# 10. Save trained model
# --------------------------------------------------

torch.save(
    model.state_dict(),
    "cifar10_cnn.pth"
)

print(
    "Model saved as cifar10_cnn.pth"
)