import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    Convolutional Neural Network for 64x64 RGB images.

    Architecture:
        Input: 3 x 64 x 64
        Conv2d: 3 -> 16, kernel_size=3, stride=1, padding=1
        ReLU
        MaxPool2d: kernel_size=2, stride=2

        Conv2d: 16 -> 32, kernel_size=3, stride=1, padding=1
        ReLU
        MaxPool2d: kernel_size=2, stride=2

        Flatten
        Linear: 8192 -> 100
        ReLU
        Linear: 100 -> 10
    """

    def __init__(self):
        super().__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # ReLU activation
        self.relu = nn.ReLU()

        # Max pooling layer
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # First fully connected layer
        self.fc1 = nn.Linear(
            32 * 16 * 16,
            100
        )

        # Output layer
        self.fc2 = nn.Linear(
            100,
            10
        )

    def forward(self, x):

        # First convolution block
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Second convolution block
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten feature maps
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)

        return x


if __name__ == "__main__":

    # Create the model
    model = CNN()

    # Create one fake RGB image:
    # batch_size=1, channels=3, height=64, width=64
    test_input = torch.randn(1, 3, 64, 64)

    # Forward pass
    output = model(test_input)

    print(model)
    print()
    print("Input shape:", test_input.shape)
    print("Output shape:", output.shape)