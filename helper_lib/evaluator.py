import torch


def evaluate_model(
    model,
    data_loader,
    criterion,
    device="cpu"
):
    # Move model to selected device
    model = model.to(device)

    # Set model to evaluation mode
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    # Disable gradient calculation
    with torch.no_grad():

        for images, labels in data_loader:

            # Move data to device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            # Get predicted class
            _, predicted = torch.max(outputs, 1)

            # Count total observations
            total += labels.size(0)

            # Count correct predictions
            correct += (predicted == labels).sum().item()

    # Calculate average loss
    avg_loss = running_loss / len(data_loader)

    # Calculate accuracy
    accuracy = correct / total

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    return avg_loss, accuracy