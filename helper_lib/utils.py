import torch


def save_model(
    model,
    optimizer,
    epoch,
    loss,
    accuracy,
    file_path
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "accuracy": accuracy
    }

    torch.save(checkpoint, file_path)

    print(f"Model saved to {file_path}")


def load_model(
    model,
    optimizer,
    file_path,
    device="cpu"
):
    checkpoint = torch.load(
        file_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    accuracy = checkpoint["accuracy"]

    print(f"Model loaded from {file_path}")

    return model, optimizer, epoch, loss, accuracy