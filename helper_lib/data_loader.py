import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def preprocess_fashion_mnist(img):
    """
    Pad Fashion-MNIST images from 28x28 to 32x32.
    """
    img = np.pad(
        img,
        ((2, 2), (2, 2)),
        constant_values=0.0
    )

    return img


def get_data_loader(
    data_dir="./data",
    batch_size=32,
    train=True,
    dataset_name="CIFAR10"
):
    """
    Create a DataLoader for CIFAR-10 or Fashion-MNIST.

    Parameters
    ----------
    data_dir : str
        Directory where the dataset is stored.

    batch_size : int
        Number of images in each batch.

    train : bool
        If True, load the training dataset.
        If False, load the test dataset.

    dataset_name : str
        Dataset to load.
        Options: "CIFAR10" or "FashionMNIST".

    Returns
    -------
    DataLoader
        PyTorch DataLoader for the selected dataset.
    """

    # -------------------------------------------------
    # CIFAR-10
    # Used for the Module 4 models
    # -------------------------------------------------

    if dataset_name == "CIFAR10":

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=train,
            download=True,
            transform=transform
        )


    # -------------------------------------------------
    # Fashion-MNIST
    # Used for the Module 5 VAE
    # -------------------------------------------------

    elif dataset_name == "FashionMNIST":

        transform = transforms.Compose([
            transforms.Lambda(preprocess_fashion_mnist),
            transforms.ToTensor()
        ])

        dataset = torchvision.datasets.FashionMNIST(
            root=data_dir,
            train=train,
            download=True,
            transform=transform
        )


    # -------------------------------------------------
    # Invalid dataset name
    # -------------------------------------------------

    else:

        raise ValueError(
            "dataset_name must be 'CIFAR10' or 'FashionMNIST'"
        )


    # -------------------------------------------------
    # Create DataLoader
    # -------------------------------------------------

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train
    )

    return data_loader