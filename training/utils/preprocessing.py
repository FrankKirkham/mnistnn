# Pre processing the data for the model

import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def load_data():    
    # Produce stats used to normalise the data
    means, stds = produce_stats()
    
    # Set the transform function 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    # Download the training dataset
    train_data = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    # Create a validation set
    train_size = int(0.9 * len(train_data))
    val_size = len(train_data) - train_size

    train_dataset, val_dataset = random_split(
        train_data, [train_size, val_size]
    )

    # Create and return dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_data, batch_size=64)

    return train_loader, val_loader, test_loader, means, stds

def produce_stats():
    # Set the transform function 
    transform = transforms.ToTensor() 

    # Download the training dataset
    train_data = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    # Stack all images together
    all_images = torch.stack([img for img, _ in train_data], dim=0)

    # Compute the mean and std, needs [0,2,3] for samples, height and width
    # These will produce a mean and std for each channel (RGB), however MNIST is greyscale
    means = all_images.mean(dim=[0,2,3])
    stds = all_images.std(dim=[0,2,3])

    return means, stds

# dimensionality reduce?
