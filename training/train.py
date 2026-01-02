# To train the model

import torch
import torch.nn as nn
from tqdm import tqdm # For progress bars
from utils.preprocessing import load_data

class SimpleFCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Define our layers
        self.l1 = nn.Linear(784, 128) # 784 pixels for 28x28 MNIST image
        self.l2 = nn.Linear(128, 10) # 10 possible numbers (0-9)   
    
    def forward(self, x):
        # Flatten layer of data so it is in correct format
        # (1, 784) instead of (1, 28, 28)
        x = torch.flatten(x, 1)

        # Hidden layer
        x = nn.functional.relu(self.l1(x))
        # Output layer
        x = self.l2(x)
        return x

def train(neural_network, train_loader, epochs, learn_rate):
    # Set the model/nn to work on the set device (allows us to use GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neural_network = neural_network.to(device)

    # Set up a simple gradient descent optimiser
    optimizer = torch.optim.SGD(neural_network.parameters(), lr=learn_rate)
    # Set up a cross entropy loss function
    criterion = nn.CrossEntropyLoss()

    for _ in tqdm(range(epochs), desc="Training Model"):    
        # Set the model to train mode
        neural_network.train()

        for images, labels in train_loader:
            # Make sure they are running on the same device
            images, labels = images.to(device), labels.to(device)
            # Feed the data into the model
            optimizer.zero_grad()
            outputs = neural_network.forward(images)
            # Apply loss function
            loss = criterion(outputs, labels)
            # Backpropagate
            loss.backward()
            optimizer.step()

def main():
    train_loader, val_loader, test_loader, means, stds = load_data()

    neural_network = SimpleFCNN()

    train(neural_network, train_loader, 1000, 0.001)


if __name__ == "__main__":
    main()