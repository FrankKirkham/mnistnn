# To train the model

import torch
import torch.nn as nn
from tqdm import tqdm # For progress bars
from utils.preprocessing import load_data
from utils.utils import SimpleFCNN

def train(neural_network, train_loader, val_loader, epochs, learn_rate):
    # Set the model/nn to work on the set device (allows us to use GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neural_network = neural_network.to(device)

    # Set up a simple gradient descent optimiser
    optimizer = torch.optim.Adam(neural_network.parameters(), lr=learn_rate)
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
            outputs = neural_network(images)
            # Apply loss function
            loss = criterion(outputs, labels)
            # Backpropagate
            loss.backward()
            optimizer.step()

    print("--------------\nTraining complete!\n--------------")
    # Validation
    correct = 0
    total = 0
    with torch.no_grad(): # Don't need gradients for classifying
        for images, labels in tqdm(val_loader, desc="Running Validation On Model"):
            # Make sure they are running on the same device
            images, labels = images.to(device), labels.to(device)
            
            # Set the model to eval mode as not to alter weights
            neural_network.eval()
            # Perform forward pass
            outputs = neural_network(images)
            # Classify result
            results = neural_network.classify(outputs)

            for i in range(len(labels)):
                total += 1
                if results[i] == labels[i]:
                    correct += 1

    accuracy = 100 * (correct / total)
    print(f"--------------\nValidation Results: {accuracy:.2f}% correct ({(100 - accuracy):.2f}% loss)")

def main():
    train_loader, val_loader, _, means, stds = load_data()

    neural_network = SimpleFCNN()

    # Train the model
    train(neural_network, train_loader, val_loader, 1, 0.001) # 1 IS USED FOR TESTING, CHANGE LATER!

    # Save trained model and stats to a file to be used later:
    model_with_stats = {
        "model": neural_network.state_dict(),
        "means": means,
        "stds": stds
    }

    torch.save(model_with_stats, "model_with_stats.pth")
    print("--------------\nModel saved")



if __name__ == "__main__":
    main()