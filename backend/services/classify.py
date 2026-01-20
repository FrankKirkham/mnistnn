import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

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
        x = F.relu(self.l1(x))
        # Output layer
        x = self.l2(x)
        return x
        
class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Define our convultion layers
        self.l1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        ) # (n, 1, 28, 28) to a (n, 32, 28, 28)
        self.l2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        ) # (n, 32, 14, 14) to a (n, 64, 14, 14)
        # Define our pooling
        self.pool = nn.MaxPool2d(2, 2)
        # Define our fully connected layers
        self.l3 = nn.Linear(64*7*7, 128)
        self.l4 = nn.Linear(128, 10) # 10 possible numbers (0-9)

    def forward(self, x):
        # Run our image through the convultion layers
        x = F.relu(self.l1(x))
        x = self.pool(x) # Run a max pool for translation invariance
        x = F.relu(self.l2(x))
        x = self.pool(x)
        
        # Flatten layer of data so it is in correct format
        # (n, 1, ...) instead of (n, 1, ...)
        x = torch.flatten(x, 1)

        # Run our feature vectors through the FC layers
        x = F.relu(self.l3(x))
        x = self.l4(x)
        
        return x

    def classify(self, x):
        return torch.argmax(x, dim=1)
    
def find_percentages(tensor) -> List[List[Tuple]]:
    # Initialise the model
    model = MnistCNN()

    # Load the model from the saved file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Allows us to run it on GPU
    checkpoint = torch.load("services/model/model_with_stats.pth", map_location=device)
    model.load_state_dict(checkpoint["model"])

    # Calculate our percentages
    model.eval() # so we don't affect weights
    results = model(tensor)
    results = nn.functional.softmax(results, dim=1) # Converts logits to actual percentages
    results = results.detach().cpu().numpy() # Turns it to a numpy array

    # Save them to a readable format
    final = []
    for result in results:
        # Find the order of percentages
        inds = np.argsort(result)[::-1] # Highest to lowest
        sorted_result = result[inds]
        # Save the percentage with the number
        zipped_result = zip(inds.tolist(), sorted_result.tolist())
        final.append(list(zipped_result))

    return final