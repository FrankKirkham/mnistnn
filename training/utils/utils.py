"""
Helper functions and classes for train.py and test.py
"""

import torch
import torch.nn as nn

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
    
    def classify(self, x):
        return torch.argmax(x, dim=1)