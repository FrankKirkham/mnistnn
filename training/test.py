# Test the model
import torch
from tqdm import tqdm
from utils.preprocessing import load_data
from utils.utils import SimpleFCNN

def eval(neural_network, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Classifying Test Samples"):
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
    print(f"--------------\nTest Results: {accuracy:.2f}% correct ({(100 - accuracy):.2f}% loss)")

def main():
    _, _, test_loader, _, _ = load_data()

    # Declare the device we are using (allows us to use GPU if there)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the saved model
    model_with_stats = torch.load("model_with_stats.pth", map_location=device)
    model = model_with_stats["model"]
    means = model_with_stats["means"]
    stds = model_with_stats["stds"]
    # Turn the model into a useable neural network
    neural_network = SimpleFCNN().to(device)
    neural_network.load_state_dict(model)

    # Run the test eval
    eval(neural_network, test_loader)


if __name__ == "__main__":
    main()