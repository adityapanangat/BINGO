import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import copy

# Define the network
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 300)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to train the model
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Function to evaluate the model
def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Function to prune the model by magnitude
def prune_model_by_magnitude(model, prune_percentage):
    # Flatten all weights into a single tensor
    all_weights = torch.cat([param.data.view(-1) for param in model.parameters()])
    threshold = torch.quantile(torch.abs(all_weights), prune_percentage)

    # Create mask for weights
    masks = {}
    for name, param in model.named_parameters():
        masks[name] = torch.abs(param.data) > threshold

    return masks

# Apply the mask to the model
def apply_pruning_mask(model, masks):
    for name, param in model.named_parameters():
        param.data *= masks[name]

# Reset pruned weights to their initial values
def reset_to_initial(model, initial_state, masks):
    for name, param in model.named_parameters():
        param.data[masks[name] == 0] = initial_state[name][masks[name] == 0]

# Main script
def main():
    # Settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    epochs = 5
    prune_percentage = 0.2  # Fraction of weights to prune per round
    prune_rounds = 5

    # Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_data = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Initialize model, loss, and optimizer
    model = SimpleMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Save the initial state
    initial_state = {name: param.data.clone() for name, param in model.named_parameters()}

    # Train and evaluate before pruning
    print("Training initial model...")
    for epoch in range(epochs):
        train_model(model, train_loader, criterion, optimizer, device)
    initial_accuracy = evaluate_model(model, test_loader, device)
    print(f"Initial accuracy: {initial_accuracy:.2f}%")

    # Iterative Magnitude Pruning
    for round in range(prune_rounds):
        print(f"Pruning round {round + 1}/{prune_rounds}...")

        # Prune and reset weights
        masks = prune_model_by_magnitude(model, prune_percentage)
        apply_pruning_mask(model, masks)
        reset_to_initial(model, initial_state, masks)

        # Train the pruned model
        for epoch in range(epochs):
            train_model(model, train_loader, criterion, optimizer, device)

        # Evaluate the pruned model
        accuracy = evaluate_model(model, test_loader, device)
        print(f"Accuracy after round {round + 1}: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
