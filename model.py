import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from copy import deepcopy

percent_to_prune = 0.3  


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def calculate_importance_scores(model, criterion, inputs, targets, mode, initial_state=None):
    """
    Calculate importance scores for neurons.

    Args:
        model: The model to evaluate.
        criterion: Loss function.
        inputs: Input data.
        targets: Target labels.
        mode: 'dropout' or 'reset'.
        initial_state: Initial weights of the model (used for 'reset' mode).

    Returns:
        importance_scores: A dictionary containing importance scores for each neuron.
    """

    model.eval()
    with torch.no_grad():
        baseline_loss = criterion(model(inputs), targets).item()
        
        importance_scores = {}
        for name, param in model.named_parameters():
            if 'weight' in name and len(param.shape) == 2:  # Focus on layer weights
                scores = []
                for i in range(param.shape[0]):  # iterative over all neurons
                    modified_model = deepcopy(model)
                    modified_layer = getattr(modified_model, name.split('.')[0])

                    if mode == 'dropout':
                        modified_layer.weight.data[i, :] = 0  # simulates dropout
                    elif mode == 'reset' and initial_state is not None:
                        modified_layer.weight.data[i, :] = initial_state[name][i, :]  # reset to initial weights

                    loss = criterion(modified_model(inputs), targets).item()
                    scores.append(baseline_loss - loss)  # Higher difference -> higher importance
                
                importance_scores[name] = torch.tensor(scores)
    return importance_scores

# function to prune neurons based on importance scores
def prune_neurons(model, importance_scores, percent_to_prune):
    for name, scores in importance_scores.items():
        num_to_prune = int(len(scores) * percent_to_prune)
        _, indices_to_prune = torch.topk(scores, num_to_prune, largest=False)
        layer = getattr(model, name.split('.')[0])
        layer.weight.data[indices_to_prune, :] = 0  # Prune weights

# init two models with the same weights
input_size = 28 * 28
hidden_size = 128
output_size = 10

model1 = SimpleNN(input_size, hidden_size, output_size)
model2 = SimpleNN(input_size, hidden_size, output_size)

# copy initial weights for model2 and for reset tracking
initial_weights = deepcopy(model1.state_dict())
model2.load_state_dict(initial_weights)

# Load MNIST in
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)

# training loop starts here --
num_epochs = 10
for epoch in range(num_epochs):
    model1.train()
    model2.train()

    for inputs, targets in train_loader:
        inputs = inputs.view(-1, 28 * 28)

        # foward
        outputs1 = model1(inputs)
        loss1 = criterion(outputs1, targets)

        outputs2 = model2(inputs)
        loss2 = criterion(outputs2, targets)

        # backward!

        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()

        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss1: {loss1.item():.2f}, Loss2: {loss2.item():.2f}")

inputs, targets = next(iter(train_loader))
inputs = inputs.view(-1, 28 * 28)

importance_scores1 = calculate_importance_scores(model1, criterion, inputs, targets, mode='dropout')
importance_scores2 = calculate_importance_scores(model2, criterion, inputs, targets, mode='reset', initial_state=initial_weights)

# eval
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.view(-1, 28 * 28)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return 100 * correct / total

# count num neurons (used before and after pruning processes -- should update to count weights)
def count_neurons(model):
    """Count the total number of neurons in the model."""
    neuron_counts = {}
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) == 2:  
            neuron_counts[name] = param.shape[0]  
    return neuron_counts

prePruneAcc1 = evaluate_model(model1, test_loader)
prePruneAcc2 = evaluate_model(model2, test_loader)

print(f"Prepruned Accuracy of Model 1 (Dropout): {prePruneAcc1:.2f}% || num neurons: {count_neurons(model1)}")
print(f"Prepruned Accuracy of Model 2 (Reset): {prePruneAcc1:.2f}% || num neurons: {count_neurons(model2)}")

prune_neurons(model1, importance_scores1, percent_to_prune)
prune_neurons(model2, importance_scores2, percent_to_prune)

accuracy1 = evaluate_model(model1, test_loader)
accuracy2 = evaluate_model(model2, test_loader)

print(f"Final Accuracy of Model 1 (Dropout): {accuracy1:.2f}% || num neurons: {count_neurons(model1)}")
print(f"Final Accuracy of Model 2 (Reset): {accuracy2:.2f}% || num neurons: {count_neurons(model2)}")
