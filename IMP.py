import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import os
from copy import deepcopy

# simply MNIST FC
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def count_weights(model):
    """Returns the total number of weights (parameters) in the model."""
    total_weights = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_weights += param.numel()  
    return total_weights

def train(model, trainloader, criterion, optimizer, epochs=5):
    """Train the model for a specified number of epochs."""
    model.train()
    for _ in range(epochs):
        for inputs, targets in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

def evaluate(model, testloader):
    """Evaluate the model and return accuracy."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in testloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total

def count_zero_weights(model):
    """Count how many weights in the model are equal to zero."""
    zero_count = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            zero_count += (param.data == 0).sum().item()  
    return zero_count

def iterative_magnitude_pruning(model, trainloader, testloader, prune_rounds=10, prune_ratio=0.2, epochs=5):
    """Performs iterative magnitude pruning following the lottery ticket hypothesis."""
   
    # Store initial weights for reset after pruning
    initial_state = deepcopy(model.state_dict())

    # create pruning masks
    pruning_masks = {}
    for name, param in model.named_parameters():
        if "weight" in name:
            pruning_masks[name] = torch.ones_like(param.data, dtype=torch.bool)  # Initially, all weights are trainable

    # train OG network
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, trainloader, criterion, optimizer, epochs)

    # eval
    initial_accuracy = evaluate(model, testloader)
    print(f"Initial accuracy: {initial_accuracy:.4f}")

    total_pruned_weights = 0  # Track total pruned weights

    # IMP
    for round in range(prune_rounds):
        print(f"\nPruning round {round + 1}/{prune_rounds}")
        print(f"Num zero weights before pruning: {count_zero_weights(model)}")

        round_pruned_weights = 0  # Track weights pruned in this round

        # Apply pruning mask per layer
        for name, param in model.named_parameters():
            if "weight" in name:
                # Compute layer-specific threshold
                threshold = torch.quantile(torch.abs(param.data.view(-1)), prune_ratio)
                
                # update the pruning mask: a weight is pruned if its magnitude is below the threshold
                new_mask = torch.abs(param.data) > threshold
                pruning_masks[name] &= new_mask  # Maintain previously pruned weights
                
                round_pruned_weights += (~pruning_masks[name]).sum().item()  # Count newly pruned weights
                
                # zero out pruned weights
                param.data *= pruning_masks[name]

                # only reset the unpruned weights to their initial values
                param.data[pruning_masks[name]] = initial_state[name][pruning_masks[name]]

        total_pruned_weights += round_pruned_weights  # Accumulate pruned weights
        print(f"Num zero weights before training: {count_zero_weights(model)}")

        # retrain the pruned model while enforcing the pruning mask
        def train_with_mask(model, trainloader, criterion, optimizer, epochs, pruning_masks):
            model.train()
            for _ in range(epochs):
                for inputs, targets in trainloader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    # Reapply pruning masks after every update to ensure pruned weights remain zero
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            if name in pruning_masks:
                                param.data *= pruning_masks[name]

        train_with_mask(model, trainloader, criterion, optimizer, epochs, pruning_masks)

        print(f"Num zero weights after training: {count_zero_weights(model)}")

        pruned_accuracy = evaluate(model, testloader)
        print(f"Accuracy after pruning round {round + 1}: {pruned_accuracy:.4f}")

    return model, total_pruned_weights


def count_zero_weights(model):
    return sum((param.data == 0).sum().item() for name, param in model.named_parameters() if "weight" in name)

def iterative_magnitude_pruning2(model, trainloader, testloader, prune_rounds=10, prune_ratio=0.2, epochs=5):    
    # Store initial weights for reset after pruning
    initial_state = deepcopy(model.state_dict())

    # Initialize pruning mask (same shape as weights)
    pruning_masks = {name: torch.ones_like(param.data, dtype=torch.bool) for name, param in model.named_parameters() if "weight" in name}

    # Train the initial network
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, trainloader, criterion, optimizer, epochs)

    # Evaluate accuracy before pruning
    initial_accuracy = evaluate(model, testloader)
    print(f"Initial accuracy: {initial_accuracy:.4f}")

    total_params = sum(param.numel() for name, param in model.named_parameters() if "weight" in name)
    total_pruned_weights = 0  # track total pruned weights

    # iterative pruning rounds
    for round in range(prune_rounds):
        print(f"\nPruning round {round + 1}/{prune_rounds}")
        print(f"Num zero weights before pruning: {count_zero_weights(model)}")

        # Flatten all remaining unpruned weights
        all_unpruned_weights = torch.cat([
            param.data[pruning_masks[name]].view(-1) for name, param in model.named_parameters() if "weight" in name
        ])

        num_to_prune = int(prune_ratio * total_params)  # Fixed 20% of the original model weights
        if num_to_prune > len(all_unpruned_weights):  # to stop from over-pruning
            num_to_prune = len(all_unpruned_weights)

        if num_to_prune > 0:
            threshold = torch.topk(torch.abs(all_unpruned_weights), num_to_prune, largest=False).values[-1]  # Get global threshold

            round_pruned_weights = 0
            for name, param in model.named_parameters():
                if "weight" in name:
                    # Generate new mask based on global threshold
                    new_mask = torch.abs(param.data) > threshold
                    pruning_masks[name] &= new_mask  # Maintain previous pruning
                    
                    round_pruned_weights += (~pruning_masks[name]).sum().item()  # Count newly pruned weights

                    param.data *= pruning_masks[name]

                    # restore only  unpruned weights to their OG values
                    param.data[pruning_masks[name]] = initial_state[name][pruning_masks[name]]

            total_pruned_weights += round_pruned_weights
            print(f"Num zero weights before training: {count_zero_weights(model)}")

        def train_with_mask(model, trainloader, criterion, optimizer, epochs, pruning_masks):
            model.train()
            for _ in range(epochs):
                for inputs, targets in trainloader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    # reapply pruning masks to ensure pruned weights remain zero
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            if name in pruning_masks:
                                param.data *= pruning_masks[name]

        train_with_mask(model, trainloader, criterion, optimizer, epochs, pruning_masks)

        print(f"Num zero weights after training: {count_zero_weights(model)}")

        pruned_accuracy = evaluate(model, testloader)
        print(f"Accuracy after pruning round {round + 1}: {pruned_accuracy:.4f}")

    return model, total_pruned_weights




#load Mnist
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

# check if a pre-trained model exists
model_path = "mnist_model.pth"
model = MNISTModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if os.path.exists(model_path):
    print("\nLoading pre-trained model...")
    model.load_state_dict(torch.load(model_path))
    train_time = 180.68
else:
    print("\nTraining new model...")
    start_time = time.time()
    train(model, trainloader, criterion, optimizer, epochs=10)
    train_time = time.time() - start_time
    torch.save(model.state_dict(), model_path)  
    print(f"\nModel saved as {model_path}")
    print(f"Training Time: {train_time:.2f} seconds")


originalAmountWeights = count_weights(model)
print(originalAmountWeights)

#Run IMP
pruned_model, weightsPruned = iterative_magnitude_pruning2(model, trainloader, testloader, prune_rounds=3, prune_ratio=0.2, epochs=5)
print(f"\nPercent of model pruned: {weightsPruned/originalAmountWeights * 100}%")

torch.save(model.state_dict(), "pruned_model.pth")  #saving model
print(f"\nModel saved as pruned_model.pth")

zeroed_weights = count_zero_weights(pruned_model)
print(f"Number of zeroed weights: {zeroed_weights}")
