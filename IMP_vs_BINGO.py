import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import os
import random

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

# loading MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Function to train model with progress updates
def train_model(model, trainloader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        total_batches = len(trainloader)
        for batch_idx, (images, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Printing training progress
            if (batch_idx + 1) % 100 == 0 or batch_idx + 1 == total_batches:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{total_batches}] - Loss: {loss.item():.4f}")
    
    return model

# Checking if a pre-trained model exists
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
    model = train_model(model, trainloader, criterion, optimizer, epochs=10)
    train_time = time.time() - start_time
    torch.save(model.state_dict(), model_path)  # Save the model
    print(f"\nModel saved as {model_path}")
    print(f"Training Time: {train_time:.2f} seconds")

original_accuracy = evaluate(model, testloader)
accuracy_threshold = 0.8 * original_accuracy
print(f"\nOriginal Model Accuracy: {(original_accuracy * 100):.2f}%")
print(f"Accuracy threshold for pruning: {(accuracy_threshold * 100):.2f}%")


def ticket_search(model, trainloader, percentage_reinitialize=0.01):
    model.eval()  # Set the model to evaluation mode
    
    # Initialize significance scores
    significance_scores = {name: torch.zeros_like(param) for name, param in model.named_parameters() if "weight" in name}

    # Loop over all batches in the trainloader
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(trainloader):
            # Real training pass (accuracy before pruning)
            outputs_before = model(images)
            _, predicted_before = torch.max(outputs_before, 1)
            accuracy_before = (predicted_before == labels).float().mean().item()

            print(f"Batch {batch_idx + 1}/{len(trainloader)}: Real pass accuracy: {accuracy_before * 100}%")

            # Calculate significance for each layer
            for name, param in model.named_parameters():
                if "weight" in name:
                    original_weights = param.clone()  # storing the original weights
                    
                    # 1: Create a mask to "turn off" a random 1% of neurons
                    mask = torch.ones_like(param)
                    num_neurons_to_turn_off = int(param.numel() * percentage_reinitialize)
                    indices_to_turn_off = random.sample(range(param.numel()), num_neurons_to_turn_off)
                    mask.flatten()[indices_to_turn_off] = 0  # Set the 1% of neurons to 0
                    
                    # apply the mask to the weights (turn off neurons/weights)
                    param.data *= mask.float()

                    # Print how many neurons were turned off
                    print(f"Layer: {name} | Turning off {num_neurons_to_turn_off} neurons (1% of layer size).")

                    # fake training pass (accuracy after pruning)
                    outputs_after = model(images)
                    _, predicted_after = torch.max(outputs_after, 1)
                    accuracy_after = (predicted_after == labels).float().mean().item()

                    print(f"Layer: {name} | Fake pass accuracy (after turning off neurons): {accuracy_after * 100}%")

                    # 3: Calculate accuracy loss
                    accuracy_loss = abs(accuracy_before - accuracy_after)
                    print(f"Layer: {name} | Accuracy loss: {accuracy_loss * 100}%")

                    # 4: Updating significance scores only for the neurons that were turned off (new significance score calculation should be made)
                    for idx in indices_to_turn_off:
                        # Convert the index to its corresponding position in the tensor
                        param_flat = param.flatten()
                        # updating significance score for the neuron that was turned off
                        significance_scores[name].flatten()[idx] = (significance_scores[name].flatten()[idx] + accuracy_loss) / 2
                    
                    # restoring the original weights after the fake "training" pass
                    param.data = original_weights

            # printing out significance score updates for each layer after the batch
            for name, scores in significance_scores.items():
                max_significance = scores.max().item()
                min_significance = scores.min().item()
                print(f"Layer: {name} | Highest significance: {max_significance}")
                print(f"Layer: {name} | Lowest significance: {min_significance}")
                print("-" * 50) 

    return significance_scores



def prune_model(model, significance_scores, accuracy_threshold, prune_step=0.01, evaluate_fn=evaluate, testloader=None):
    total_zeroed_neurons = 0  

    total_neurons = sum(param.numel() for name, param in model.named_parameters() if "weight" in name)

    original_accuracy = evaluate_fn(model, testloader)  
    print(f"Original accuracy: {original_accuracy * 100}%")

    prune_ratio = 0  # Start pruning from 0% -- number will go up over time
    
    while prune_ratio <= 1:  # Stopping when prune_ratio reaches the target 
        total_neurons_to_prune = int(prune_ratio * total_neurons)

        print(f"Pruning {prune_ratio * 100}% of neurons ({total_neurons_to_prune} neurons)")

        for name, param in model.named_parameters():
            if "weight" in name:
                # getting the number of neurons that exist in this layer
                layer_neurons = param.numel()

                layer_neurons_to_prune = int(total_neurons_to_prune * (layer_neurons / total_neurons))
                print(f"{layer_neurons_to_prune} neurons to prune in {name}")

                # sorting significance scores
                flattened_scores = significance_scores[name].flatten()
                sorted_scores, sorted_indices = torch.sort(flattened_scores)

                neurons_to_prune_indices = sorted_indices[:layer_neurons_to_prune]

                mask = torch.ones_like(param)
                mask.flatten()[neurons_to_prune_indices] = 0  # Set the neurons with lowest significance to 0

                # count the num of how many neurons were zeroed in this layer
                zeroed_neurons_in_layer = (mask == 0).sum().item()
                total_zeroed_neurons += zeroed_neurons_in_layer
                print(f"Zeroing {zeroed_neurons_in_layer} neurons in {name}")

                # apply the mask to zero out neurons in this layer
                param.data *= mask.float()

        pruned_accuracy = evaluate_fn(model, testloader)
        print(f"Accuracy after pruning: {pruned_accuracy * 100}%")

        if pruned_accuracy < accuracy_threshold * original_accuracy:
            print(f"Accuracy dropped below {accuracy_threshold * 100}% of the original accuracy. Stopping pruning.")
            break

        prune_ratio += prune_step

    return total_zeroed_neurons





# Iterative Magnitude Pruning (IMP) - Zeroing only
def iterative_magnitude_pruning(model, accuracy_threshold):
    total_zeroed_neurons = 0  

    for prune_ratio in torch.linspace(0.05, 1, steps=20):
        zeroed_neurons = 0 

        for name, param in model.named_parameters():
            if "weight" in name:
                threshold = torch.quantile(torch.abs(param).flatten(), prune_ratio)
                mask = torch.abs(param) > threshold
                zeroed_neurons += (mask == 0).sum().item()  
                print(f"Zeroing {zeroed_neurons} neurons in {name}")

                param.data *= mask.float()  # zero out  all theinsignificant weights
                param.data.copy_(param * mask.float()) 

        pruned_accuracy = evaluate(model, testloader)
        total_zeroed_neurons += zeroed_neurons  
        print(f"Final accuracy IMP: {pruned_accuracy * 100}")

        if pruned_accuracy < accuracy_threshold:
            break

    return total_zeroed_neurons

# Define your evaluate function
def evaluate(model, testloader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return accuracy



# Applying bingo
print("\nApplying BINGO Pruning...")
start_bingo = time.time()
significance_scores = ticket_search(model, trainloader)
bingo_pruned_neurons = prune_model(
    model,
    significance_scores,
    accuracy_threshold=0.8,
    prune_step=0.01,
    evaluate_fn=evaluate,  # Ensure evaluate function is correct
    testloader=testloader  # Pass testloader explicitly
)
bingo_time = time.time() - start_bingo

# IMP apply
print("\nApplying IMP Pruning...")
start_imp = time.time()
imp_pruned_neurons = iterative_magnitude_pruning(model, accuracy_threshold)
imp_time = time.time() - start_imp

common_pruned = min(bingo_pruned_neurons, imp_pruned_neurons)

bingo_ratio = bingo_pruned_neurons / (bingo_time * 1000)
imp_ratio = imp_pruned_neurons / (imp_time * 1000)

print("\n--- RESULTS ---")
print(f"Original Model Accuracy: {original_accuracy:.4f}")
print(f"BINGO Pruned Neurons Before 80% Accuracy Drop: {bingo_pruned_neurons}")
print(f"IMP Pruned Neurons Before 80% Accuracy Drop: {imp_pruned_neurons}")
print(f"Common Pruned Neurons (BOTH methods): {common_pruned}")
print(f"BINGO Dropped Neurons per ms: {bingo_ratio:.2f}")
print(f"IMP Dropped Neurons per ms: {imp_ratio:.2f}")
