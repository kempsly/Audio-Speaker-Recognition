import random

def generate_sequence(seq_len, increase=True):
    """Generate a random sequence."""
    start = random.uniform(-10, 10)
    step = random.uniform(0.1, 1) if increase else random.uniform(-1, -0.1)
    return [start + i * step for i in range(seq_len)]

def generate_class_samples(num_samples, seq_len):
    """Generate samples for a class."""
    return [generate_sequence(seq_len, increase=True) for _ in range(num_samples)], \
           [generate_sequence(seq_len, increase=False) for _ in range(num_samples)]

#############################################################################################################
import numpy as np

# Define parameters
T = 4  # Length of the sequence
num_samples_per_class = 1000  # Number of samples per class

# Generate samples for class 0 (increasing sequence) and class 1 (decreasing sequence)
class_0_samples, class_1_samples = generate_class_samples(num_samples_per_class, T)

# Concatenate samples and labels for training set
train_samples = np.array(class_0_samples + class_1_samples)
train_labels = np.array([0] * num_samples_per_class + [1] * num_samples_per_class)

# Shuffle training data
shuffle_indices = np.random.permutation(len(train_samples))
train_samples = train_samples[shuffle_indices]
train_labels = train_labels[shuffle_indices]

# Define testing set size (e.g., 20% of the total samples)
test_set_size = int(0.2 * len(train_samples))

# Split training and testing sets
test_samples = train_samples[:test_set_size]
test_labels = train_labels[:test_set_size]
train_samples = train_samples[test_set_size:]
train_labels = train_labels[test_set_size:]

######################################################################################
# ###########################################################################
import torch
import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out
##################################################################

# Define model parameters
input_size = 1  # Input size (dimensionality of each element in the sequence)
hidden_size = 64  # Hidden size of the RNN
num_layers = 1  # Number of recurrent layers
output_size = 1  # Output size (binary classification)

# Instantiate the model
model = RNNClassifier(input_size, hidden_size, num_layers, output_size)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Convert training data to tensors
train_samples_tensor = torch.tensor(train_samples).unsqueeze(-1).float()
train_labels_tensor = torch.tensor(train_labels).unsqueeze(-1).float()

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(train_samples_tensor)
    loss = criterion(outputs, train_labels_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print progress
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
#######################################################################################33
# Convert testing data to tensors
test_samples_tensor = torch.tensor(test_samples).unsqueeze(-1).float()
test_labels_tensor = torch.tensor(test_labels).unsqueeze(-1).float()

# Evaluate the model on the testing set
with torch.no_grad():
    outputs = model(test_samples_tensor)
    predicted = (outputs > 0.5).float()
    accuracy = (predicted == test_labels_tensor).float().mean()
    print(f'Accuracy on test data: {accuracy.item():.4f}')

