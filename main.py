# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define the LSTM model for a regression problem
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyModel, self).__init__()
        self.hidden_size = hidden_size

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size)

        # Output layer
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        # Pass the input through the LSTM layer
        output, hidden = self.lstm(input, hidden)

        # Pass the output of the LSTM layer through the output layer
        output = self.linear(output.view(output.size(0), -1))

        return output, hidden

    def initHidden(self, batch_size):
        # Initialize the hidden state for the LSTM layer
        return torch.zeros(1, batch_size, self.hidden_size)

# Define the training function
def train(model, criterion, optimizer, input, target):
    # Initialize the hidden state for the LSTM layer
    hidden = model.initHidden(len(input))

    # Set the model to training mode
    model.train()

    # Set the gradients to zero for each parameter
    optimizer.zero_grad()

    # Pass each input through the model
    for i in range(input.size(0)):
        output, hidden = model(input[i], hidden)

    # Calculate the loss
    loss = criterion(output, target)

    # Calculate the gradients for each parameter
    loss.backward()

    # Update the parameters using the gradients and the optimizer
    optimizer.step()

    # Return the loss value
    return loss.item()

# Define the testing function
def test(model, criterion, input, target):
    # Initialize the hidden state for the LSTM layer
    hidden = model.initHidden(len(input))

    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        # Pass each input through the model
        for i in range(input.size(0)):
            output, hidden = model(input[i], hidden)

    # Calculate the loss
    loss = criterion(output, target)

    # Return the loss value
    return loss.item()

# Define the function to plot model weights
def plot_model_weights(model):
    for name, param in model.named_parameters():
        plt.figure()
        plt.title(f"Layer: {name}")

        if len(param.size()) == 2:
            plt.imshow(param.data.numpy(), cmap='viridis', aspect='auto')
        elif len(param.size()) == 1:
            plt.plot(param.data.numpy())

        plt.colorbar()
        plt.show()

# Define the function to visualize model output
def visualize_model_output(output):
    plt.figure()
    plt.plot(output.detach().numpy())
    plt.xlabel('Timestep')
    plt.ylabel('Output')
    plt.title('Model Output')
    plt.show()

# Create an instance of the model
model = MyModel(input_size=50, hidden_size=100, output_size=1)

# Define a loss function using PyTorch's nn module
criterion = nn.MSELoss()

# Define an optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Move the model and data to the GPU if available
if torch.cuda.is_available():
    model.cuda()
    criterion.cuda()

# Determine the device to use
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Is torch using CUDNN? {torch.backends.cudnn.enabled}")
print(f"Using {device} device")
print(torch.__version__)

# Define your input and target data
input = torch.randn(10, 1, 50)
target = torch.randn(10, 1, 1)

# Train and test the model
loss = train(model, criterion, optimizer, input, target)
print(f"Loss after training: {loss}")

# Plot model weights
plot_model_weights(model)

# Visualize model output
output, _ = model(input[0], model.initHidden(1))
visualize_model_output(output)