from ast import mod
import torch
import os
import torch.nn as nn
import torch.optim as optim

from torch import nn
# from torch.utils.data import DataLoader

# from torchvision import datasets, transforms
# define a model using PyTorch's nn module
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# create an instance of the model
model = MyModel()

# define a loss function using PyTorch's nn module
criterion = nn.MSELoss()

# define an optimizer
# optimizer = optim.SGD(mod   el.parameters(), lr=0.01)

# move the model and data to the GPU if available
if torch.cuda.is_available():
    model.cuda()
    criterion.cuda()

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Is torch using CUDNN? torch.backends.cudnn.enabled")
print(f"Using {device} device")
print(torch.__version__)