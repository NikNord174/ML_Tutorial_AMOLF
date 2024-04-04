import torch
from torch import nn


device = (
    "cuda"  # for NVidia GPU cards
    if torch.cuda.is_available()
    else "mps"  # for mac with M chips
    if torch.backends.mps.is_available()
    else "cpu"
)

# learning parameters
batch_size = 64
lr = 1e-3
epochs = 5
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD

# model parameters
version = 1.0
model_name = f'results/model_{version}.pth'
load = False  # parameter to specify loading parameters for model
load_version = 0.1
