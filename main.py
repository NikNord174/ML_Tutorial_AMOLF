import torch

import constants
from models.FCM import NeuralNetwork
from train_test import train, test
from dataset import train_dataloader, test_dataloader


# import learning parameters
model = NeuralNetwork().to(constants.device)

# load model
if constants.load:
    model.load_state_dict(
        torch.load("results/model_{constants.load_version}.pth"))

loss_fn = constants.loss_fn
lr = constants.lr
optimizer = constants.optimizer(params=model.parameters(), lr=lr)

for t in range(constants.epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(
        train_dataloader,
        model,
        loss_fn,
        optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# save model
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
