import torch
from torch import nn
from torch import flatten


class CNN(nn.Module):
    """Convolutional Neural Network
    """
    def __init__(self):
        super().__init__()
        out_channels = [2, 4, 8, 16]
        shape = 20
        self.conv_layers = nn.Sequential(
            self.conv(1, 2, 3),
            self.conv(2, 4, 3),
            self.conv(4, 8, 3),
            self.conv(8, 16, 3))
        self.layer_stack = nn.Sequential(
            nn.Linear(shape*shape*out_channels[-1], 10))

    def conv(
            self,
            input: int = 1,
            output: int = 10,
            kernel: int = 3) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels=input,
                out_channels=output,
                kernel_size=kernel,
            ),
            nn.BatchNorm2d(output),
            nn.ReLU()
        )

    def forward(self, x):
        conv_output = self.conv_layers(x)
        conv_output = flatten(conv_output, 1)
        logits = self.layer_stack(conv_output)
        return logits


if __name__ == '__main__':
    model = CNN()
    print(model)
    image = torch.rand(2, 1, 28, 28)
    output = model(image)
    print(output.shape)
