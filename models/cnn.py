import torch
from torch import nn
from torch import flatten
import numpy as np


class CNN(nn.Module):
    """Convolutional Neural Network
    """
    def __init__(self):
        super().__init__()
        img_shape = 28  # initial image size
        kernel_size = 3  # kernel size for convolutional layers
        out_channels = [8]  # list of features in conv layers

        # make sure that linear layers always get the right amount of neurons
        for _ in range(len(out_channels)):
            output_shape = img_shape - kernel_size + 1
            img_shape = output_shape
        out_shape = img_shape
        linear_input = out_shape*out_shape*out_channels[-1]

        # create a stack of convolutions
        self.conv_layers = nn.Sequential(
            self.conv(1, out_channels[0], kernel_size)
        )
        if len(out_channels) > 1:
            for i in np.arange(1, len(out_channels)):
                self.conv_layers.append(
                    self.conv(
                        out_channels[i-1],
                        out_channels[i],
                        kernel_size))

        # create a stack of linear layers
        self.layer_stack = nn.Sequential(
            nn.Linear(linear_input, 1024),
            nn.Linear(1024, 10))

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
