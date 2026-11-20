import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalLayer(nn.Module):
    """A 2D convolution layer that slides filters horizontally and vertically

    Input shape : (B, C_in, H, W)
    Output shape: (B, C_out, H, W)

    The letters mean:
        B      batch size: how many images we process together
        C_in   number of input channels (1 for a grayscale image)
        C_out  number of filters (number of output channels)
        H, W   height and width of image in pixels

    kernel_size is the width and height of the filter window
    """

    # set up the filters we use to find patterns in out image
    def __init__(self, in_channels, out_channels, kernel_size=3):
        # nn.Module PyTorch base class
        super().__init__()

        # set up layer's setting for the forward function
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = kernel_size

        # pad zeros to add on each side of the image for symmetry
        self.pad = kernel_size // 2

        # weight tensor has shape: (out_channels, in_channels, k, k)
        # weight tensor is the collection of all a layer's filters
        # e.g. (32, 1, 3, 3) for the first layer: 32 filters

        # training starts from random numbers, but their size matters.
        # if they start too big, values explode as they pass through layers,
        # but if they are too small and they shrink

        # good resource for this:
        # https://www.telesens.co/2018/04/09/initializing-weights-for-the-convolutional-and-fully-connected-layers/
        n = in_channels * kernel_size * kernel_size
        std = math.sqrt(2.0 / n)

        # https://discuss.pytorch.org/t/how-to-share-weights-of-dilated-convolutional-kernels/136390/3
        # filters filled with random numbers, that training adjusts
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels,
                        kernel_size, kernel_size) * std
        )
        self.bias = nn.Parameter(torch.randn(out_channels))

    # for every image that passes through the layer, calculate the scoring for a filter
    def forward(self, x):
        """Run the layer. x has shape (B, C_in, H, W)."""

        B, C, H, W = x.shape
        k = self.k
        p = self.pad

        # pad with zeros so the filter can process pixels at the edges
        # corner pixels have 8 neighboring pixels surrounding them
        padded = F.pad(x, (p, p, p, p), mode="constant", value=0)

        # make shifted copies of each pixel multiple times that uses
        # one matrix multiplication to score everything
        # python loops would be too slow for this
        shifted = []
        for i in range(k):
            for j in range(k):
                # (batch, channel, row, column)
                shifted.append(padded[:, :, i:i + H, j:j + W])

        # stack tensors into one tensor
        stacked = torch.stack(shifted, dim=2)

        # good resource for the matrix multiplication
        # https://cs231n.github.io/convolutional-networks/

        # reshape so each filter becomes one row of numbers representing one spot's
        # eight neighbors
        cols = stacked.reshape(B, C * k * k, H * W)

        # rearrange values of filter weights into one row of eight numbers, so filter numbers align
        # with the neighbor numbers
        weight_matrix = self.weight.reshape(self.out_channels, C * k * k)

        # take each filter row and each spot column,
        # multiply the pair and add them up
        # done for every filter at every spot in a single step.
        out = weight_matrix @ cols

        # add each filter's bias
        out = out + self.bias[None, :, None]

        # fold the flat position axis back into a picture shape
        return out.reshape(B, self.out_channels, H, W)


class MaxPoolingLayer(nn.Module):
    """
    Reduces the spatial dimensions of CNN feature maps
    by selecting the maximum value in each pooling window.
    """

    def __init__(self, kernel_size=2, stride=2):
        super().__init__()

        self.pool = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride
        )

    def forward(self, x):
        return self.pool(x)

# create the CNN


# create the RNN
