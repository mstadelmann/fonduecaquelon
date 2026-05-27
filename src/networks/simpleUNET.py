import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn


class simpleUNET(nn.Module):
    """A simple parametrizable U-Net for dense prediction tasks (input shape == output shape)."""

    def __init__(
        self,
        in_channels: int = 1,
        input_shape: list[int] = [256, 256],
        encoder_channels: list[int] = [64, 128, 256],
        out_channels: int = 1,
        kernel_size: int = 3,
    ) -> None:
        """Initialize the simpleUNET neural network.

        Args:
            in_channels (int): Number of input channels.
            input_shape (list): Shape of the input (height, width).
            encoder_channels (list): Channel counts for each encoder stage; decoder mirrors this in reverse.
            out_channels (int): Number of output channels (e.g. number of segmentation classes).
            kernel_size (int): Kernel size for all convolutional layers.
        """
        super().__init__()

        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.input_shape: list[int] = input_shape
        pad = kernel_size // 2

        # Encoder: one conv block per stage, followed by max-pooling in forward()
        enc_in_ch = [in_channels] + encoder_channels[:-1]
        self.enc_blocks: nn.ModuleList = nn.ModuleList(
            [nn.Conv2d(enc_in_ch[i], encoder_channels[i], kernel_size=kernel_size, padding=pad)
             for i in range(len(encoder_channels))]
        )
        self.pool: nn.MaxPool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck: nn.Conv2d = nn.Conv2d(
            encoder_channels[-1], encoder_channels[-1] * 2, kernel_size=kernel_size, padding=pad
        )

        # Decoder: upsample then concat skip connection, then conv block
        dec_ch = list(reversed(encoder_channels))          # [256, 128, 64]
        dec_up_in = [encoder_channels[-1] * 2] + dec_ch[:-1]  # [512, 256, 128]
        self.dec_up: nn.ModuleList = nn.ModuleList(
            [nn.ConvTranspose2d(dec_up_in[i], dec_ch[i], kernel_size=2, stride=2)
             for i in range(len(dec_ch))]
        )
        self.dec_blocks: nn.ModuleList = nn.ModuleList(
            [nn.Conv2d(dec_ch[i] * 2, dec_ch[i], kernel_size=kernel_size, padding=pad)
             for i in range(len(dec_ch))]
        )

        # Final 1x1 projection to output channels
        self.conv_end: nn.Conv2d = nn.Conv2d(encoder_channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.type(torch.float32)

        # Encoder: save skip connections before pooling
        skips = []
        for block in self.enc_blocks:
            x = F.relu(block(x))
            skips.append(x)
            x = self.pool(x)

        # Bottleneck
        x = F.relu(self.bottleneck(x))

        # Decoder: upsample, concat skip, conv
        for up, block, skip in zip(self.dec_up, self.dec_blocks, reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = F.relu(block(x))

        x = self.conv_end(x)
        return x

    def example(self) -> torch.Tensor:
        """Generate a random tensor example input for the network."""
        return torch.rand(1, self.in_channels, *self.input_shape)
