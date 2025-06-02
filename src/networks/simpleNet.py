import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn


class simpleNet(nn.Module):
    """A simple fully connected neural network for classification tasks."""

    def __init__(
        self,
        nb_in_channels: int = 1,
        input_shape: list[int] = [28, 28],
        nodes_per_layer: list[int] = [84, 50],
        nb_out_channels: int = 10,
    ) -> None:
        """Initialize the simpleNet neural network.

        Args:
            nb_in_channels (int): Number of input channels.
            input_shape (list): Shape of the input (height, width).
            nodes_per_layer (list): List with the number of nodes in each hidden layer.
            nb_out_channels (int): Number of output channels/classes.
        """
        super().__init__()

        self.nb_in_channels: int = nb_in_channels
        self.nb_out_channels: int = nb_out_channels
        self.input_shape: list[int] = input_shape
        self.in_shape_flat: int = input_shape[0] * input_shape[1] * self.nb_in_channels

        self.fc_start: nn.Linear = nn.Linear(self.in_shape_flat, nodes_per_layer[0])

        # make a cleaner version of this with nn.Sequential(...)
        self.fcvar: nn.ModuleList = nn.ModuleList(
            [
                nn.Linear(nodes_per_layer[i], nodes_per_layer[i + 1])
                for i in range(len(nodes_per_layer) - 1)
            ]
        )

        self.fc_end: nn.Linear = nn.Linear(nodes_per_layer[-1], self.nb_out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # can we rewrite this as = x.view(-1).unsqueeze(0) ?
        x = x.reshape(-1, self.in_shape_flat).type(torch.float32)
        x = F.relu(self.fc_start(x))
        for layer in self.fcvar:
            x = F.relu(layer(x))

        # make and example with nn.LogSoftmax() and nn.NLLLoss() function
        # x = F.softmax(self.fc3(x)) # softmax is included in cross_entropy!
        x = self.fc_end(x)
        return x

    def example(self) -> torch.Tensor:
        """Generate a random tensor example input for the network."""
        return torch.rand(1, self.nb_in_channels, 165, 270)


# def create(experiment):
#     nodes_per_layer = experiment.exp_file.get("net_conf", {}).get("fc_Layers", [84, 50])

#     return simpleNet(experiment, nodes_per_layer)


###########

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SimpleNet(nn.Module):
#     def __init__(self, experiment, nodes_per_layer):
#         super().__init__()

#         margs = experiment.exp_def.models.simpleNet.args

#         self.nb_in_channels = margs.nb_in_channels
#         self.in_shape_flat = margs.input_shape[0] * margs.input_shape[1] * self.nb_in_channels

#         layers = [nn.Flatten(), nn.Linear(self.in_shape_flat, nodes_per_layer[0]), nn.ReLU()]
#         for in_features, out_features in zip(nodes_per_layer[:-1], nodes_per_layer[1:]):
#             layers.extend([nn.Linear(in_features, out_features), nn.ReLU()])

#         layers.append(nn.Linear(nodes_per_layer[-1], margs.nb_out_channels))
#         layers.append(nn.LogSoftmax(dim=1))  # For use with nn.NLLLoss()

#         self.model = nn.Sequential(*layers)

#     def forward(self, x):
#         x = x.type(torch.float32)
#         return self.model(x)

#     def example(self):
#         return torch.rand(1, self.nb_in_channels, 165, 270)
