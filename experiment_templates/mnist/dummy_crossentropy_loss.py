import torch


class DummyCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """Dummy CrossEntropyLoss class that inherits from torch.nn.CrossEntropyLoss. This class is only used to showcase how custom losses can be defined in the fdq framework."""

    def __init__(self, *args, **kwargs):
        """Initialize the DummyCrossEntropyLoss class.

        Args:
            *args: Positional arguments for the base class.
            **kwargs: Keyword arguments for the base class.
        """
        super().__init__(*args, **kwargs)
