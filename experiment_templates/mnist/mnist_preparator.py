"""MNIST dataset preparation utilities for fonduecaquelon experiments."""

import os
from torch.utils.data import DataLoader, random_split
from torchvision import datasets


def create_datasets(experiment, args):
    """Creates MNIST  DataLoaders.

    Creates and returns MNIST dataset DataLoaders for training, validation, and testing,
    along with sample and batch counts, based on the experiment configuration.

    Args:
        experiment: An experiment object containing configuration for dataset preparation.
        args: An object or dictionary containing arguments and configuration options for dataset preparation.

    Returns:
        dict: A dictionary with DataLoaders and dataset statistics.
    """
    pin_mem = False if not experiment.is_cuda else args.get("pin_memory", False)
    drop_last = args.get("drop_last", True)

    if not os.path.exists(args.base_path):
        os.makedirs(args.base_path)

    transform = experiment.transformers["resize_norm_inp"]

    train_all_set = datasets.MNIST(
        args.base_path, train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(args.base_path, train=False, transform=transform)

    n_train_all = len(train_all_set)
    n_test_samples = len(test_set)

    # subsets
    subset_ratio = args.get("subset_train", 1)
    if subset_ratio < 1:
        n_subset_samples = int(n_train_all * subset_ratio)
        train_all_set, _ = random_split(
            train_all_set, [n_subset_samples, n_train_all - n_subset_samples]
        )
        n_train_all = len(train_all_set)

    subset_ratio = args.get("subset_test", 1)
    if subset_ratio < 1:
        n_subset_samples = int(n_test_samples * subset_ratio)
        test_set, _ = random_split(
            test_set, [n_subset_samples, n_test_samples - n_subset_samples]
        )
        n_test_samples = len(test_set)

    # val set = subset from train
    val_ratio = args.val_ratio
    if val_ratio is not None and val_ratio > 0:
        n_val_samples = int(n_train_all * val_ratio)
        n_train_samples = n_train_all - n_val_samples
        train_set, val_set = random_split(
            train_all_set, [n_train_samples, n_val_samples]
        )
    else:
        n_val_samples = 0
        n_train_samples = n_train_all
        train_set = train_all_set
        val_set = []

    train_loader = DataLoader(
        train_set,
        batch_size=args.train_batch_size,
        shuffle=args.shuffle_train,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.test_batch_size,
        shuffle=args.shuffle_test,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
    )

    if n_val_samples > 0:
        val_loader = DataLoader(
            val_set,
            batch_size=args.val_batch_size,
            shuffle=args.shuffle_val,
            num_workers=args.num_workers,
            pin_memory=pin_mem,
            drop_last=drop_last,
        )
    else:
        val_loader = None

    return {
        "train_data_loader": train_loader,
        "val_data_loader": val_loader,
        "test_data_loader": test_loader,
        "n_train_samples": n_train_samples,
        "n_val_samples": n_val_samples,
        "n_test_samples": n_test_samples,
        "n_train_batches": len(train_loader),
        "n_val_batches": len(val_loader) if val_loader is not None else 0,
        "n_test_batches": len(test_loader),
    }
