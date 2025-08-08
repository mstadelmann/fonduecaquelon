import os
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from fdq.misc import DictToObj


def create_cache_dir(cache_dir: str) -> None:
    """Create the cache directory if it doesn't exist."""
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)


class CachedDataset(Dataset):
    """A dataset that loads cached data from RAM for fast access."""

    def __init__(self, cache_file_path: str):
        """Initialize the cached dataset.

        Args:
            cache_file_path: Path to the cached .h5 file
        """
        self.cache_file_path = cache_file_path
        # Load the data into memory for fast access
        with h5py.File(cache_file_path, "r") as f:
            # Load metadata from attributes
            if "num_samples" in f.attrs:
                self.num_samples = f.attrs["num_samples"]
            else:
                # Fallback: count groups
                self.num_samples = len([k for k in f.keys() if k.startswith("sample_")])

            # Pre-load all data into memory
            self.cached_data = []
            for i in range(self.num_samples):
                sample_group = f[f"sample_{i}"]
                sample = self._load_sample_from_group(sample_group)
                self.cached_data.append(sample)

    def _load_sample_from_group(self, group):
        """Load a sample from an HDF5 group."""
        if "type" in group.attrs:
            sample_type = group.attrs["type"]

            if sample_type == "dict":
                sample = {}
                for key in group.keys():
                    if key.endswith("_data"):
                        original_key = key[:-5]  # Remove '_data' suffix
                        value = group[key][:]
                        # Convert back to tensor
                        sample[original_key] = torch.from_numpy(value)

                # Load non-array attributes
                for attr_name in group.attrs.keys():
                    if attr_name != "type":
                        sample[attr_name] = group.attrs[attr_name]
                return sample

            elif sample_type == "tuple":
                items = []
                # Get the number of items from attributes
                num_items = group.attrs.get("num_items", 0)
                for i in range(num_items):
                    if f"item_{i}_data" in group:
                        value = group[f"item_{i}_data"][:]
                        items.append(torch.from_numpy(value))
                    else:
                        # Get scalar value from attributes
                        attr_name = f"item_{i}_value"
                        if attr_name in group.attrs:
                            items.append(group.attrs[attr_name])
                return tuple(items)

            elif sample_type == "list":
                items = []
                # Get the number of items from attributes
                num_items = group.attrs.get("num_items", 0)
                for i in range(num_items):
                    if f"item_{i}_data" in group:
                        value = group[f"item_{i}_data"][:]
                        items.append(torch.from_numpy(value))
                    else:
                        # Get scalar value from attributes
                        attr_name = f"item_{i}_value"
                        if attr_name in group.attrs:
                            items.append(group.attrs[attr_name])
                return items

            elif sample_type == "tensor":
                value = group["data"][:]
                return torch.from_numpy(value)

            elif sample_type == "other":
                # Get the value from attributes
                return group.attrs.get("value", None)

        # Fallback for older format or unknown type
        if "data" in group:
            return torch.from_numpy(group["data"][:])
        return None

    def __len__(self):
        """Return the number of cached samples."""
        return len(self.cached_data)

    def __getitem__(self, idx):
        """Return the cached sample at the given index."""
        return self.cached_data[idx]


def cache_datasets(experiment, processor, args, data_name, data_source):
    """Cache dataset to disk and return a RAM-based dataset.

    Args:
        experiment: The experiment object
        processor: Data processor object
        args: Arguments for dataset creation
        data_name: Name of the dataset
        data_source: Data source configuration

    Returns:
        DictToObj: Updated data object with cached dataloaders
    """
    data = DictToObj(processor.create_datasets(experiment, args))

    cache_dir = data_source.caching.cache_dir
    create_cache_dir(cache_dir)

    # Define cache file paths for each split
    cache_files = {
        "train": os.path.join(cache_dir, f"{data_name}_train_cache.h5"),
        "val": os.path.join(cache_dir, f"{data_name}_val_cache.h5"),
        "test": os.path.join(cache_dir, f"{data_name}_test_cache.h5"),
    }

    # Define which dataloaders to cache
    loaders_to_cache = {
        "train": data.train_data_loader,
        "val": data.val_data_loader if hasattr(data, "val_data_loader") else None,
        "test": data.test_data_loader if hasattr(data, "test_data_loader") else None,
    }

    cached_loaders = {}

    # Cache each split
    for split_name, dataloader in loaders_to_cache.items():
        if dataloader is None:
            print(f"No {split_name} dataloader found, skipping...")
            continue

        cache_file_path = cache_files[split_name]

        # Check if cache already exists
        if not os.path.exists(cache_file_path):
            print(f"Caching {split_name} dataset to {cache_file_path}...")
            cached_samples = _cache_dataloader(dataloader, split_name)

            # Save cached data to disk
            _save_samples_to_hdf5(cached_samples, cache_file_path)
            print(
                f"{split_name.capitalize()} dataset cached successfully! {len(cached_samples)} samples saved."
            )
        else:
            print(
                f"Cache file already exists at {cache_file_path}, loading {split_name} from cache..."
            )

        # Create cached dataset that loads data into RAM
        cached_dataset = CachedDataset(cache_file_path)

        # Create new DataLoader with cached dataset
        cached_loader = DataLoader(
            cached_dataset,
            batch_size=dataloader.batch_size,
            shuffle=_get_shuffle_setting(dataloader, split_name),
            num_workers=0,  # No need for workers since data is in RAM
            pin_memory=False,  # Data is already in memory
            drop_last=getattr(dataloader, "drop_last", False),
            sampler=None,  # Remove sampler for cached data
        )

        cached_loaders[split_name] = cached_loader

    # Update the data object with cached loaders
    if "train" in cached_loaders:
        data.train_data_loader = cached_loaders["train"]
    if "val" in cached_loaders:
        data.val_data_loader = cached_loaders["val"]
    if "test" in cached_loaders:
        data.test_data_loader = cached_loaders["test"]

    return data


def _save_samples_to_hdf5(samples, file_path):
    """Save samples to HDF5 format.

    Args:
        samples: List of samples to save
        file_path: Path to save the HDF5 file
    """
    with h5py.File(file_path, "w") as f:
        # Save metadata as root attributes
        f.attrs["num_samples"] = len(samples)

        for i, sample in enumerate(samples):
            sample_group = f.create_group(f"sample_{i}")
            _save_sample_to_group(sample, sample_group)


def _save_sample_to_group(sample, group):
    """Save a single sample to an HDF5 group.

    Args:
        sample: The sample to save
        group: HDF5 group to save to
    """
    if isinstance(sample, dict):
        group.attrs["type"] = "dict"
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                # Save numpy array directly
                group.create_dataset(f"{key}_data", data=value, compression="gzip")
            elif isinstance(value, (int, float, str, bool, np.integer, np.floating)):
                # Save scalar values as attributes
                group.attrs[key] = value
            else:
                # For complex types, convert to string representation
                group.attrs[key] = str(value)

    elif isinstance(sample, tuple):
        group.attrs["type"] = "tuple"
        group.attrs["num_items"] = len(sample)
        for i, item in enumerate(sample):
            if isinstance(item, np.ndarray):
                group.create_dataset(f"item_{i}_data", data=item, compression="gzip")
            elif isinstance(item, (int, float, str, bool, np.integer, np.floating)):
                group.attrs[f"item_{i}_value"] = item
            else:
                group.attrs[f"item_{i}_value"] = str(item)

    elif isinstance(sample, list):
        group.attrs["type"] = "list"
        group.attrs["num_items"] = len(sample)
        for i, item in enumerate(sample):
            if isinstance(item, np.ndarray):
                group.create_dataset(f"item_{i}_data", data=item, compression="gzip")
            elif isinstance(item, (int, float, str, bool, np.integer, np.floating)):
                group.attrs[f"item_{i}_value"] = item
            else:
                group.attrs[f"item_{i}_value"] = str(item)

    elif isinstance(sample, np.ndarray):
        group.attrs["type"] = "tensor"
        group.create_dataset("data", data=sample, compression="gzip")

    else:
        # For other types, store as attributes
        group.attrs["type"] = "other"
        if isinstance(sample, (int, float, str, bool, np.integer, np.floating)):
            group.attrs["value"] = sample
        else:
            group.attrs["value"] = str(sample)


def set_dataloader_workers_to_zero(dataloader):
    """Create a new DataLoader identical to the input but with num_workers=0.

    Args:
        dataloader: PyTorch DataLoader to modify

    Returns:
        DataLoader: New DataLoader with num_workers=0 and all other parameters preserved
    """
    return DataLoader(
        dataset=dataloader.dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,
        sampler=dataloader.sampler,
        # batch_sampler=dataloader.batch_sampler,
        # batch_sampler option is mutually exclusive with batch_size, shuffle, sampler, and drop_last
        num_workers=0,  # Set to 0
        collate_fn=dataloader.collate_fn,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last,
        timeout=dataloader.timeout,
        worker_init_fn=dataloader.worker_init_fn,
        multiprocessing_context=dataloader.multiprocessing_context,
        generator=dataloader.generator,
        prefetch_factor=None,  # Set to None to avoid issues with num_workers=0
        persistent_workers=dataloader.persistent_workers,
    )


def _cache_dataloader(dataloader, split_name):
    """Cache a single dataloader's data.

    Args:
        dataloader: PyTorch DataLoader to cache
        split_name: Name of the split (for progress bar)

    Returns:
        list: List of cached samples
    """
    cached_samples = []

    if dataloader.num_workers != 0:
        # If num_workers is not zero, set it to zero for caching
        print(
            "WARNING: multiple dataloader workers might cause CUDA issues during caching."
        )
        print("Setting dataloader num_workers to 0 for caching.")
        dataloader = set_dataloader_workers_to_zero(dataloader)

    # Iterate through the entire dataset and cache it
    for batch in tqdm(dataloader, desc=f"Caching {split_name} dataset"):
        # Store each sample in the batch individually
        if isinstance(batch, dict):
            # Handle dict-style batches (common for complex datasets)
            batch_size = len(next(iter(batch.values())))
            for i in range(batch_size):
                sample = {}
                for key, value in batch.items():
                    # Convert tensor to numpy array for efficient storage
                    if torch.is_tensor(value):
                        # Move to CPU first, then convert to numpy
                        tensor = value[i].cpu()
                        if not tensor.is_contiguous():
                            tensor = tensor.contiguous()
                        # Convert to numpy array
                        sample[key] = tensor.numpy()
                    else:
                        sample[key] = value[i]
                cached_samples.append(sample)
        else:
            # Handle tuple/list-style batches
            if isinstance(batch, list | tuple) and len(batch) == 2:
                inputs, targets = batch
                batch_size = len(inputs)
                for i in range(batch_size):
                    # Convert tensors to numpy arrays
                    inp = inputs[i]
                    tgt = targets[i]

                    if torch.is_tensor(inp):
                        inp = inp.cpu()
                        if not inp.is_contiguous():
                            inp = inp.contiguous()
                        inp = inp.numpy()

                    if torch.is_tensor(tgt):
                        tgt = tgt.cpu()
                        if not tgt.is_contiguous():
                            tgt = tgt.contiguous()
                        tgt = tgt.numpy()

                    cached_samples.append((inp, tgt))
            else:
                # Handle single tensor batches
                for i in range(len(batch)):
                    item = batch[i]
                    if torch.is_tensor(item):
                        item = item.cpu()
                        if not item.is_contiguous():
                            item = item.contiguous()
                        item = item.numpy()
                    cached_samples.append(item)

    return cached_samples


def _get_shuffle_setting(dataloader, split_name):
    """Determine shuffle setting for cached dataloader.

    Args:
        dataloader: Original dataloader
        split_name: Name of the split

    Returns:
        bool: Whether to shuffle the cached dataloader
    """
    # Generally, only shuffle training data
    if split_name == "train":
        # Try to get shuffle setting from original dataloader
        if hasattr(dataloader, "shuffle"):
            return dataloader.shuffle
        # If sampler exists, it handles shuffling
        elif hasattr(dataloader, "sampler") and dataloader.sampler is not None:
            return False  # Sampler handles shuffling
        else:
            return True  # Default to shuffle for training
    else:
        # Validation and test sets typically shouldn't be shuffled
        return False
