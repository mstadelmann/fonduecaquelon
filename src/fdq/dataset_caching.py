import os
import torch
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
            cache_file_path: Path to the cached .pt file
        """
        self.cached_data = torch.load(cache_file_path, map_location="cpu")

    def __len__(self):
        """Return the number of cached samples."""
        return len(self.cached_data)

    def __getitem__(self, idx):
        """Return the cached sample at the given index."""
        sample = self.cached_data[idx]

        # Ensure tensors are contiguous and properly aligned for CUDA operations
        if isinstance(sample, dict):
            result = {}
            for key, value in sample.items():
                if torch.is_tensor(value):
                    # Ensure tensor is contiguous and create a new aligned tensor
                    tensor = value.contiguous()
                    # Force memory alignment by cloning if needed
                    if not tensor.is_contiguous():
                        tensor = tensor.clone()
                    result[key] = tensor
                else:
                    result[key] = value
            return result
        elif isinstance(sample, tuple | list):
            result = []
            for item in sample:
                if torch.is_tensor(item):
                    # Ensure tensor is contiguous and create a new aligned tensor
                    tensor = item.contiguous()
                    # Force memory alignment by cloning if needed
                    if not tensor.is_contiguous():
                        tensor = tensor.clone()
                    result.append(tensor)
                else:
                    result.append(item)
            return tuple(result) if isinstance(sample, tuple) else result
        else:
            if torch.is_tensor(sample):
                # Ensure tensor is contiguous and create a new aligned tensor
                tensor = sample.contiguous()
                # Force memory alignment by cloning if needed
                if not tensor.is_contiguous():
                    tensor = tensor.clone()
                return tensor
            return sample


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
        "train": os.path.join(cache_dir, f"{data_name}_train_cache.pt"),
        "val": os.path.join(cache_dir, f"{data_name}_val_cache.pt"),
        "test": os.path.join(cache_dir, f"{data_name}_test_cache.pt"),
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
            torch.save(cached_samples, cache_file_path)
            print(f"{split_name.capitalize()} dataset cached successfully! {len(cached_samples)} samples saved.")
        else:
            print(f"Cache file already exists at {cache_file_path}, loading {split_name} from cache...")

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


def _cache_dataloader(dataloader, split_name):
    """Cache a single dataloader's data.

    Args:
        dataloader: PyTorch DataLoader to cache
        split_name: Name of the split (for progress bar)

    Returns:
        list: List of cached samples
    """
    cached_samples = []

    # Iterate through the entire dataset and cache it
    for batch in tqdm(dataloader, desc=f"Caching {split_name} dataset"):
        # Store each sample in the batch individually
        if isinstance(batch, dict):
            # Handle dict-style batches (common for complex datasets)
            batch_size = len(next(iter(batch.values())))
            for i in range(batch_size):
                sample = {}
                for key, value in batch.items():
                    # Ensure tensor is contiguous, aligned, and on CPU for proper serialization
                    if torch.is_tensor(value):
                        # Move to CPU first, then ensure contiguity and proper alignment
                        tensor = value[i].cpu()
                        if not tensor.is_contiguous():
                            tensor = tensor.contiguous()
                        # Clone to ensure proper memory alignment
                        sample[key] = tensor.clone()
                    else:
                        sample[key] = value[i]
                cached_samples.append(sample)
        else:
            # Handle tuple/list-style batches
            if isinstance(batch, list | tuple) and len(batch) == 2:
                inputs, targets = batch
                batch_size = len(inputs)
                for i in range(batch_size):
                    # Ensure tensors are contiguous, aligned, and on CPU
                    inp = inputs[i]
                    tgt = targets[i]

                    if torch.is_tensor(inp):
                        inp = inp.cpu()
                        if not inp.is_contiguous():
                            inp = inp.contiguous()
                        inp = inp.clone()

                    if torch.is_tensor(tgt):
                        tgt = tgt.cpu()
                        if not tgt.is_contiguous():
                            tgt = tgt.contiguous()
                        tgt = tgt.clone()

                    cached_samples.append((inp, tgt))
            else:
                # Handle single tensor batches
                for i in range(len(batch)):
                    item = batch[i]
                    if torch.is_tensor(item):
                        item = item.cpu()
                        if not item.is_contiguous():
                            item = item.contiguous()
                        item = item.clone()
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


def cache_single_dataset(dataloader, cache_file_path, split_name="dataset"):
    """Cache a single dataloader to a specific file path.

    Args:
        dataloader: PyTorch DataLoader to cache
        cache_file_path: Path where to save the cached data
        split_name: Name of the split (for logging)

    Returns:
        DataLoader: New DataLoader with cached data
    """
    # Ensure cache directory exists
    cache_dir = os.path.dirname(cache_file_path)
    create_cache_dir(cache_dir)

    # Check if cache already exists
    if not os.path.exists(cache_file_path):
        print(f"Caching {split_name} dataset to {cache_file_path}...")
        cached_samples = _cache_dataloader(dataloader, split_name)

        # Save cached data to disk with proper serialization
        print(f"Saving {len(cached_samples)} samples to {cache_file_path}...")

        # Ensure all tensors are properly formatted before saving
        cleaned_samples = []
        for sample in cached_samples:
            if isinstance(sample, dict):
                cleaned_sample = {}
                for key, value in sample.items():
                    if torch.is_tensor(value):
                        # Ensure tensor is in the most compatible format
                        value = value.cpu().contiguous().clone()
                        # Force float32 for images and masks to ensure compatibility
                        if value.dtype in [torch.float64, torch.float16]:
                            value = value.float()
                        cleaned_sample[key] = value
                    else:
                        cleaned_sample[key] = value
                cleaned_samples.append(cleaned_sample)
            elif isinstance(sample, tuple | list):
                cleaned_sample = []
                for item in sample:
                    if torch.is_tensor(item):
                        item = item.cpu().contiguous().clone()
                        if item.dtype in [torch.float64, torch.float16]:
                            item = item.float()
                        cleaned_sample.append(item)
                    else:
                        cleaned_sample.append(item)
                cleaned_samples.append(tuple(cleaned_sample) if isinstance(sample, tuple) else cleaned_sample)
            else:
                if torch.is_tensor(sample):
                    sample = sample.cpu().contiguous().clone()
                    if sample.dtype in [torch.float64, torch.float16]:
                        sample = sample.float()
                cleaned_samples.append(sample)

        torch.save(cleaned_samples, cache_file_path)
        print(f"{split_name.capitalize()} dataset cached successfully! {len(cached_samples)} samples saved.")
    else:
        print(f"Cache file already exists at {cache_file_path}, loading {split_name} from cache...")

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
    )

    return cached_loader
