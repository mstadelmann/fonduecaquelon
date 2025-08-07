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

        # Ensure tensors are contiguous for CUDA operations
        if isinstance(sample, dict):
            result = {}
            for key, value in sample.items():
                if torch.is_tensor(value):
                    result[key] = value.contiguous()
                else:
                    result[key] = value
            return result
        elif isinstance(sample, tuple | list):
            result = []
            for item in sample:
                if torch.is_tensor(item):
                    result.append(item.contiguous())
                else:
                    result.append(item)
            return tuple(result) if isinstance(sample, tuple) else result
        else:
            if torch.is_tensor(sample):
                return sample.contiguous()
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

    cache_file_path = os.path.join(cache_dir, f"{data_name}_cache.pt")

    train_data_loader = data.train_data_loader

    # Check if cache already exists
    if not os.path.exists(cache_file_path):
        print(f"Caching dataset to {cache_file_path}...")
        cached_samples = []

        # Iterate through the entire dataset and cache it
        for batch in tqdm(train_data_loader, desc="Caching dataset"):
            # Store each sample in the batch individually
            if isinstance(batch, dict):
                # Handle dict-style batches (common for complex datasets)
                batch_size = len(next(iter(batch.values())))
                for i in range(batch_size):
                    sample = {}
                    for key, value in batch.items():
                        # Ensure tensor is contiguous and on CPU
                        if torch.is_tensor(value):
                            sample[key] = value[i].cpu().contiguous().clone()
                        else:
                            sample[key] = value[i]
                    cached_samples.append(sample)
            else:
                # Handle tuple/list-style batches
                if isinstance(batch, list | tuple) and len(batch) == 2:
                    inputs, targets = batch
                    batch_size = len(inputs)
                    for i in range(batch_size):
                        # Ensure tensors are contiguous and on CPU
                        inp = inputs[i].cpu().contiguous().clone() if torch.is_tensor(inputs[i]) else inputs[i]
                        tgt = targets[i].cpu().contiguous().clone() if torch.is_tensor(targets[i]) else targets[i]
                        cached_samples.append((inp, tgt))
                else:
                    # Handle single tensor batches
                    for i in range(len(batch)):
                        item = batch[i].cpu().contiguous().clone() if torch.is_tensor(batch[i]) else batch[i]
                        cached_samples.append(item)

        # Save cached data to disk
        torch.save(cached_samples, cache_file_path)
        print(f"Dataset cached successfully! {len(cached_samples)} samples saved.")
    else:
        print(f"Cache file already exists at {cache_file_path}, loading from cache...")

    # Create cached dataset that loads data into RAM
    cached_dataset = CachedDataset(cache_file_path)

    # Create new DataLoader with cached dataset
    cached_train_loader = DataLoader(
        cached_dataset,
        batch_size=train_data_loader.batch_size,
        shuffle=hasattr(train_data_loader, "shuffle") and train_data_loader.shuffle,
        num_workers=0,  # No need for workers since data is in RAM
        pin_memory=False,  # Data is already in memory
        drop_last=train_data_loader.drop_last if hasattr(train_data_loader, "drop_last") else False,
    )

    # Update the data object with cached loader
    data.train_data_loader = cached_train_loader

    return data
