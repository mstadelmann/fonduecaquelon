import os
import numpy as np
import torch
import h5py
import json
import hashlib
import glob
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from fdq.misc import DictToObj
from fdq.ui_functions import iprint, wprint, eprint

FDQ_CACHE_HASH_KEY = "fdq_data_hash"


def get_file_size_mb(file_path):
    """Get file size in MB.

    Args:
        file_path: Path to the file

    Returns:
        float: File size in MB
    """
    if os.path.exists(file_path):
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    return 0.0


def print_cache_summary(cache_paths, data_name):
    """Print a summary of cache file sizes.

    Args:
        cache_paths: Dictionary of cache file paths
        data_name: Name of the dataset
    """
    iprint(f"\n=== Cache Summary for {data_name} ===")
    total_size_mb = 0.0

    for split_name, file_path in cache_paths.items():
        if os.path.exists(file_path):
            size_mb = get_file_size_mb(file_path)
            total_size_mb += size_mb

            try:
                with h5py.File(file_path, "r") as f:
                    num_samples = f.attrs.get("num_samples", 0)
                    avg_size_kb = (size_mb * 1024) / num_samples if num_samples > 0 else 0
                    iprint(
                        f"  {split_name:>5}: {size_mb:>8.2f} MB ({num_samples:>6} samples, {avg_size_kb:>6.2f} KB/sample)"
                    )
            except Exception as e:
                eprint(f"  {split_name:>5}: {size_mb:>8.2f} MB (unable to read sample count: {e})")
        else:
            iprint(f"  {split_name:>5}: File not found")

    if total_size_mb > 1024:
        iprint(f"  {'Total':>5}: {total_size_mb / 1024:>8.2f} GB")
    else:
        iprint(f"  {'Total':>5}: {total_size_mb:>8.2f} MB")
    iprint("=" * 40)


class CachedDataset(Dataset):
    """A dataset that loads cached data from RAM for fast access."""

    def __init__(self, cache_file_path: str, data_source, experiment):
        """Initialize the cached dataset.

        Args:
            cache_file_path: Path to the cached .h5 file
            data_source: Data source configuration object
            experiment: The experiment object containing class imports and configuration
        """
        self.experiment = experiment
        self.augmenter_path = data_source.caching.get("nondeterministic_transforms", {}).get("processor", None)
        if self.augmenter_path is not None:
            self.augmenter = experiment.import_class(file_path=self.augmenter_path)
        else:
            self.augmenter = None

        self.cache_file_path = cache_file_path
        with h5py.File(cache_file_path, "r") as f:
            if "num_samples" in f.attrs:
                self.num_samples = f.attrs["num_samples"]
            else:
                # Fallback: count groups
                # self.num_samples = len([k for k in f.keys() if k.startswith("sample_")])
                raise ValueError("Cache file is missing 'num_samples' attribute!")

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
                        sample[original_key] = torch.from_numpy(value)

                for attr_name in group.attrs.keys():
                    if attr_name != "type":
                        sample[attr_name] = group.attrs[attr_name]
                return sample

            elif sample_type == "tuple":
                items = []
                num_items = group.attrs.get("num_items", 0)
                for i in range(num_items):
                    if f"item_{i}_data" in group:
                        value = group[f"item_{i}_data"][:]
                        items.append(torch.from_numpy(value))
                    else:
                        attr_name = f"item_{i}_value"
                        if attr_name in group.attrs:
                            items.append(group.attrs[attr_name])
                return tuple(items)

            elif sample_type == "list":
                items = []
                num_items = group.attrs.get("num_items", 0)
                for i in range(num_items):
                    if f"item_{i}_data" in group:
                        value = group[f"item_{i}_data"][:]
                        items.append(torch.from_numpy(value))
                    else:
                        attr_name = f"item_{i}_value"
                        if attr_name in group.attrs:
                            items.append(group.attrs[attr_name])
                return items

            elif sample_type == "tensor":
                value = group["data"][:]
                return torch.from_numpy(value)

            elif sample_type == "other":
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
        try:
            sample = self.cached_data[idx]
        except IndexError:
            raise IndexError(f"Index {idx} out of bounds for cached dataset with {len(self.cached_data)} samples.")
        if self.augmenter is not None:
            return self.augmenter.augment(sample, self.experiment)
        return sample


def hash_conf(conf):
    """Create a hash from a dictionary."""
    dict_string = json.dumps(conf.to_dict(), sort_keys=True, ensure_ascii=True)
    return hashlib.md5(dict_string.encode()).hexdigest()


def find_valid_cache_file(cache_dir, data_name, split_name, expected_hash):
    """Find a cache file with the correct configuration hash.

    Args:
        cache_dir: Directory to search for cache files
        data_name: Name of the dataset
        split_name: Split name (train/val/test)
        expected_hash: Expected configuration hash

    Returns:
        str or None: Path to valid cache file or None if not found
    """
    pattern = os.path.join(cache_dir, f"{data_name}_{split_name}_*.h5")
    candidate_files = glob.glob(pattern)

    for file_path in candidate_files:
        try:
            with h5py.File(file_path, "r") as f:
                stored_hash = f.attrs.get(FDQ_CACHE_HASH_KEY, "")
                if stored_hash == expected_hash:
                    return file_path
        except Exception:
            continue

    return None


def cache_datasets_ddp_handler(experiment, processor, data_name, data_source):
    """Cache datasets with DDP synchronization.

    In distributed (DDP) runs, the main process performs caching first while
    child processes wait at a barrier. After caching is complete, child
    processes proceed to load/use the cache so all ranks return consistent
    dataloaders. Two barriers ensure deterministic ordering and shared state.

    Args:
        experiment: Experiment object providing DDP helpers and configuration.
        processor: Data processor with a ``create_datasets`` method.
        data_name: Human-readable dataset name used in messages/paths.
        data_source: Configuration object that controls caching/creation.

    Returns:
        DictToObj: Data object whose train/val/test dataloaders point to
        cached datasets.
    """
    data = None
    if experiment.is_main_process():
        if experiment.is_distributed():
            iprint(f"DDP training: caching data ...", dist_print=True)
        data = cache_datasets(experiment, processor, data_name, data_source)
    experiment.dist_barrier()

    if experiment.is_child_process():
        iprint(f"DDP training: loading cached data ...", dist_print=True)
        data = cache_datasets(experiment, processor, data_name, data_source)
    experiment.dist_barrier()

    return data


def get_loaders_to_cache(experiment, data):
    """Determine which dataloaders to cache based on experiment configuration.

    Args:
        experiment: The experiment object containing training/testing flags
        data: Data object containing train/val/test dataloaders

    Returns:
        dict: Dictionary mapping split names to dataloaders or None if not needed
    """
    is_train = experiment.inargs.train_model
    is_test = experiment.inargs.test_model_auto or experiment.inargs.test_model_ia
    return {
        "train": data.train_data_loader if hasattr(data, "train_data_loader") and is_train else None,
        "val": data.val_data_loader if hasattr(data, "val_data_loader") and is_train else None,
        "test": data.test_data_loader if hasattr(data, "test_data_loader") and is_test else None,
    }


def cache_datasets(experiment, processor, data_name, data_source):
    """Cache dataset to disk and return a RAM-based dataset.

    Args:
        experiment: The experiment object
        processor: Data processor object
        data_name: Name of the dataset
        data_source: Data source configuration

    Returns:
        DictToObj: Updated data object with cached dataloaders
    """
    data_hash = hash_conf(data_source)

    os.makedirs(data_source.caching.cache_dir, exist_ok=True)

    cache_paths = {}
    for split_name in ["train", "val", "test"]:
        existing_file = find_valid_cache_file(data_source.caching.cache_dir, data_name, split_name, data_hash)
        if existing_file:
            cache_paths[split_name] = existing_file
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cache_paths[split_name] = os.path.join(
                data_source.caching.cache_dir, f"{data_name}_{split_name}_{timestamp}.h5"
            )

    data = DictToObj(processor.create_datasets(experiment, data_source.args))
    loaders_to_cache = get_loaders_to_cache(experiment, data)
    cached_loaders = {}
    total_cache_size_mb = 0.0

    for split_name, orig_dataloader in loaders_to_cache.items():
        if orig_dataloader is None:
            wprint(f"No '{split_name}' dataloader found/required for this run, skipping...")
            continue

        # Check if cache with correct hash already exists
        if os.path.exists(cache_paths[split_name]):
            try:
                with h5py.File(cache_paths[split_name], "r") as f:
                    stored_hash = f.attrs.get(FDQ_CACHE_HASH_KEY, "")
                    if stored_hash == data_hash:
                        file_size_mb = get_file_size_mb(cache_paths[split_name])
                        iprint(
                            f"Cache file already exists at {cache_paths[split_name]}, loading {split_name} from file..."
                        )
                        iprint(f"Existing cache file size: {file_size_mb:.2f} MB")
                        cache_exists = True
                    else:
                        iprint("Cache file exists but configuration hash mismatch. Creating new cache...")
                        cache_exists = False
            except Exception:
                wprint("Cache file exists but is corrupted. Creating new cache...")
                cache_exists = False
        else:
            cache_exists = False

        if not cache_exists:
            iprint(f"Caching {split_name} dataset to {cache_paths[split_name]}...")
            cached_samples = cache_dataloader(orig_dataloader, split_name)

            _save_samples_to_hdf5(
                samples=cached_samples,
                file_path=cache_paths[split_name],
                config_hash=data_hash,
                compression=data_source.caching.get("compress_cache", True),
            )

            file_size_mb = get_file_size_mb(cache_paths[split_name])
            iprint(
                f"{split_name.capitalize()} dataset cached successfully! {len(cached_samples)} samples saved. Cache file size: {file_size_mb:.2f} MB"
            )
        else:
            file_size_mb = get_file_size_mb(cache_paths[split_name])

        total_cache_size_mb += file_size_mb

        # Create cached dataset that loads data into RAM
        cached_dataset = CachedDataset(cache_paths[split_name], data_source, experiment)

        # Create new DataLoader with cached dataset
        # - Set to num_workers = 0 since data is already in RAM, avoiding multiprocessing overhead
        # - no need to PIN memory as data is in RAM
        # - shuffling is managed by sampler

        cached_loader = DataLoader(
            dataset=cached_dataset,
            batch_size=orig_dataloader.batch_size,
            shuffle=False,
            # batch_sampler=orig_dataloader.batch_sampler,
            sampler=orig_dataloader.sampler,
            num_workers=data_source.caching.get("num_workers", 0),
            collate_fn=orig_dataloader.collate_fn,
            pin_memory=data_source.caching.get("pin_memory", False),
            drop_last=orig_dataloader.drop_last,
            timeout=orig_dataloader.timeout,
            worker_init_fn=orig_dataloader.worker_init_fn,
            multiprocessing_context=orig_dataloader.multiprocessing_context,
            generator=orig_dataloader.generator,
            prefetch_factor=data_source.caching.get("prefetch_factor", None),
            persistent_workers=orig_dataloader.persistent_workers,
        )

        cached_loaders[split_name] = cached_loader

    # Update the data object with cached loaders
    if "train" in cached_loaders:
        data.train_data_loader = cached_loaders["train"]
    if "val" in cached_loaders:
        data.val_data_loader = cached_loaders["val"]
    if "test" in cached_loaders:
        data.test_data_loader = cached_loaders["test"]

    # Print detailed cache summary
    print_cache_summary(cache_paths, data_name)

    return data


def _save_samples_to_hdf5(samples, file_path, config_hash=None, compression=True):
    """Save samples to HDF5 format.

    Args:
        samples: List of samples to save
        file_path: Path to save the HDF5 file
        config_hash: Configuration hash to store as attribute
        compression: Apply gzip compression when saving data
    """
    with h5py.File(file_path, "w") as f:
        # Save metadata as root attributes
        f.attrs["num_samples"] = len(samples)
        if config_hash:
            f.attrs[FDQ_CACHE_HASH_KEY] = config_hash

        for i, sample in enumerate(samples):
            sample_group = f.create_group(f"sample_{i}")
            _save_sample_to_group(sample, sample_group, compression)


def _save_sample_to_group(sample, group, compression):
    """Save a single sample to an HDF5 group.

    Args:
        sample: The sample to save
        group: HDF5 group to save to
        compression: Apply gzip compression when saving data
    """
    compression_algo = "gzip" if compression else None
    if isinstance(sample, dict):
        group.attrs["type"] = "dict"
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                group.create_dataset(f"{key}_data", data=value, compression=compression_algo)
            elif isinstance(value, int | float | str | bool | np.integer | np.floating):
                group.attrs[key] = value
            else:
                group.attrs[key] = str(value)

    elif isinstance(sample, tuple):
        group.attrs["type"] = "tuple"
        group.attrs["num_items"] = len(sample)
        for i, item in enumerate(sample):
            if isinstance(item, np.ndarray):
                group.create_dataset(f"item_{i}_data", data=item, compression=compression_algo)
            elif isinstance(item, int | float | str | bool | np.integer | np.floating):
                group.attrs[f"item_{i}_value"] = item
            else:
                group.attrs[f"item_{i}_value"] = str(item)

    elif isinstance(sample, list):
        group.attrs["type"] = "list"
        group.attrs["num_items"] = len(sample)
        for i, item in enumerate(sample):
            if isinstance(item, np.ndarray):
                group.create_dataset(f"item_{i}_data", data=item, compression=compression_algo)
            elif isinstance(item, int | float | str | bool | np.integer | np.floating):
                group.attrs[f"item_{i}_value"] = item
            else:
                group.attrs[f"item_{i}_value"] = str(item)

    elif isinstance(sample, np.ndarray):
        group.attrs["type"] = "tensor"
        group.create_dataset("data", data=sample, compression=compression_algo)

    else:
        group.attrs["type"] = "other"
        if isinstance(sample, int | float | str | bool | np.integer | np.floating):
            group.attrs["value"] = sample
        else:
            group.attrs["value"] = str(sample)


def reconfig_orig_dataloader(dataloader):
    """Create a new DataLoader with modified parameters.

    - num_workers = recommended 0 to avoid CUDA issues.
    - batch size = 1 to avoid missing samples (drop last)
    - shuffle = False to ensure consistent ordering in cache file.
    - drop_last = False (no influence with batch size = 1)

    Args:
        dataloader: PyTorch DataLoader to modify

    Returns:
        DataLoader: New DataLoader shuffle=False and BatchSize=1
    """
    if dataloader.num_workers != 0:
        wprint(
            f"WARNING: num_workers is set to {dataloader.num_workers}. If you encounter issues during data caching, try setting it to 0."
        )
    return DataLoader(
        dataset=dataloader.dataset,
        batch_size=1,
        shuffle=False,
        # sampler=dataloader.sampler,
        # batch_sampler=dataloader.batch_sampler,
        # batch_sampler option is mutually exclusive with batch_size, shuffle, sampler, and drop_last
        num_workers=dataloader.num_workers,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=dataloader.timeout,
        worker_init_fn=None,
        multiprocessing_context=None,
        generator=None,
        prefetch_factor=None,  # Set to None to avoid issues with num_workers=0
        persistent_workers=False,
    )


def cache_dataloader(dataloader, split_name):
    """Cache a single dataloader's data.

    Args:
        dataloader: PyTorch DataLoader to cache
        split_name: Name of the split (for progress bar)

    Returns:
        list: List of cached samples
    """
    cached_samples = []

    # Iterate through the entire dataset and cache it
    for batch in tqdm(reconfig_orig_dataloader(dataloader), desc=f"Caching {split_name} dataset"):
        if isinstance(batch, dict):
            batch_size = len(next(iter(batch.values())))
            for i in range(batch_size):
                sample = {}
                for key, value in batch.items():
                    if torch.is_tensor(value):
                        tensor = value[i].cpu()
                        if not tensor.is_contiguous():
                            tensor = tensor.contiguous()
                        sample[key] = tensor.numpy()
                    else:
                        sample[key] = value[i]
                cached_samples.append(sample)
        elif isinstance(batch, list):
            new_elt = []
            for item in batch:
                if torch.is_tensor(item):
                    item = item.cpu()
                    if not item.is_contiguous():
                        item = item.contiguous()
                    item = item.squeeze(0)  # remove batch dimension
                    item = item.numpy()
                    new_elt.append(item)
                else:
                    raise ValueError(f"Catching for list item type: {type(item)} is currently not implemented!")
            cached_samples.append(new_elt)
        elif isinstance(batch, torch.Tensor):
            batch = batch.cpu()
            if not batch.is_contiguous():
                batch = batch.contiguous()
            batch = batch.squeeze(0)  # remove batch dimension
            batch = batch.numpy()
            cached_samples.append(batch)
        else:
            raise ValueError(f"Catching for batch type: {type(batch)} is currently not implemented!")
        # else:
        #     # Handle tuple/list-style batches
        #     if isinstance(batch, list | tuple) and len(batch) == 2:
        #         inputs, targets = batch
        #         batch_size = len(inputs)
        #         for i in range(batch_size):
        #             # Convert tensors to numpy arrays
        #             inp = inputs[i]
        #             tgt = targets[i]

        #             if torch.is_tensor(inp):
        #                 inp = inp.cpu()
        #                 if not inp.is_contiguous():
        #                     inp = inp.contiguous()
        #                 inp = inp.numpy()

        #             if torch.is_tensor(tgt):
        #                 tgt = tgt.cpu()
        #                 if not tgt.is_contiguous():
        #                     tgt = tgt.contiguous()
        #                 tgt = tgt.numpy()

        #             cached_samples.append((inp, tgt))

    return cached_samples
