def augment(sample, experiment=None):
    """Apply custom augmentations to cached dataset samples. This function is not mandatory!

    See 'experiment_templates/segment_pets/segment_pets_cached.json' for an example of how to use it.

    This function implements the FDQ dataset caching augmentation interface. It applies
    random transformations on-the-fly to samples that have already been processed
    through deterministic transformations and cached to disk.

    Args:
        sample: A sample from the cached dataset (same format as returned by the original dataloader).
        experiment: The main FDQ experiment object, which provides access to transformers and other configurations.
                    This allows you e.g. to read the current epoch or loss values, and apply transformations accordingly.

    Returns:
        dict: The sample with augmentations applied. Must maintain the same
            structure as the input sample!

    Note:
        This function is called for each sample when loaded from cache during training.
        The two-stage approach (deterministic caching + random augmentation) allows for:
        - Faster training by caching expensive deterministic operations
        - Flexible random augmentations that vary between epochs
        - Synchronized transformations across related tensors (e.g., image and mask)
    """
    sample["image"], sample["mask"] = experiment.transformers["random_vertical_flip_sync"](
        sample["image"], sample["mask"]
    )

    # random_vertical_flip_sync is defined in 'experiment_templates/segment_pets/segment_pets_01.json'
    # This function applies a synchronized vertical flip to both the image and mask tensors.
    # You can of course define your own operations directly in this script, or add additional transformations in the JSON file.
    # See 'src/fdq/transformers.py' for all available predefined transformations.

    return sample
