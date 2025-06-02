import os
import shutil
from urllib.request import urlretrieve

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm

from fdq.misc import get_subset


# based on https://github.com/qubvel-org/segmentation_models.pytorch


class OxfordPetDataset(torch.utils.data.Dataset):
    """A PyTorch Dataset for the Oxford-IIIT Pet dataset, supporting train/validation/test splits and optional binary masks."""

    def __init__(
        self,
        root,
        mode="train",
        transform_image=None,
        transform_mask=None,
        binary=False,
    ):
        """Initialize the OxfordPetDataset.

        Args:
            root (str): Root directory of the dataset.
            mode (str): One of 'train', 'valid', or 'test' to specify the dataset split.
            transform_image (callable, optional): Transformation to apply to images.
            transform_mask (callable, optional): Transformation to apply to masks.
            binary (bool, optional): If True, convert masks to binary format.
        """
        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.binary = binary
        self.to_tensor = transforms.ToTensor()
        self.transform_img = transform_image
        self.transform_mask = transform_mask

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")

        self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.filenames)

    def __getitem__(self, idx):
        """Retrieve the image and mask pair at the specified index."""
        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        image = self.to_tensor(Image.open(image_path).convert("RGB"))
        mask = torch.from_numpy(np.array(Image.open(mask_path)))

        if self.binary:
            mask = torch.where(mask == 2.0, torch.tensor(0.0, dtype=mask.dtype), mask)
            mask = torch.where(
                (mask == 1.0) | (mask == 3.0), torch.tensor(1.0, dtype=mask.dtype), mask
            )
            # add channel dimension
            mask = mask.unsqueeze(0)
        else:
            # one hot encoding
            mask = (
                torch.nn.functional.one_hot((mask - 1).long(), num_classes=3)
                .permute(2, 0, 1)
                .float()
            )

        if self.transform_img is not None:
            image = self.transform_img(image)
        if self.transform_mask is not None:
            mask = self.transform_mask(mask)

        return {"image": image, "mask": mask}

    def _read_split(self):
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath, encoding="utf8") as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames

    @staticmethod
    def download(root):
        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)


def create_datasets(experiment, args=None):
    """Creates and returns data loaders and dataset statistics for the Oxford Pet dataset based on the experiment configuration."""
    pin_mem = False if not experiment.is_cuda else args.get("pin_memory", False)
    drop_last = args.get("drop_last", True)

    if not os.path.exists(args.base_path):
        os.makedirs(args.base_path)

    annotations_path = os.path.join(args.base_path, "annotations.tar.gz")
    images_path = os.path.join(args.base_path, "images.tar.gz")
    if not (os.path.exists(annotations_path) and os.path.exists(images_path)):
        OxfordPetDataset.download(args.base_path)

    transform_img = experiment.transformers["resize_and_pad_bilinear"]
    transform_mask = experiment.transformers["resize_and_pad_nearest"]

    train_set = OxfordPetDataset(
        args.base_path,
        "train",
        transform_image=transform_img,
        transform_mask=transform_mask,
    )
    val_set = OxfordPetDataset(
        args.base_path,
        "valid",
        transform_image=transform_img,
        transform_mask=transform_mask,
    )
    test_set = OxfordPetDataset(
        args.base_path,
        "test",
        transform_image=transform_img,
        transform_mask=transform_mask,
    )

    # subsets
    train_set = get_subset(train_set, args.get("subset_train", 1))
    val_set = get_subset(val_set, args.get("subset_val", 1))
    test_set = get_subset(test_set, args.get("subset_test", 1))

    n_train = len(train_set)
    n_val = len(val_set)
    n_test = len(test_set)

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

    val_loader = DataLoader(
        val_set,
        batch_size=args.val_batch_size,
        shuffle=args.shuffle_val,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
    )

    return {
        "train_data_loader": train_loader,
        "val_data_loader": val_loader,
        "test_data_loader": test_loader,
        "n_train_samples": n_train,
        "n_val_samples": n_val,
        "n_test_samples": n_test,
        "n_train_batches": len(train_loader),
        "n_val_batches": len(val_loader) if val_loader is not None else 0,
        "n_test_batches": len(test_loader),
    }


class TqdmUpTo(tqdm):
    """A tqdm progress bar subclass that provides an 'update_to' method for use with urlretrieve reporthook."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    """Download a file from a URL to the specified filepath, showing a progress bar."""
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    """Extracts the archive at the given filepath to its containing directory if not already extracted."""
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)
