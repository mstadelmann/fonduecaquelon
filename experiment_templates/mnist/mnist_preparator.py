import os
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

# from fdq.experiment import fdqData
# from preparator.transformers import create_transformers


def createDatasets(experiment):
    # create_transformers(experiment)

    dargs = experiment.exp_def.data.MNIST.args

    if not os.path.exists(dargs.base_path):
        os.makedirs(dargs.base_path)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_all_set = datasets.MNIST(
        dargs.base_path, train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(dargs.base_path, train=False, transform=transform)

    # val set = subset from train
    val_ratio = dargs.val_ratio
    n_train_all = len(train_all_set)
    n_test = len(test_set)
    if val_ratio is not None and val_ratio > 0:
        n_val = int(n_train_all * val_ratio)
        n_train = n_train_all - n_val
        train_set, val_set = random_split(train_all_set, [n_train, n_val])
    else:
        n_val = 0
        n_train = n_train_all
        train_set = train_all_set

    train_loader = DataLoader(
        train_set,
        batch_size=dargs.train_batch_size,
        shuffle=dargs.shuffle_train,
        num_workers=dargs.num_workers,
        pin_memory=dargs.pin_memory,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=dargs.test_batch_size,
        shuffle=dargs.shuffle_test,
        num_workers=dargs.num_workers,
        pin_memory=dargs.pin_memory,
    )

    if n_val > 0:
        val_loader = DataLoader(
            val_set,
            batch_size=dargs.val_batch_size,
            shuffle=dargs.shuffle_val,
            num_workers=dargs.num_workers,
            pin_memory=dargs.pin_memory,
        )
    else:
        val_loader = None

    # experiment.print_dataset_infos()

    return {
        "train_loader": train_loader,
        "val_data_loader": val_loader,
        "test_data_loader": test_loader,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
    }
