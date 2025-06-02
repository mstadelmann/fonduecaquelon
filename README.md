# FDQ | Fonduecaquelon

If you’d rather enjoy a delicious fondue than waste time on repetitive PyTorch boilerplate, this project is for you. FDQ streamlines your deep learning workflow so you can focus on what matters—your experiments, not your setup.

https://github.com/mstadelmann/fonduecaquelon

https://pypi.org/project/fdq/

## SETUP


```bash
pip install fdq
```

or

```bash
git clone https://github.com/mstadelmann/fonduecaquelon.git
cd fonduecaquelon
pip install -e .
```

## USAGE

### Local
All experiment parameters must be stored in a [config file](experiment_templates/mnist/mnist_class_dense.json). Note that config files can be based on a [parent file](experiment_templates/mnist/mnist_parent.json).

```bash
fdq <path_to_config_file.json>
```

### Slum Cluster
If you want to run your experiment on a Slurm cluster, you have to add the `slurm_cluster` section, check [here](experiment_templates/segment_pets/segment_pets.json) for an example.

```bash
python <path_to>/fdq_submit.py <path_to_config_file.json>
```

## Configuration
To run an experiment with FDQ, you need to define your [experiment loop](experiment_templates/segment_pets/train_oxpets.py), a [data-loader](experiment_templates/segment_pets/oxfordpet_preparator.py) and, optionally, a [test loop](experiment_templates/segment_pets/oxpets_test.py). 

### Model(s)
The model can either be a pre-installed one—such as [Chuchichaestli](https://github.com/CAIIVS/chuchichaestli) — or a custom model that you define and import yourself. Models, losses, and data loaders are always defined as dictionaries. For example, the following configuration:

```json
"models": {
    "ccUNET": {
        "class_name": "chuchichaestli.models.unet.unet.UNet"
    }
}
```

allows you to access the model in your training loop via `experiment.models["ccUNET"]`. The same dictionary-based structure applies to losses and data loaders as well. This setup enables you to define and manage as many models, losses, and data loaders as needed for your experiment.

### Data-Loader(s)
The Data-Loader class must provide a function `create_datasets(experiment, args)` which is expected to return a dictionary, e.g.
```json
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
```
These values can then be accessed from your training loop.

### Train
Define the path to your training loop in the train section. FDQ expects this file to provide the following function:

```python
def fdq_train(experiment: fdqExperiment):
```

Within the training loop, you can access the training arguments, model, and data as follows:

```python
nb_epochs = experiment.exp_def.train.args.epochs
data_loader = experiment.data["OXPET"].train_data_loader
model = experiment.models["ccUNET"]
```

See [train_oxpets.py](experiment_templates/segment_pets/train_oxpets.py) for an example.


### Test
Similar to the training loop, the test loop can be defined in a custom file (this can also be the same file as the training loop). FDQ expects the specified file to provide the following function:

```python
def fdq_test(experiment: fdqExperiment):
```

See [oxpets_test.py](experiment_templates/segment_pets/oxpets_test.py) for an example.

### Install additional pip packages
If your experiment needs extra Python packages, you can install them on the worker by specifying them in your configuration. Just add the required package names (and optionally the version) under a `additional_pip_packages` section in your experiment setup file. This ensures all dependencies are installed automatically before your code runs.

Example:

```json
    "slurm_cluster": {
        "fdq_version": "0.0.7",
        "...": "...",
        "additional_pip_packages": [
            "monai==1.4.0",
            "prettytable"
        ]
    }
```