# FDQ | Fonduecaquelon

Fonduecaquelon (FDQ) is designed for researchers and practitioners who want to focus on deep learning experiments, not boilerplate code. FDQ streamlines your PyTorch workflow, automating repetitive tasks and providing a flexible, extensible framework for experiment management—so you can spend more time on innovation and less on setup.

- [GitHub Repository](https://github.com/mstadelmann/fonduecaquelon)
- [PyPI Package](https://pypi.org/project/fdq/)

---

## 🚀 Features

- **Minimal Boilerplate:** Define only what matters — FDQ handles the rest.
- **Flexible Experiment Configuration:** Use JSON config files with inheritance support for easy experiment management.
- **Multi-Model Support:** Seamlessly manage multiple models, losses, and data loaders.
- **Cluster Ready:** Effortlessly submit jobs to SLURM clusters with built-in utilities.
- **Extensible:** Easily integrate custom models, data loaders, and training/testing loops.
- **Automatic Dependency Management:** Install additional pip packages per experiment.
- **Distributed Training:** Out-of-the-box support for distributed training using PyTorch DDP.

---

## 🛠️ Installation

Install the latest release from PyPI:

```bash
pip install fdq
```

Or, for development and the latest features, clone the repository:

```bash
git clone https://github.com/mstadelmann/fonduecaquelon.git
cd fonduecaquelon
pip install -e .
```

---

## 📖 Usage

### Local Experiments

All experiment parameters are defined in a [config file](experiment_templates/mnist/mnist_class_dense.json). Config files can inherit from a [parent file](experiment_templates/mnist/mnist_parent.json) for easy reuse and organization.

To run an experiment locally:

```bash
fdq <path_to_config_file.json>
```

### SLURM Cluster Execution

To run experiments on a SLURM cluster, add a `slurm_cluster` section to your config. See [this example](experiment_templates/segment_pets/segment_pets.json).

Submit your experiment to the cluster:

```bash
python <path_to>/fdq_submit.py <path_to_config_file.json>
```

---

## ⚙️ Configuration Overview

FDQ uses JSON configuration files to define experiments. These files specify models, data loaders, training/testing scripts, and cluster settings.

### Models

Models are defined as dictionaries. You can use pre-installed models (e.g., [Chuchichaestli](https://github.com/CAIIVS/chuchichaestli)) or your own. Example:

```json
"models": {
    "ccUNET": {
        "class_name": "chuchichaestli.models.unet.unet.UNet"
    }
}
```

Access models in your training loop via `experiment.models["ccUNET"]`. The same structure applies to losses and data loaders.

### Data Loaders

Your data loader class must implement a `create_datasets(experiment, args)` function, returning a dictionary like:

```python
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

These values are accessible from your training loop as `experiment.data["<name>"].<key>`.

### Training Loop

Specify the path to your training script in the config. FDQ expects a function:

```python
def fdq_train(experiment: fdqExperiment):
```

Within this function, you can access all experiment components:

```python
nb_epochs = experiment.exp_def.train.args.epochs
data_loader = experiment.data["OXPET"].train_data_loader
model = experiment.models["ccUNET"]
```

See [train_oxpets.py](experiment_templates/segment_pets/train_oxpets.py) for a full example.

### Testing Loop

Testing works similarly. Define a function:

```python
def fdq_test(experiment: fdqExperiment):
```

See [oxpets_test.py](experiment_templates/segment_pets/oxpets_test.py) for an example.

---

## 📦 Installing Additional Python Packages

If your experiment requires extra Python packages, specify them in your config under `additional_pip_packages`. FDQ will install them automatically before running your experiment.

Example:

```json
"slurm_cluster": {
    "fdq_version": "0.0.48",
    "...": "...",
    "additional_pip_packages": [
        "monai==1.4.0",
        "prettytable"
    ]
}
```

---

## 📝 Tips

- **Config Inheritance:** Use the `parent` key in your config to inherit settings from another file, reducing duplication.
- **Multiple Models/Losses:** FDQ supports multiple models and losses per experiment — just add them to the config dictionaries.
- **Cluster Submission:** The `fdq_submit.py` utility handles SLURM job script generation and submission, including environment setup and result copying.

---

## 📚 Resources

- [Experiment Templates](experiment_templates/)
- [Example Configs](experiment_templates/mnist/)
- [Chuchichaestli Models](https://github.com/CAIIVS/chuchichaestli)

---

## 🤝 Contributing

Contributions are welcome! Please open issues or pull requests on [GitHub](https://github.com/mstadelmann/fonduecaquelon).

---

## 🧀 Enjoy your fondue and happy experimenting!