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
- **Model Export & Optimization:** Export trained models to ONNX format with optimization options.
- **High-Performance Inference:** TensorRT integration for GPU-accelerated inference with up to 10x speedup.
- **Model Compilation:** JIT tracing/scripting and torch.compile support for optimized model execution.
- **Interactive Model Dumping:** Easy-to-use interface for exporting and optimizing trained models.

---

## 🛠️ Installation

Install the latest release from PyPI:

```bash
pip install fdq
```

If you have an Nvidia GPU and want to run inference, additionally install the GPU dependencies:
```bash
pip install fdq[gpu]
```

Or, for development and the latest features, clone the repository:

```bash
git clone https://github.com/mstadelmann/fonduecaquelon.git
cd fonduecaquelon
pip install -e .[dev,gpu]
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

### Model Export and Optimization

After training, export and optimize your models for deployment:

```bash
# Interactive model dumping with export options
fdq <path_to_config_file.json> -nt -d
```

This launches an interactive interface where you can:
- **Export to ONNX:** Convert PyTorch models to ONNX format with Dynamo or TorchScript
- **JIT Compilation:** Trace or script models using PyTorch JIT
- **TensorRT Optimization:** Compile models for GPU inference with precision options (FP32, FP16, INT8)
- **Performance Benchmarking:** Compare optimized vs. original model performance

### Additional CLI Options

FDQ supports several command-line options for different workflows:

```bash
# Run only training (default)
fdq <config_file.json>

# Skip training to run other tasks
fdq <config_file.json> -nt

# Run training and automatic testing
fdq <config_file.json> -ta

# Interactive testing
fdq <config_file.json> -nt -ti

# Export and optimize models
fdq <config_file.json> -nt -d

# Run inference tests on trained models
fdq <config_file.json> -nt -i

# Print model architecture before training
fdq <config_file.json> -p

# Resume training from checkpoint
fdq <config_file.json> -rp /path/to/checkpoint
```

---

## 🚄 Model Export & Deployment

FDQ provides comprehensive model export and optimization capabilities for deployment:

### Export Options

- **ONNX Export:** Convert models to ONNX format for cross-platform deployment
  - Dynamo-based export for latest PyTorch features
  - TorchScript export for broader compatibility
  - Automatic model optimization and file size reporting

- **JIT Compilation:** PyTorch JIT tracing and scripting for optimized execution
  - Trace models for static computation graphs
  - Script models to preserve control flow
  - Automatic performance comparison with original models

- **TensorRT Integration:** GPU-accelerated inference with NVIDIA TensorRT
  - FP32, FP16, and INT8 precision modes
  - Automatic engine building and caching

### Performance Features

- **Automatic Benchmarking:** Built-in performance testing with statistical analysis
- **Memory Optimization:** Dynamic batch sizing and memory-efficient engine building
- **Cross-Platform:** Works on various GPU architectures and CUDA versions

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

## 💾 Dataset Caching

FDQ provides a powerful dataset caching system that can significantly speed up training by caching preprocessed data to disk and loading it into RAM for fast access during training.

### How It Works

The caching system operates in two stages:

1. **Deterministic Preprocessing & Caching:** Expensive, deterministic transformations (like resizing, normalization, data loading) are applied once and cached to HDF5 files on disk.

2. **On-the-fly Augmentation:** Fast, random augmentations (like flips, rotations) are applied dynamically during training for data variety.

### Configuration

Enable caching in your experiment configuration by adding a `caching` section to your data loader:

```json
"data": {
    "OXPET": {
        "class_name": "experiment_templates.segment_pets.oxpets_data.OxPetsData",
        "args": {
            "data_path": "/path/to/data",
            "batch_size": 8
        },
        "caching": {
            "cache_dir": "/path/to/cache",
            "shuffle_train": true,
            "shuffle_val": false,
            "shuffle_test": false
        }
    }
}
```

### Custom Augmentations

Create a custom augmentation script for on-the-fly transformations:

```python
# oxpets_augmentation.py
def augment(sample, transformers=None):
    """Apply custom augmentations to cached dataset samples.
    
    Args:
        sample (dict): Cached sample with keys like "image", "mask"
        transformers (dict): Dictionary of transformation functions
        
    Returns:
        dict: Augmented sample
    """
    # Apply synchronized random transformations
    sample["image"], sample["mask"] = transformers["random_vflip_sync"](
        sample["image"], sample["mask"]
    )
    return sample
```

Reference the augmentation script in your config:

```json
"data": {
    "OXPET": {
        "caching": {
            "augmentation_script": "experiment_templates.segment_pets.oxpets_augmentation"
        }
    }
}
```

### Key Features

- **Configuration Hash Validation:** Cache files are automatically invalidated when dataset configurations change
- **Faster Training:** Eliminate repeated preprocessing computations
- **Reduced I/O:** Minimize disk access during training
- **Reproducible Experiments:** Deterministic preprocessing with controllable randomness
- **Storage Efficiency:** HDF5 compression reduces disk space usage
- **Flexible Augmentation:** Combine cached preprocessing with dynamic augmentation


---

## 📦 Installing Additional Python Packages in your managed SLURM Environment

If your experiment requires extra Python packages, specify them in your config under `additional_pip_packages`. FDQ will install them automatically before running your experiment.

Example:

```json
"slurm_cluster": {
    "fdq_version": "0.0.64",
    "...": "...",
    "additional_pip_packages": [
        "monai==1.4.0",
        "prettytable"
    ]
}
```

---

## 🐛 Debugging

To debug an FDQ experiment, you'll need to install FDQ in development mode on your local or remote machine.

### Setup for Debugging

```bash
git clone https://github.com/mstadelmann/fonduecaquelon.git
cd fonduecaquelon
pip install -e .
```

### VS Code Debugging Configuration

1. Open your project in VS Code
2. Create or update your debugger configuration (`.vscode/launch.json`) to launch `run_experiment.py` with the corresponding parameters:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "FDQ Experiment Debug",
            "type": "debugpy",
            "request": "launch",
            "debugJustMyCode": false,
            "program": "${workspaceFolder}/src/fdq/run_experiment.py",
            "console": "integratedTerminal",
            "args": ["PATH_TO/experiment.json"],
            "cwd": "${workspaceFolder}"
        }
    ]
}
```

3. Debug/test your code.


---

## 📝 Tips

- **Config Inheritance:** Use the `parent` key in your config to inherit settings from another file, reducing duplication.
- **Multiple Models/Losses:** FDQ supports multiple models and losses per experiment — just add them to the config dictionaries.
- **Cluster Submission:** The `fdq_submit.py` utility handles SLURM job script generation and submission, including environment setup and result copying.
- **Model Export:** Use `-d` or `--dump` to interactively export and optimize trained models for deployment.

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