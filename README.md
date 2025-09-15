# FDQ | Fonduecaquelon

A *fonduecaquelon* is the heavy pot that keeps cheeses (e.g. 50% Gruyère and 50% Vacherin) melting smoothly into a perfectly blended whole — and FDQ does the same for deep learning. It keeps models, data loaders, training loops, and tools at a steady “temperature” so everything works seamlessly together, streamlining PyTorch workflows by automating repetitive tasks and providing a flexible, extensible framework for experiment management. Built for ML engineers who want to focus on experiments rather than boilerplate, FDQ lets you spend more time innovating and less time setting up.

* [GitHub Repository](https://github.com/mstadelmann/fonduecaquelon)
* [PyPI Package](https://pypi.org/project/fdq/)

## 🚀 Features

* **Minimal Boilerplate:** Define only what matters — FDQ handles the rest.
* **Flexible Experiment Configuration:** Use JSON config files with inheritance support for easy experiment management.
* **Multi-Model Support:** Seamlessly manage multiple models, losses, and data loaders.
* **Cluster Ready:** Submit jobs to SLURM clusters with ease using built-in utilities such as automatic job resubmission.
* **Extensible:** Easily integrate custom models, data loaders, and training/testing loops.
* **Automatic Dependency Management:** Install additional pip packages per experiment.
* **Distributed Training:** Out-of-the-box support for PyTorch DDP.
* **Model Export & Optimization:** Export trained models to ONNX with optimization options.
* **High-Performance Inference:** TensorRT integration for GPU-accelerated inference with up to 10x speedup.
* **Model Compilation:** JIT tracing/scripting and `torch.compile` support for optimized execution.
* **Interactive Model Dumping:** Intuitive interface for exporting and optimizing trained models.
* **Monitoring Tools:** Built-in support for [Weights & Biases](https://wandb.a) and [TensorBoard](https://www.tensorflow.org/tensorboard).

## 🛠️ Installation

Install the latest release from PyPI:

```bash
pip install fdq
```

If you have an NVIDIA GPU and want to run inference, install GPU dependencies:

```bash
pip install fdq[gpu]
```

For development and the latest features, clone the repository:

```bash
git clone https://github.com/mstadelmann/fonduecaquelon.git
cd fonduecaquelon
pip install -e .[dev,gpu]
```

## 📖 Usage

### Local Experiments

All experiment parameters are defined in a [config file](experiment_templates/mnist/mnist_class_dense.json). Config files can inherit from a [parent file](experiment_templates/mnist/mnist_parent.json) for easy reuse and organization.

Run an experiment locally:

```bash
fdq <path_to_config_file.json>
```

### SLURM Cluster Execution

To run experiments on a SLURM cluster, add a `slurm_cluster` section to your config. See [this example](experiment_templates/segment_pets/segment_pets_01.json).

Submit your experiment:

```bash
python <path_to>/fdq_submit.py <path_to_config_file.json>
```

### Model Export and Optimization

After training, export and optimize models for deployment:

```bash
# Interactive model dumping with export options
fdq <path_to_config_file.json> -nt -d
```

This launches an interactive interface where you can:

* **Export to ONNX:** Convert PyTorch models to ONNX format using Dynamo or TorchScript
* **JIT Compilation:** Trace or script models with PyTorch JIT
* **TensorRT Optimization:** Compile models for GPU inference with FP32, FP16, or INT8 precision
* **Performance Benchmarking:** Compare optimized vs. original model performance

### Additional CLI Options

FDQ provides multiple command-line options:

```bash
# Run training (default)
fdq <config_file.json>

# Skip training
fdq <config_file.json> -nt

# Train and test automatically
fdq <config_file.json> -ta

# Interactive testing
fdq <config_file.json> -nt -ti

# Export and optimize models
fdq <config_file.json> -nt -d

# Run inference tests
fdq <config_file.json> -nt -i

# Print model architecture before training
fdq <config_file.json> -p

# Resume from checkpoint
fdq <config_file.json> -rp /path/to/checkpoint
```

## 🚄 Model Export & Deployment

FDQ offers full model export and optimization support for deployment:

### Export Options

* **ONNX Export:** Convert models to ONNX for cross-platform use

  * Dynamo-based export for the latest PyTorch features
  * TorchScript export for broad compatibility
  * Automatic optimization and file size reporting

* **JIT Compilation:** PyTorch JIT tracing and scripting

  * Trace models for static graphs
  * Script models to preserve control flow
  * Automatic performance comparison with original models

* **TensorRT Integration:** GPU-accelerated inference with NVIDIA TensorRT

  * FP32, FP16, and INT8 precision
  * Automatic engine building and caching

### Performance Features

* **Automatic Benchmarking:** Built-in performance testing with statistics
* **Memory Optimization:** Dynamic batch sizing and memory-efficient engines
* **Cross-Platform:** Compatible with various GPU architectures and CUDA versions

## ⚙️ Configuration Overview

FDQ uses JSON config files to define experiments. These specify models, data loaders, training/testing scripts, and cluster settings.

### Models

Models are defined as dictionaries. You can use pre-installed ones (e.g. [Chuchichaestli](https://github.com/CAIIVS/chuchichaestli)) or your own. Example:

```json
"models": {
    "ccUNET": {
        "class_name": "chuchichaestli.models.unet.unet.UNet"
    }
}
```

Access models in training via `experiment.models["ccUNET"]`. The same structure applies to losses and data loaders.

### Data Loaders

Your data loader class must implement `create_datasets(experiment, args)`, returning:

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

These values are available as `experiment.data["<name>"].<key>`.

### Training Loop

Define a function in your training script:

```python
def fdq_train(experiment: fdqExperiment):
```

Within it, you can access components:

```python
nb_epochs = experiment.exp_def.train.args.epochs
data_loader = experiment.data["OXPET"].train_data_loader
model = experiment.models["ccUNET"]
```

See [train\_oxpets.py](experiment_templates/segment_pets/train_oxpets.py) for an example.

At the beginning of each epoch call `experiment.on_epoch_start()` and at the end call `experiment.on_epoch_end(...)`. These hooks reset per‑epoch timers/counters, aggregate metrics, and perform logging (TensorBoard / Weights & Biases) and any scheduling/checkpoint logic tied to epoch boundaries.

Minimal pattern:

```python
def fdq_train(experiment: fdqExperiment):
    nb_epochs = experiment.exp_def.train.args.epochs
    train_loader = experiment.data["OXPET"].train_data_loader

    for epoch in range(nb_epochs):
        experiment.on_epoch_start()

        running_loss = 0.0
        for batch in train_loader:
            # forward / loss / backward / optimizer step ...
            pass

        # Example scalar logging
        scalars = {"train/loss": running_loss / max(1, len(train_loader))}
        experiment.on_epoch_end(log_scalars=scalars)
```

See the full implementation in [train_oxpets.py](experiment_templates/segment_pets/train_oxpets.py) for a richer example (images, text, or additional metrics).

### Testing Loop

Testing is similar. Define:

```python
def fdq_test(experiment: fdqExperiment):
```

See [oxpets\_test.py](experiment_templates/segment_pets/oxpets_test.py) for reference.

## 💾 Dataset Caching

FDQ includes a dataset caching system to speed up training by caching preprocessed data to disk and loading it into RAM. See [segment\_pets\_05\_cached.json](experiment_templates/segment_pets/segment_pets_05_cached.json) for an example.

### How It Works

1. **Deterministic Preprocessing & Caching:** Expensive transformations (resizing, normalization, data loading) are applied once and cached as HDF5 files.
2. **On-the-fly Augmentation:** Fast, random augmentations (e.g. flips, rotations) are applied during training.

### Configuration

Enable caching in your config:

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

Define augmentations:

```python
# oxpets_augmentation.py
def augment(sample, transformers=None):
    """Apply custom augmentations to cached dataset samples."""
    sample["image"], sample["mask"] = transformers["random_vflip_sync"](
        sample["image"], sample["mask"]
    )
    return sample
```

Reference in your config:

```json
"data": {
    "OXPET": {
        "caching": {
            "augmentation_script": "experiment_templates.segment_pets.oxpets_augmentation"
        }
    }
}
```

## 🧮 Mixed precision

Leveraging torch.amp for mixed precision training can dramatically accelerate your training workflow. For a practical implementation, see [this](experiment_templates/segment_pets/train_oxpets.py) example.

Observed speedup on H200sxm GPUs:

| Experiment                                                                               | Time per epoch \[s] |
| ---------------------------------------------------------------------------------------- | ------------------- |
| [segment pets with AMP](experiment_templates/segment_pets/segment_pets_01.json)          | 100                 |
| [segment pets without AMP](experiment_templates/segment_pets/segment_pets_01_noAMP.json) | 170                 |

## 🖧 Distributed Training

To run with [PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html), add:

```json
"slurm_cluster": {
    "world_size": 2,
    "cpus_per_task": 16,
    "gres": "gpu:h200sxm:2",
}
```

See [segment\_pets\_03\_distributed\_w2.json](experiment_templates/segment_pets/segment_pets_03_distributed_w2.json).

Use the same number of GPUs as your world size. DDP requires more CPU cores and memory, since multiple data loaders run in parallel. It’s most beneficial for large models, as overhead is significant.

Observed speedup on H200sxm GPUs:

| Experiment                                                                               | Time per ep. w/o AMP \[s] | with AMP \[s] | 
| ---------------------------------------------------------------------------------------- | ------------------------- | ------------- |
| [segment pets default](experiment_templates/segment_pets/segment_pets_01.json)           | 170                       | 100           |
| [DDP with 2 GPUs](experiment_templates/segment_pets/segment_pets_03_distributed_w2.json) | 100                       | 65           |
| [DDP with 4 GPUs](experiment_templates/segment_pets/segment_pets_04_distributed_w4.json) | 60                        | 45            |

By toggling mixed precision, you can directly observe how more intensive workloads see greater speedups when using DDP.

## 📦 Installing Additional Python Packages in SLURM

If your experiment requires extra packages, specify them in `additional_pip_packages`. FDQ installs them before execution.

Example:

```json
"slurm_cluster": {
    "fdq_version": "0.0.73",
    "...": "...",
    "additional_pip_packages": [
        "monai==1.4.0",
        "prettytable"
    ]
}
```

## 🐛 Debugging

For debugging, install FDQ in development mode:

```bash
git clone https://github.com/mstadelmann/fonduecaquelon.git
cd fonduecaquelon
pip install -e .
```

### VS Code Setup

1. Open your project in VS Code.
2. Add or update `.vscode/launch.json` to run `run_experiment.py`:

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

## 📝 Tips

* **Config Inheritance:** Use the `parent` key to inherit from another config and reduce duplication.
* **Multiple Models/Losses:** Add multiple models and losses to config dictionaries as needed.
* **Cluster Submission:** `fdq_submit.py` handles SLURM job script generation, submission, environment setup, and result copying.
* **Model Export:** Use `-d` or `--dump` for interactive model export and optimization.

## 📚 Resources

* [Experiment Templates](experiment_templates/)
* [Chuchichaestli Models](https://github.com/CAIIVS/chuchichaestli)

## 🤝 Contributing

Contributions are welcome! Please open issues or pull requests on [GitHub](https://github.com/mstadelmann/fonduecaquelon).

## 🧀 Enjoy your Fondue!

<p align="center">
  <img src="assets/fdq_logo.jpg" alt="FDQ Logo" width="300"/>
</p>
