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
To run an experiment with FDQ, you need to define your [experiment loop](experiment_templates/segment_pets/train_oxpets.py), a [data-loader](experiment_templates/segment_pets/oxfordpet_preparator.py) and, optionally, a [test loop](experiment_templates/segment_pets/oxpets_test.py). The model can either be a pre-installed one—such as [Chuchichaestli](https://github.com/CAIIVS/chuchichaestli) — or a custom model that you define and import yourself. Models, losses, and data loaders are always defined as dictionaries. For example, the following configuration:

```json
"models": {
    "ccUNET": {
        "class_name": "chuchichaestli.models.unet.unet.UNet"
    }
}
```

allows you to access the model in your training loop via `experiment.models["ccUNET"]`. The same dictionary-based structure applies to losses and data loaders as well. This setup enables you to define and manage as many models, losses, and data loaders as needed for your experiment.
