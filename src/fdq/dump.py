import os
import time
from typing import Any
import torch
from fdq.misc import iprint, wprint
from fdq.ui_functions import getIntInput, getYesNoInput


def select_experiment(experiment: Any) -> None:
    """Interactively select and load an experiment for model dumping."""
    sel_mode: int = getIntInput(
        "Select experiment for model dumping:\n"
        "  1) last exp best val model\n"
        "  2) last exp best train model\n"
        "  3) last exp last model\n"
        "  4) custom exp best val model\n"
        "  5) custom exp best train model\n"
        "  6) custom exp last model\n"
        "  7) define custom path\n",
        drange=[1, 7],
    )

    if sel_mode == 1:
        experiment.mode.best_val()
    elif sel_mode == 2:
        experiment.mode.best_train()
    elif sel_mode == 3:
        experiment.mode.last()
    elif sel_mode == 4:
        experiment.mode.custom_best_val()
    elif sel_mode == 5:
        experiment.mode.custom_best_train()
    elif sel_mode == 6:
        experiment.mode.custom_last()
    elif sel_mode == 7:
        experiment.mode.custom_path()

    experiment.load_trained_models()


def user_set_dtype(example: torch.Tensor) -> torch.Tensor:
    """Interactively set the dtype of the example tensor based on user input."""
    print("Example data type:", example.dtype)
    print("Example shape:", example.shape)
    sel_mode: int = getIntInput(
        "Define model tracing input dtype:\n  1) float32\n  2) float16\n  3) int8\n  4) float64\n",
        drange=[1, 4],
    )
    if sel_mode == 1:
        example = example.float()
    elif sel_mode == 2:
        example = example.half()
    elif sel_mode == 3:
        example = example.int()
    elif sel_mode == 4:
        example = example.double()
    return example


def get_example_tensor(experiment: Any) -> torch.Tensor:
    """Interactively select and return an example tensor from the experiment's data sources."""
    sources = list(experiment.data.keys())
    idx: int = (
        getIntInput(
            f"Select data source for tracing sample shape: {[f'{i + 1}) {src}' for i, src in enumerate(sources)]}",
            drange=[1, len(sources)],
        )
        - 1
    )

    sample = next(iter(experiment.data[sources[idx]].train_data_loader))
    if isinstance(sample, tuple):
        sample = sample[0]
    if isinstance(sample, list):
        sample = sample[0]
    if isinstance(sample, dict):
        sample = next(iter(sample.values()))

    print(f"Shape of sample tensor: {sample.shape}")

    if not getYesNoInput("Use tensor data from dataset (y) or random tensor (n)?"):
        sample = torch.rand_like(sample)

    return user_set_dtype(sample.to(experiment.device))


def select_model(experiment: Any) -> tuple[str, torch.nn.Module]:
    """Interactively select a model from the experiment and return its name and instance."""
    model_names = list(experiment.models.keys())
    idx: int = (
        getIntInput(
            f"Select model to dump: {[f'{i + 1}) {model}' for i, model in enumerate(model_names)]}",
            drange=[1, len(model_names)],
        )
        - 1
    )
    return (
        model_names[idx],
        experiment.models[model_names[idx]].to(experiment.device).eval(),
    )


def run_test(
    experiment: Any,
    example: torch.Tensor,
    model: torch.nn.Module,
    optimized_model: torch.nn.Module,
    config: dict[str, Any] = None,
) -> None:
    """Run a test comparing the original and optimized models, measuring speed and output difference."""
    iprint("\n-----------------------------------------------------------")
    iprint("Running test")
    iprint("-----------------------------------------------------------\n")

    model.to(experiment.device).eval()
    optimized_model.to(experiment.device).eval()

    # Warm-up
    for _ in range(3):
        _ = model(example)
        _ = optimized_model(example)

    # Measure time for original model
    times = []
    for _ in range(10):
        start = time.time()
        _ = model(example)
        if experiment.is_cuda:
            torch.cuda.synchronize()
        times.append(time.time() - start)
    avg_time_model = sum(times) / len(times)

    # Measure time for optimized model
    times_opt = []
    for _ in range(10):
        start = time.time()
        _ = optimized_model(example)
        if experiment.is_cuda:
            torch.cuda.synchronize()
        times_opt.append(time.time() - start)
    avg_time_optimized = sum(times_opt) / len(times_opt)

    # compute MAE between original and optimized model outputs
    example = example.to(experiment.device)

    out1 = model(example)
    out2 = optimized_model(example)
    if isinstance(out1, tuple):
        out1 = out1[0]
    if isinstance(out2, tuple):
        out2 = out2[0]
    loss = torch.nn.L1Loss()(out1, out2)

    iprint("\n-----------------------------------------------------------")
    print(f"Average time (original model): {avg_time_model:.6f} s")
    print(f"Average time (optimized model): {avg_time_optimized:.6f} s")
    print(f"MAE between outputs: {loss.item():.6f}")

    if config:
        print("\nConfiguration used for optimization:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    iprint("-----------------------------------------------------------\n")


def jit_trace_model(
    experiment: Any,
    config: dict[str, Any],
    model: torch.nn.Module,
    model_name: str,
    example: torch.Tensor,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Interactively JIT trace or script a model, optionally save and test the JIT model, and return the processed model and updated config."""
    if getYesNoInput("\n\nJIT Trace model? (y/n)\n"):
        # Tracing is following the execution of your module; it cannot pick up OPS like control flow.
        jit_model = torch.jit.trace(model, example, strict=False)
        config["jit_traced"] = True
        iprint("Model traced successfully!")

    elif getYesNoInput("JIT Script model? (y/n)\n"):
        # By working from the Python code, the compiler can include OPS like control flow.
        jit_model = torch.jit.script(model)
        config["jit_scripted"] = True
        iprint("Model scripted successfully!")
    else:
        jit_model = model

    if config["jit_traced"] or config["jit_scripted"]:
        if getYesNoInput("Save JIT model? (y/n)"):
            save_path = os.path.join(experiment.results_dir, f"{model_name}_jit.ts")
            torch.jit.save(jit_model, save_path)
            iprint(f"Traced model saved to {save_path}")

            if getYesNoInput("Run test on traced model? (y/n)"):
                traced_model = torch.jit.load(save_path)
                traced_model.eval()
                run_test(experiment, example, model, traced_model, config)

    return jit_model, config


def compile_model(
    config: dict[str, Any],
    experiment: Any,
    example: torch.Tensor,
    model: torch.nn.Module,
    model_name: str,
) -> None:
    """Compile, optionally JIT trace/script, and optimize a model using Torch-TensorRT, with interactive configuration and testing."""
    import torch_tensorrt
    from torch_tensorrt import Input
    try:
        jit_model, config = jit_trace_model(
            experiment, config, model, model_name, example
        )

    except (RuntimeError, TypeError, ValueError) as e:
        wprint("Failed to JIT Trace model!")
        print(e)

    inputs = [
        Input(
            example.shape,
            dtype=example.dtype,
            device={"device_type": "cuda" if experiment.is_cuda else "cpu"},
        )
    ]

    if getYesNoInput("Compile model? (y/n)\n"):
        if config["jit_traced"] or config["jit_scripted"]:
            inter_rep = "torchscript"
        else:
            inter_rep_choice = getIntInput(
                "Select intermediate representation:\n"
                "  1) default: Let Torch-TensorRT decide\n"
                "  2) ts: TorchScript\n",
                drange=[1, 2],
            )
            inter_rep = "default" if inter_rep_choice == 1 else "ts"

        truncate_double = getYesNoInput(
            "Truncate long and double? (y/n), default = y\n"
        )

        enabled_precisions = set()
        if getYesNoInput("Enable float32 precision? (y/n)"):
            enabled_precisions.add(torch.float32)
        if getYesNoInput("Enable float16 precision? (y/n)"):
            enabled_precisions.add(torch.float16)
        if getYesNoInput("Enable bfloat16 precision? (y/n)"):
            enabled_precisions.add(torch.bfloat16)
        if getYesNoInput("Enable float64 precision? (y/n)"):
            enabled_precisions.add(torch.float64)
        if getYesNoInput("Enable int8 precision? (y/n)"):
            enabled_precisions.add(torch.int8)
        if getYesNoInput("Enable quint8 precision? (y/n)"):
            enabled_precisions.add(torch.quint8)

        config.update(
            {
                "intermediate representation": inter_rep,
                "truncate double": truncate_double,
                "enabled precisions": enabled_precisions,
            }
        )

        optimized_model = torch_tensorrt.compile(
            jit_model,
            backend="torch_tensorrt",
            ir=inter_rep,
            inputs=inputs,
            enabled_precisions=enabled_precisions,
            debug=True,
            truncate_long_and_double=truncate_double,
        )
        iprint("Model compiled successfully!")

        if getYesNoInput("Run test on compiled model? (y/n)"):
            run_test(experiment, example, model, optimized_model, config)

        if getYesNoInput("Save optimized model? (y/n)"):
            save_path = os.path.join(
                experiment.results_dir, f"{model_name}_optimized.ts"
            )
            torch.save(optimized_model, save_path)
            iprint(f"Optimized model saved to {save_path}")


def dump_model(experiment: Any) -> None:
    """Interactively dumps, traces, scripts, compiles, tests, and saves a model from the given experiment."""
    iprint("\n-----------------------------------------------------------")
    iprint("Dump model")
    iprint("-----------------------------------------------------------\n")
    if experiment.is_distributed():
        raise ValueError(
            "ERROR: Cannot dump with world size > 1; please run in single process mode."
        )
    # dumping requires a sample input to trace the model
    # -> set exp t train mode so that train loader is created.
    experiment.mode.train()
    experiment.setupData()
    experiment.init_models(instantiate=False)

    select_experiment(experiment)
    model_name, model = select_model(experiment)
    iprint(f"Compiling {model_name}...")

    while True:
        example = get_example_tensor(experiment)

        config: dict[str, Any] = {
            "jit_traced": False,
            "jit_scripted": False,
            "input shape": example.shape,
            "input dtype": example.dtype,
        }

        try:
            compile_model(config, experiment, example, model, model_name)

        except (RuntimeError, TypeError, ValueError) as e:
            wprint("Failed to compile model!")
            print(e)
            if getYesNoInput("Try again? (y/n)"):
                continue
            break

        if not getYesNoInput("Dump another model? (y/n)"):
            break
