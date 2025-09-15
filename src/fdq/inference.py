import os
import traceback
import onnxruntime as ort
import numpy as np
from fdq.ui_functions import getIntInput, save_images
from typing import Any
from fdq.misc import iprint
from fdq.testing import find_model_path
from fdq.dump import get_example_tensor


def find_onnx_models(experiment: Any) -> str:
    """Find all ONNX model files in the given directory and let user select one."""
    # set mode to custom to trigger experiment selection. (the "last" part is irrelevant)
    experiment.mode.custom_last()
    path, _ = find_model_path(experiment)

    onnx_files = []
    if os.path.exists(path):
        for file in os.listdir(path):
            if file.endswith(".onnx"):
                onnx_files.append(file)

    if not onnx_files:
        raise FileNotFoundError(f"No ONNX files found in directory: {path}. Dump model first!")

    onnx_files.sort()
    iprint("\nAvailable ONNX models:")
    for i, model_file in enumerate(onnx_files):
        iprint(f"{i + 1}) {model_file}")

    idx = getIntInput(f"Select ONNX model (1-{len(onnx_files)}): ", drange=[1, len(onnx_files)]) - 1

    selected_model = onnx_files[idx]
    selected_path = os.path.join(path, selected_model)

    iprint(f"Selected model: {selected_model} \npath: {selected_path}")

    return selected_path


def get_precision_choice() -> str:
    """Prompt user for precision choice."""
    choices = ["fp32", "fp16", "int8"]
    iprint("\nSelect TensorRT precision mode:")
    for i, choice in enumerate(choices):
        iprint(f"{i + 1}) {choice}")

    idx = getIntInput(f"Choose precision (1-{len(choices)}): ", drange=[1, len(choices)]) - 1
    return choices[idx]


def compare_with_pytorch(onnx_path: str, sample_input: np.ndarray, trt_output: np.ndarray):
    """Compare TensorRT trt_result with PyTorch ONNX runtime (if available)."""
    try:
        iprint("\n-----------------------------------------------------------")
        iprint("COMPARING WITH PyTorch ONNX RUNTIME")
        iprint("-----------------------------------------------------------\n")

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1

        ort_session = ort.InferenceSession(onnx_path, sess_options)

        input_name = ort_session.get_inputs()[0].name

        # OnnxRunTime inference
        ort_inputs = {input_name: sample_input}
        ort_outputs = ort_session.run(None, ort_inputs)
        ort_output = ort_outputs[0]

        # Compare outputs
        if isinstance(trt_output, np.ndarray) and isinstance(ort_output, np.ndarray):
            mae = np.mean(np.abs(trt_output - ort_output))
            mse = np.mean((trt_output - ort_output) ** 2)
            max_diff = np.max(np.abs(trt_output - ort_output))

            print("Output comparison (TensorRT vs ONNX Runtime):")
            print(f"  Mean Absolute Error: {mae:.6f}")
            print(f"  Mean Squared Error:  {mse:.6f}")
            print(f"  Max Absolute Diff:   {max_diff:.6f}")

    except ImportError:
        print("ONNX Runtime not available for comparison")
    except Exception as e:
        print(f"Comparison failed: {e}")


def run_tensorrt_inference(onnx_model_path: str, precision: str = "fp32", experiment: Any = None) -> None:
    """Run TensorRT inference on an ONNX model.

    Args:
        onnx_model_path: Path to the ONNX model file
        precision: Precision mode - "fp32", "fp16", or "int8"
        experiment: Experiment object providing context and data for inference
    """
    print(f"Loading ONNX model: {onnx_model_path}")
    print(f"Using precision: {precision}")

    try:
        # Create TensorRT inference object
        from fdq.tensorrt_engine import TensorRTInference

        trt_inference = TensorRTInference(onnx_model_path, precision=precision)

        # Print model information
        info = trt_inference.get_model_info()
        iprint("\n-----------------------------------------------------------")
        iprint("MODEL INFORMATION")
        iprint("-----------------------------------------------------------\n")
        for key, value in info.items():
            print(f"{key:20s}: {value}")

        sample_input = get_example_tensor(experiment).detach().cpu().numpy()
        print(f"\nSample input shape: {sample_input.shape}")
        print(f"Sample input dtype: {sample_input.dtype}")

        iprint("\n-----------------------------------------------------------")
        iprint("RUNNING SINGLE TRT INFERENCE")
        iprint("-----------------------------------------------------------\n")
        trt_result = trt_inference.infer(sample_input)

        save_images(
            images=[sample_input, trt_result],
            save_path=experiment.get_next_export_fn(),
            figsize=(12, 5),
            titles=["Input", "TensorRT Output"],
        )

        print(f"Output shape: {trt_result.shape}")
        print(f"Output dtype: {trt_result.dtype}")
        print(f"Output min/max: {trt_result.min():.6f} / {trt_result.max():.6f}")

        iprint("\n-----------------------------------------------------------")
        iprint("BENCHMARKING TRT PERFORMANCE")
        iprint("-----------------------------------------------------------\n")
        results = trt_inference.benchmark(sample_input, num_runs=100, warmup_runs=10)

        print("\nPerformance Summary:")
        print(f"  Average Latency: {results['avg_latency_ms']:.2f} ms")
        print(f"  Throughput:      {results['throughput_fps']:.2f} FPS")
        print(f"  Min Latency:     {results['min_latency_ms']:.2f} ms")
        print(f"  Max Latency:     {results['max_latency_ms']:.2f} ms")
        print(f"  Std Deviation:   {results['std_latency_ms']:.2f} ms")

        try:
            compare_with_pytorch(onnx_model_path, sample_input, trt_result)
        except Exception as e:
            print(f"\nPyTorch comparison failed: {e}")

        iprint("\n-----------------------------------------------------------")
        iprint("INFERENCE COMPLETE")
        iprint("-----------------------------------------------------------\n")

    except Exception as e:
        print(f"Error during TensorRT inference: {e}")
        traceback.print_exc()


def inference_model(experiment: Any) -> None:
    """Run model inference using ONNX and TensorRT optimization.

    Args:
        experiment: The experiment object containing model and configuration.

    Raises:
        ValueError: If distributed training is enabled (not supported for inference).
    """
    iprint("\n-----------------------------------------------------------")
    iprint("Run model inference with ONNX - TensorRT")
    iprint("-----------------------------------------------------------\n")

    if experiment.is_distributed():
        raise ValueError("ERROR: Cannot run inference with world size > 1; please run in single process mode!")
    experiment.setupData()

    onnx_path = find_onnx_models(experiment)
    precision = get_precision_choice()

    run_tensorrt_inference(onnx_path, precision, experiment=experiment)
