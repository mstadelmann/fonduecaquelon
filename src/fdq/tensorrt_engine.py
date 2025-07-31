import os
import numpy as np
import time
import tensorrt as trt
from typing import Any
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401


class TensorRTInference:
    """TensorRT inference class for ONNX models."""

    def __init__(self, onnx_path: str, engine_path: str = None, precision: str = "fp32"):
        """Initialize TensorRT inference.

        Args:
            onnx_path: Path to the ONNX model
            engine_path: Path to save/load TensorRT engine (optional)
            precision: Precision mode - "fp32", "fp16", or "int8"
        """
        self.onnx_path = onnx_path
        self.engine_path = engine_path or onnx_path.replace(".onnx", f"_{precision}.trt")
        self.precision = precision

        # TensorRT objects
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = None
        self.engine = None
        self.context = None

        # Memory management
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        # Model info
        self.input_shapes = {}
        self.output_shapes = {}

        self._build_engine()
        self._allocate_buffers()

    def _build_engine(self):
        """Build TensorRT engine from ONNX model."""
        if os.path.exists(self.engine_path):
            print(f"Loading existing TensorRT engine from {self.engine_path}")
            self._load_engine()
        else:
            print(f"Building TensorRT engine from {self.onnx_path}")
            self._build_engine_from_onnx()
            self._save_engine()

    def _build_engine_from_onnx(self):
        """Build TensorRT engine from ONNX model."""
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)

        # Parse ONNX model
        print(f"Parsing ONNX model: {self.onnx_path}")
        with open(self.onnx_path, "rb") as model:
            if not parser.parse(model.read()):
                print("ONNX Parser Errors:")
                for error in range(parser.num_errors):
                    print(f"  Error {error}: {parser.get_error(error)}")
                raise RuntimeError("Failed to parse ONNX model")

        print(f"Successfully parsed ONNX model with {network.num_layers} layers")

        # Configure builder
        config = builder.create_builder_config()

        # Set precision with validation
        if self.precision == "fp16":
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("Using FP16 precision")
            else:
                print("Warning: FP16 not supported on this platform, falling back to FP32")
                self.precision = "fp32"
        elif self.precision == "int8":
            if builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                print("Using INT8 precision")
            else:
                print("Warning: INT8 not supported on this platform, falling back to FP32")
                self.precision = "fp32"
        else:
            print("Using FP32 precision")

        # Set memory pool with error checking
        try:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
            print("Set workspace memory limit to 1GB")
        except Exception as e:
            print(f"Warning: Could not set memory pool limit: {e}")

        # Print network info before building
        print(f"Network has {network.num_inputs} inputs and {network.num_outputs} outputs")
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            print(f"  Input {i}: {input_tensor.name}, shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")

        # Build engine with detailed logging
        print("Building TensorRT engine... This may take a while.")
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            print("Engine building failed. Possible reasons:")
            print("1. GPU compute capability not supported")
            print("2. Insufficient GPU memory")
            print("3. Unsupported ONNX operations")
            print("4. Model too complex for current TensorRT version")
            raise RuntimeError("Failed to build TensorRT engine")

        print("Engine built successfully, deserializing...")
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(serialized_engine)

        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        print("TensorRT engine ready for inference")

    def _save_engine(self):
        """Save TensorRT engine to file."""
        if self.engine:
            with open(self.engine_path, "wb") as f:
                f.write(self.engine.serialize())
            print(f"TensorRT engine saved to {self.engine_path}")

    def _load_engine(self):
        """Load TensorRT engine from file."""
        with open(self.engine_path, "rb") as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()

    def _allocate_buffers(self):
        """Allocate GPU and CPU buffers for inference."""
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            tensor_dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))

            # Calculate size
            size = trt.volume(tensor_shape)

            # Allocate CPU and GPU memory
            host_mem = cuda.pagelocked_empty(size, tensor_dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Store tensor info
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append(
                    {
                        "name": tensor_name,
                        "host": host_mem,
                        "device": device_mem,
                        "shape": tensor_shape,
                    }
                )
                self.input_shapes[tensor_name] = tensor_shape
            else:
                self.outputs.append(
                    {
                        "name": tensor_name,
                        "host": host_mem,
                        "device": device_mem,
                        "shape": tensor_shape,
                    }
                )
                self.output_shapes[tensor_name] = tensor_shape

            self.bindings.append(int(device_mem))

    def preprocess_input(self, data: np.ndarray) -> np.ndarray:
        """Preprocess input data. Override this method for custom preprocessing.

        Args:
            data: Input data as numpy array

        Returns:
            Preprocessed data
        """
        # Default: just ensure correct dtype and shape
        if len(self.inputs) == 1:
            expected_shape = self.inputs[0]["shape"]
            if data.shape != expected_shape:
                print(f"Warning: Input shape {data.shape} doesn't match expected {expected_shape}")

        return data.astype(np.float32)

    def postprocess_output(self, outputs: list[np.ndarray]) -> Any:
        """Postprocess output data. Override this method for custom postprocessing.

        Args:
            outputs: List of output arrays

        Returns:
            Postprocessed outputs
        """
        # Default: return raw outputs
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def infer(self, input_data: np.ndarray) -> Any:
        """Run inference on input data.

        Args:
            input_data: Input data as numpy array

        Returns:
            Inference results
        """
        # Preprocess input
        input_data = self.preprocess_input(input_data)

        # Copy input data to GPU
        for i, inp in enumerate(self.inputs):
            np.copyto(inp["host"], input_data.ravel())
            cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)

        # Set tensor addresses
        for i, inp in enumerate(self.inputs):
            self.context.set_tensor_address(inp["name"], inp["device"])
        for i, out in enumerate(self.outputs):
            self.context.set_tensor_address(out["name"], out["device"])

        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy outputs from GPU
        outputs = []
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
            self.stream.synchronize()
            output = out["host"].copy()
            output = output.reshape(out["shape"])
            outputs.append(output)

        return self.postprocess_output(outputs)

    def benchmark(self, input_data: np.ndarray, num_runs: int = 100, warmup_runs: int = 10) -> dict[str, float]:
        """Benchmark inference performance.

        Args:
            input_data: Sample input data
            num_runs: Number of inference runs for timing
            warmup_runs: Number of warmup runs

        Returns:
            Performance metrics
        """
        print(f"Benchmarking with {warmup_runs} warmup runs and {num_runs} timing runs...")

        # Warmup
        for _ in range(warmup_runs):
            self.infer(input_data)

        # Timing runs
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            self.infer(input_data)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms

        # Calculate statistics
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)

        results = {
            "avg_latency_ms": avg_time,
            "min_latency_ms": min_time,
            "max_latency_ms": max_time,
            "std_latency_ms": std_time,
            "throughput_fps": 1000.0 / avg_time,
        }

        print(f"Average latency: {avg_time:.2f} ms")
        print(f"Min latency: {min_time:.2f} ms")
        print(f"Max latency: {max_time:.2f} ms")
        print(f"Std latency: {std_time:.2f} ms")
        print(f"Throughput: {results['throughput_fps']:.2f} FPS")

        return results

    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        info = {
            "engine_path": self.engine_path,
            "precision": self.precision,
            "input_shapes": self.input_shapes,
            "output_shapes": self.output_shapes,
            "num_inputs": len(self.inputs),
            "num_outputs": len(self.outputs),
        }
        return info

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, "stream"):
            self.stream.synchronize()
