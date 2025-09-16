#!/usr/bin/env python3
"""
Example showing how FP4 weight loading works in vLLM
"""

import torch
import tempfile
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.platforms import current_platform

def setup_distributed():
    """Setup distributed environment for vLLM"""
    temp_file = tempfile.mkstemp()[1]
    backend = "nccl"
    if current_platform.is_cpu() or current_platform.is_tpu():
        backend = "gloo"

    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{temp_file}",
        local_rank=0,
        backend=backend
    )
    initialize_model_parallel(1, 1)

def create_fp4_layer():
    """Create an FP4 quantized layer"""
    input_size = 512
    output_size = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    
    # Create FP4 config
    fp4_config = ModelOptNvFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        exclude_modules=[],
        kv_cache_quant_algo=None,
    )
    
    # Create the layer
    fp4_layer = ReplicatedLinear(
        input_size=input_size,
        output_size=output_size,
        bias=True,
        params_dtype=dtype,
        quant_config=fp4_config,
    ).to(device)
    
    return fp4_layer

def inspect_fp4_parameters(layer):
    """Inspect the parameters created by FP4 quantization"""
    print("=== FP4 Layer Parameters ===")
    for name, param in layer.named_parameters():
        print(f"\n{name}:")
        print(f"  Shape: {param.shape}")
        print(f"  Dtype: {param.dtype}")
        print(f"  Device: {param.device}")
        print(f"  Parameter type: {type(param).__name__}")
        
        # Check if it has a custom weight_loader
        if hasattr(param, 'weight_loader'):
            print(f"  Has custom weight_loader: Yes")
            print(f"  Weight loader: {param.weight_loader}")
        else:
            print(f"  Has custom weight_loader: No")

def simulate_weight_loading(layer):
    """Simulate loading weights into the FP4 layer"""
    print("\n=== Simulating Weight Loading ===")
    
    # Get layer dimensions
    input_size = layer.input_size_per_partition
    output_size = layer.output_size_per_partition
    group_size = layer.quant_method.quant_config.group_size
    
    print(f"Input size: {input_size}")
    print(f"Output size: {output_size}")
    print(f"Group size: {group_size}")
    
    # Create mock weight data that would come from a checkpoint
    device = next(layer.parameters()).device
    mock_weights = {
        'weight': torch.randint(0, 255, (output_size, input_size // 2), dtype=torch.uint8, device=device),
        'input_scale': torch.abs(torch.randn(1, dtype=torch.float32, device=device)) + 0.1,  # Positive scale
        'weight_scale_2': torch.abs(torch.randn(1, dtype=torch.float32, device=device)) + 0.1,  # Positive scale
        'weight_scale': torch.randint(0, 255, (output_size, input_size // group_size), dtype=torch.uint8, device=device).to(torch.float8_e4m3fn),
        'bias': torch.randn(output_size, dtype=torch.float16, device=device)
    }
    
    print(f"\nMock weight shapes:")
    for name, weight in mock_weights.items():
        print(f"  {name}: {weight.shape} ({weight.dtype})")
    
    # Load the weights using each parameter's weight_loader
    print(f"\nLoading weights...")
    for name, mock_weight in mock_weights.items():
        if hasattr(layer, name):
            param = getattr(layer, name)
            if param is not None:
                try:
                    # Use the parameter's weight_loader if available
                    if hasattr(param, 'weight_loader'):
                        param.weight_loader(param, mock_weight)
                    else:
                        # Fallback to direct copy
                        param.data.copy_(mock_weight)
                    print(f"  ✓ Loaded {name}")
                except Exception as e:
                    print(f"  ✗ Failed to load {name}: {e}")
            else:
                print(f"  - Parameter {name} is None")
        else:
            print(f"  - Parameter {name} not found")

def demonstrate_forward_pass(layer):
    """Demonstrate a forward pass through the FP4 layer"""
    print(f"\n=== Forward Pass ===")
    
    batch_size = 4
    input_size = layer.input_size_per_partition
    device = next(layer.parameters()).device
    dtype = torch.float16
    
    # Create input tensor
    input_tensor = torch.randn(batch_size, input_size, dtype=dtype, device=device)
    print(f"Input shape: {input_tensor.shape}")
    
    try:
        with torch.no_grad():
            output = layer(input_tensor)
            if isinstance(output, tuple):
                output = output[0]
            print(f"Output shape: {output.shape}")
            print(f"Output dtype: {output.dtype}")
            print("✓ Forward pass successful")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")

def main():
    """Main demonstration"""
    print("FP4 Weight Loading Demonstration")
    print("=" * 50)
    
    # Setup
    setup_distributed()
    
    # Create layer
    print("\n1. Creating FP4 layer...")
    fp4_layer = create_fp4_layer()
    
    # Inspect parameters
    print("\n2. Inspecting parameters...")
    inspect_fp4_parameters(fp4_layer)
    
    # Simulate weight loading
    print("\n3. Simulating weight loading...")
    simulate_weight_loading(fp4_layer)
    
    # Demonstrate forward pass
    print("\n4. Testing forward pass...")
    demonstrate_forward_pass(fp4_layer)
    
    print(f"\n=== Summary ===")
    print("The FP4 layer creates 4 main parameters:")
    print("1. weight (uint8, packed FP4 data)")
    print("2. input_scale (float32, per-tensor input scaling)")
    print("3. weight_scale_2 (float32, global weight scaling)")
    print("4. weight_scale (float8_e4m3fn, per-block weight scaling)")
    print("5. bias (optional, same dtype as params_dtype)")
    print("\nEach parameter has a custom weight_loader that handles")
    print("tensor parallelism and proper data loading from checkpoints.")

if __name__ == "__main__":
    main()
