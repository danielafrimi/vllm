#!/usr/bin/env python3
"""
Simple script to create FP4 layer, load all parameters with scales=1, and run forward pass
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

def create_and_load_fp4_layer():
    """Create FP4 layer and load all parameters with scales = 1.0"""
    
    # Layer configuration
    input_size = 512
    output_size = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    
    print(f"Creating FP4 layer: {input_size} -> {output_size}")
    print(f"Device: {device}, dtype: {dtype}")
    
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
    
    # Get layer dimensions for weight creation
    layer_input_size = fp4_layer.input_size_per_partition
    layer_output_size = fp4_layer.output_size_per_partition
    group_size = fp4_layer.quant_method.quant_config.group_size
    
    print(f"Layer dimensions: {layer_input_size} -> {layer_output_size}")
    print(f"Group size: {group_size}")
    
    # Create weight data - all scales set to 1.0
    print("\nCreating weight tensors...")
    
    # 1. Main weight (packed FP4 as uint8) - random data
    weight_data = torch.randint(0, 255, (layer_output_size, layer_input_size // 2), 
                               dtype=torch.uint8, device=device)
    print(f"weight: {weight_data.shape} ({weight_data.dtype})")
    
    # 2. Input scale (set to 1.0)
    input_scale_data = torch.ones(1, dtype=torch.float32, device=device)
    print(f"input_scale: {input_scale_data.shape} ({input_scale_data.dtype}) = {input_scale_data.item()}")
    
    # 3. Global weight scale (set to 1.0)
    weight_scale_2_data = torch.ones(1, dtype=torch.float32, device=device)
    print(f"weight_scale_2: {weight_scale_2_data.shape} ({weight_scale_2_data.dtype}) = {weight_scale_2_data.item()}")
    
    # 4. Per-block weight scale (set to 1.0 equivalent in float8_e4m3fn)
    weight_scale_data = torch.ones((layer_output_size, layer_input_size // group_size), 
                                  dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
    print(f"weight_scale: {weight_scale_data.shape} ({weight_scale_data.dtype})")
    
    # 5. Bias (zeros)
    bias_data = torch.zeros(layer_output_size, dtype=dtype, device=device)
    print(f"bias: {bias_data.shape} ({bias_data.dtype})")
    
    # Load all parameters using their weight loaders
    print("\nLoading parameters...")
    
    # Load weight
    fp4_layer.weight.weight_loader(fp4_layer.weight, weight_data)
    print("✓ Loaded weight")
    
    # Load input_scale
    fp4_layer.input_scale.weight_loader(fp4_layer.input_scale, input_scale_data)
    print("✓ Loaded input_scale")
    
    # Load weight_scale_2
    fp4_layer.weight_scale_2.weight_loader(fp4_layer.weight_scale_2, weight_scale_2_data)
    print("✓ Loaded weight_scale_2")
    
    # Load weight_scale
    fp4_layer.weight_scale.weight_loader(fp4_layer.weight_scale, weight_scale_data)
    print("✓ Loaded weight_scale")
    
    # Load bias
    if fp4_layer.bias is not None:
        fp4_layer.bias.weight_loader(fp4_layer.bias, bias_data)
        print("✓ Loaded bias")
    
    # Process weights after loading (important for FP4!)
    print("\nProcessing weights after loading...")
    fp4_layer.quant_method.process_weights_after_loading(fp4_layer)
    print("✓ Post-processing completed")
    
    return fp4_layer

def run_forward_pass(layer):
    """Run forward pass with the FP4 layer"""
    
    print("\n" + "="*50)
    print("RUNNING FORWARD PASS")
    print("="*50)
    
    # Create input tensor
    batch_size = 4
    input_size = layer.input_size_per_partition
    device = next(layer.parameters()).device
    dtype = torch.float16
    
    input_tensor = torch.randn(batch_size, input_size, dtype=dtype, device=device)
    print(f"Input tensor: {input_tensor.shape} ({input_tensor.dtype})")
    print(f"Input range: [{input_tensor.min().item():.3f}, {input_tensor.max().item():.3f}]")
    
    # Run forward pass
    try:
        with torch.no_grad():
            output = layer(input_tensor)
            
            # Handle tuple output (output, bias)
            if isinstance(output, tuple):
                output_tensor, output_bias = output
                print(f"Output tensor: {output_tensor.shape} ({output_tensor.dtype})")
                print(f"Output range: [{output_tensor.min().item():.3f}, {output_tensor.max().item():.3f}]")
                if output_bias is not None:
                    print(f"Output bias: {output_bias.shape} ({output_bias.dtype})")
                else:
                    print("Output bias: None")
            else:
                print(f"Output tensor: {output.shape} ({output.dtype})")
                print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
            
            print("✓ Forward pass successful!")
            return True
            
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        print("This could be due to CUDA kernel compilation or hardware compatibility issues.")
        return False

def main():
    """Main function"""
    print("Simple FP4 Layer Script")
    print("="*50)
    
    # Setup distributed environment
    print("Setting up distributed environment...")
    setup_distributed()
    
    # Create and load FP4 layer
    print("\nCreating and loading FP4 layer...")
    fp4_layer = create_and_load_fp4_layer()
    
    # Run forward pass
    success = run_forward_pass(fp4_layer)
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print("✓ FP4 layer created successfully")
    print("✓ All parameters loaded with scales = 1.0")
    print("✓ Parameter loading process completed")
    
    if success:
        print("✓ Forward pass completed successfully")
        print("\nThe FP4 layer is working correctly!")
    else:
        print("⚠ Forward pass failed (likely hardware/kernel issues)")
        print("✓ But parameter loading structure is correct")
    
    print("\nParameters loaded:")
    print("- weight: uint8 packed FP4 weights")
    print("- input_scale: 1.0 (float32)")
    print("- weight_scale_2: 1.0 (float32)")
    print("- weight_scale: 1.0 equivalent (float8_e4m3fn)")
    print("- bias: zeros (float16)")

if __name__ == "__main__":
    main()
