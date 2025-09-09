#!/usr/bin/env python3
"""
Script to run VLM evaluation using vLLM + VLMEvalKit
"""

import argparse
import subprocess
import sys
import time
import os
import requests
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from vlmeval_vllm_config import register_vllm_models, list_available_models


def check_vllm_server(api_base="http://localhost:8000"):
    """Check if vLLM server is running"""
    try:
        response = requests.get(f"{api_base}/v1/models", timeout=5)
        return response.status_code == 200
    except:
        return False


def start_vllm_server(model_path, port=8000, **kwargs):
    """Start vLLM server"""
    print(f"🚀 Starting vLLM server with model: {model_path}")
    
    cmd = [
        "vllm", "serve", model_path,
        "--port", str(port),
        "--trust-remote-code"
    ]
    
    # Add additional arguments
    if kwargs.get('dtype'):
        cmd.extend(["--dtype", kwargs['dtype']])
    if kwargs.get('tensor_parallel_size'):
        cmd.extend(["--tensor-parallel-size", str(kwargs['tensor_parallel_size'])])
    if kwargs.get('max_model_len'):
        cmd.extend(["--max-model-len", str(kwargs['max_model_len'])])
    if kwargs.get('gpu_memory_utilization'):
        cmd.extend(["--gpu-memory-utilization", str(kwargs['gpu_memory_utilization'])])
    
    print(f"Command: {' '.join(cmd)}")
    
    # Start server in background
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    max_wait_time = 300  # 5 minutes
    wait_time = 0
    
    while wait_time < max_wait_time:
        if check_vllm_server(f"http://localhost:{port}"):
            print("✅ vLLM server is ready!")
            return process
        time.sleep(5)
        wait_time += 5
        print(f"   Waiting... ({wait_time}s)")
    
    print("❌ Server failed to start within timeout")
    process.terminate()
    return None


def run_evaluation(model_name, datasets, **eval_kwargs):
    """Run VLMEvalKit evaluation"""
    print(f"📊 Running evaluation with model: {model_name}")
    print(f"📋 Datasets: {', '.join(datasets)}")
    
    # Register vLLM models
    register_vllm_models()
    
        # Import VLMEvalKit run function
        try:
            from vlmeval.config import supported_VLM
            from vlmeval.dataset import SUPPORTED_DATASETS
            
            # Check if model is registered
            if model_name not in supported_VLM:
                print(f"❌ Model {model_name} not found!")
                print("Available models:")
                list_available_models()
                return False
            
            # Check datasets
            for dataset in datasets:
                if dataset not in SUPPORTED_DATASETS:
                    print(f"❌ Dataset {dataset} not found!")
                    print(f"Available datasets: {', '.join(sorted(SUPPORTED_DATASETS))}")
                    return False
        
        print("✅ All checks passed, starting evaluation...")
        
        # Import and run evaluation
        from vlmeval.api import infer_data_root, load_dataset
        from vlmeval.evaluate import evaluate
        
        # Load model
        model = supported_VLM[model_name]()
        
        # Run evaluation for each dataset
        for dataset in datasets:
            print(f"\n🔄 Evaluating on {dataset}...")
            
            # Load dataset
            data_root = infer_data_root()
            dataset_data = load_dataset(dataset, data_root)
            
            # Run inference
            result_file = f"{model_name}_{dataset}.xlsx"
            model.generate_dataset(dataset_data, dataset=dataset, out_file=result_file)
            
            # Run evaluation if supported
            try:
                eval_result = evaluate(result_file, dataset)
                print(f"✅ Evaluation completed for {dataset}")
                print(f"📄 Results saved to: {result_file}")
            except Exception as e:
                print(f"⚠️  Inference completed but evaluation failed: {e}")
                print(f"📄 Raw results saved to: {result_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Run VLM evaluation using vLLM + VLMEvalKit")
    parser.add_argument("--model-path", required=True, help="Path to your VLM model")
    parser.add_argument("--model-name", default="nano_nemotron_vl_vllm", 
                       help="Model name to use in VLMEvalKit")
    parser.add_argument("--datasets", nargs="+", default=["MMBench_DEV_EN"], 
                       help="Datasets to evaluate on")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--dtype", default="auto", help="Model dtype")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max-model-len", type=int, help="Maximum model length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, 
                       help="GPU memory utilization")
    parser.add_argument("--skip-server-start", action="store_true", 
                       help="Skip starting vLLM server (assume already running)")
    parser.add_argument("--list-datasets", action="store_true", help="List available datasets")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    if args.list_datasets:
        from vlmeval.dataset import SUPPORTED_DATASETS
        print("📋 Available datasets:")
        for dataset in sorted(SUPPORTED_DATASETS):
            print(f"   {dataset}")
        return
    
    if args.list_models:
        register_vllm_models()
        list_available_models()
        return
    
    server_process = None
    
    try:
        # Start vLLM server if needed
        if not args.skip_server_start:
            if check_vllm_server(f"http://localhost:{args.port}"):
                print(f"✅ vLLM server already running on port {args.port}")
            else:
                server_process = start_vllm_server(
                    args.model_path,
                    port=args.port,
                    dtype=args.dtype,
                    tensor_parallel_size=args.tensor_parallel_size,
                    max_model_len=args.max_model_len,
                    gpu_memory_utilization=args.gpu_memory_utilization
                )
                if not server_process:
                    print("❌ Failed to start vLLM server")
                    return 1
        
        # Run evaluation
        success = run_evaluation(args.model_name, args.datasets)
        
        if success:
            print("\n🎉 Evaluation completed successfully!")
            return 0
        else:
            print("\n❌ Evaluation failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Clean up server process
        if server_process:
            print("\n🔄 Shutting down vLLM server...")
            server_process.terminate()
            server_process.wait()


if __name__ == "__main__":
    exit(main())
