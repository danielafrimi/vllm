#!/usr/bin/env python3
"""
Simple script to run VLM evaluation using vLLM + VLMEvalKit
This script registers the vLLM model and then uses VLMEvalKit's built-in evaluation system
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def register_vllm_models():
    """Register vLLM models with VLMEvalKit"""
    from vlmeval_vllm_config import register_vllm_models
    return register_vllm_models()

def main():
    parser = argparse.ArgumentParser(description="Run VLM evaluation using vLLM + VLMEvalKit")
    parser.add_argument("--model-name", default="nano_nemotron_vl_vllm", 
                       help="Model name to use in VLMEvalKit")
    parser.add_argument("--datasets", nargs="+", default=["MME"], 
                       help="Datasets to evaluate on")
    parser.add_argument("--work-dir", default="./vlm_eval_results", 
                       help="Output directory for results")
    parser.add_argument("--mode", choices=["all", "infer"], default="all",
                       help="Evaluation mode: 'all' for inference+eval, 'infer' for inference only")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--limit", type=int, help="Limit number of samples to evaluate")
    
    args = parser.parse_args()
    
    print("🚀 Starting vLLM + VLMEvalKit Evaluation")
    print("=" * 50)
    
    try:
        # Register vLLM models
        print("📋 Registering vLLM models...")
        registered_models = register_vllm_models()
        print(f"✅ Registered models: {registered_models}")
        
        # Check if our model is registered
        if args.model_name not in registered_models:
            print(f"❌ Model {args.model_name} not found in registered models!")
            return 1
        
        # Prepare VLMEvalKit command
        vlm_cmd = [
            sys.executable, "-m", "vlmeval",
            "--data"] + args.datasets + [
            "--model", args.model_name,
            "--work-dir", args.work_dir,
            "--mode", args.mode
        ]
        
        if args.verbose:
            vlm_cmd.append("--verbose")
        
        if args.limit:
            vlm_cmd.extend(["--limit", str(args.limit)])
        
        print(f"🔄 Running VLMEvalKit command:")
        print(f"   {' '.join(vlm_cmd)}")
        print()
        
        # Run VLMEvalKit evaluation
        result = subprocess.run(vlm_cmd, cwd=current_dir)
        
        if result.returncode == 0:
            print("\n🎉 Evaluation completed successfully!")
            print(f"📁 Results saved to: {args.work_dir}")
            return 0
        else:
            print(f"\n❌ Evaluation failed with return code: {result.returncode}")
            return result.returncode
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
