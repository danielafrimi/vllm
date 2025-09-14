#!/usr/bin/env python3
"""
Run VLM evaluation using EvalScope with VLMEvalKit backend
This uses the cleaner EvalScope interface instead of direct VLMEvalKit
"""

import json
from pathlib import Path
from evalscope.run import run_task
from evalscope.summarizer import Summarizer

def run_evaluation_with_config(config_file="config_eval.json"):
    """Run evaluation using EvalScope with JSON config"""
    
    config_path = Path(config_file)
    if not config_path.exists():
        print(f"❌ Config file {config_file} not found!")
        return False
    
    print(f"📋 Loading config from {config_file}")
    
    try:
        # Load and run evaluation
        print("🚀 Starting evaluation with EvalScope...")
        run_task(task_cfg=str(config_path))
        
        print("📊 Getting evaluation report...")
        report_list = Summarizer.get_report_from_cfg(str(config_path))
        print(f"✅ Evaluation completed!")
        print(f"📄 Report list: {report_list}")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_evaluation_with_dict():
    """Run evaluation using EvalScope with Python dict config"""
    
    task_cfg_dict = {
        "work_dir": "outputs",
        "eval_backend": "VLMEvalKit", 
        "eval_config": {
            "model": [
                {
                    "type": "/home/dafrimi/projects/models/working_13p41",
                    "name": "CustomAPIModel",
                    "api_base": "http://localhost:8000/v1/chat/completions",
                    "key": "EMPTY",
                    "temperature": 0.0,
                    "max_tokens": 1024,
                    "img_size": -1,
                    "video_llm": True  # Enable video support
                }
            ],
            "data": ["MME-RealWorld-Lite", "MMBench_DEV_EN"],  # Start with image datasets
            "mode": "all",
            "limit": 10,
            "reuse": False,
            "nproc": 4
        }
    }
    
    try:
        print("🚀 Starting evaluation with EvalScope (dict config)...")
        run_task(task_cfg=task_cfg_dict)
        
        print("📊 Getting evaluation report...")
        report_list = Summarizer.get_report_from_cfg(task_cfg_dict)
        print(f"✅ Evaluation completed!")
        print(f"📄 Report list: {report_list}")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run VLM evaluation using EvalScope")
    parser.add_argument("--config", default="config_eval.json", help="Config file path")
    parser.add_argument("--use-dict", action="store_true", help="Use Python dict config instead of JSON file")
    parser.add_argument("--datasets", nargs="+", help="Override datasets to evaluate")
    parser.add_argument("--limit", type=int, help="Override sample limit")
    
    args = parser.parse_args()
    
    if args.use_dict:
        success = run_evaluation_with_dict()
    else:
        success = run_evaluation_with_config(args.config)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
