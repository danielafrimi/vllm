#!/usr/bin/env python3
"""
Simple test of EvalScope with VLMEvalKit backend
"""

from evalscope.run import run_task

def test_simple_evaluation():
    """Run a simple evaluation with EvalScope"""
    
    # Simple task configuration
    task_cfg = {
        "work_dir": "outputs_test",
        "eval_backend": "VLMEvalKit",
        "eval_config": {
            "model": [
                {
                    "type": "/home/dafrimi/projects/models/working_13p41",
                    "name": "CustomAPIModel", 
                    "api_base": "http://localhost:8000/v1/chat/completions",
                    "key": "EMPTY",
                    "temperature": 0.0,
                    "max_tokens": 512,
                    "img_size": -1,
                    "video_llm": False  # Start with images only
                }
            ],
            "data": ["MME-RealWorld-Lite"],  # Small dataset
            "mode": "all",
            "limit": 5,  # Very small test
            "reuse": False,
            "nproc": 1
        }
    }
    
    print("🚀 Running simple EvalScope test...")
    print(f"📊 Dataset: MME-RealWorld-Lite (limit: 5)")
    print(f"🤖 Model: nano_nemotron_vl via vLLM")
    
    try:
        run_task(task_cfg=task_cfg)
        print("✅ EvalScope evaluation completed!")
        return True
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False

if __name__ == "__main__":
    success = test_simple_evaluation()
    exit(0 if success else 1)
