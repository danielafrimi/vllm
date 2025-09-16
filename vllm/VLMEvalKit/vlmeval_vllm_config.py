"""
Configuration file to register vLLM models with VLMEvalKit
"""

import sys
import os

# Add the current directory to Python path so we can import our wrapper
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from vllm_vlmevalkit_wrapper import VLLMWrapper

# Import VLMEvalKit configuration
try:
    from vlmeval.config import supported_VLM
    from vlmeval.utils import track_progress_rich
    print("✅ VLMEvalKit imported successfully")
except ImportError as e:
    print(f"❌ Failed to import VLMEvalKit: {e}")
    print("Make sure VLMEvalKit is installed: pip install vlmeval")
    sys.exit(1)


def register_vllm_models():
    """Register vLLM models with VLMEvalKit"""
    
    # Define your vLLM model configurations
    vllm_models = {
        # Nano Nemotron VL model
        'nano_nemotron_vl_vllm': lambda: VLLMWrapper(
            model_name="/home/dafrimi/projects/models/vlm_update_ckpt",  # Your actual model path
            api_base="http://localhost:8081/v1",  # Fixed port to match standard vLLM port
            max_tokens=1024,
            temperature=0.0
        ),
        
        # You can add more models here
        # 'your_other_model_vllm': lambda: VLLMWrapper(
        #     model_name="your_model_path",
        #     api_base="http://localhost:8001/v1",  # different port if needed
        #     max_tokens=512,
        #     temperature=0.1
        # ),
    }
    
    # Register models with VLMEvalKit
    for model_name, model_factory in vllm_models.items():
        supported_VLM[model_name] = model_factory
        print(f"✅ Registered {model_name} with VLMEvalKit")
    
    return list(vllm_models.keys())


def list_available_models():
    """List all available models including newly registered vLLM models"""
    print("\n📋 Available VLM models:")
    print("=" * 50)
    
    for model_name in sorted(supported_VLM.keys()):
        if 'vllm' in model_name.lower():
            print(f"🚀 {model_name} (vLLM)")
        else:
            print(f"   {model_name}")


if __name__ == "__main__":
    # Register models when this script is run
    registered_models = register_vllm_models()
    print(f"\n✅ Successfully registered {len(registered_models)} vLLM models")
    list_available_models()
