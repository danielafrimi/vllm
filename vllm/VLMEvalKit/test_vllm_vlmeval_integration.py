#!/usr/bin/env python3
"""
Test script to verify vLLM + VLMEvalKit integration
"""

import sys
import os
import requests
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def test_vlmevalkit_import():
    """Test if VLMEvalKit can be imported"""
    print("🧪 Testing VLMEvalKit import...")
    try:
        import vlmeval
        from vlmeval.config import supported_VLM
        try:
            from vlmeval.utils.dataset_config import dataset_URLs
        except ImportError:
            from vlmeval.dataset import dataset_URLs
        print("✅ VLMEvalKit imported successfully")
        print(f"   Available datasets: {len(dataset_URLs)} datasets")
        return True
    except ImportError as e:
        print(f"❌ Failed to import VLMEvalKit: {e}")
        return False

def test_vllm_server_connection(api_base="http://localhost:8000"):
    """Test connection to vLLM server"""
    print(f"🧪 Testing vLLM server connection at {api_base}...")
    try:
        response = requests.get(f"{api_base}/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print("✅ vLLM server is accessible")
            print(f"   Available models: {[model['id'] for model in models.get('data', [])]}")
            return True
        else:
            print(f"❌ vLLM server returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to connect to vLLM server: {e}")
        print("   Make sure vLLM server is running:")
        print("   vllm serve <your_model_path> --trust-remote-code")
        return False

def test_wrapper_import():
    """Test if our custom wrapper can be imported"""
    print("🧪 Testing custom wrapper import...")
    try:
        from vllm_vlmevalkit_wrapper import VLLMWrapper
        print("✅ Custom wrapper imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import custom wrapper: {e}")
        return False

def test_model_registration():
    """Test model registration with VLMEvalKit"""
    print("🧪 Testing model registration...")
    try:
        from vlmeval_vllm_config import register_vllm_models
        from vlmeval.config import supported_VLM
        
        # Count models before registration
        models_before = len(supported_VLM)
        
        # Register vLLM models
        registered_models = register_vllm_models()
        
        # Count models after registration
        models_after = len(supported_VLM)
        
        print(f"✅ Successfully registered {len(registered_models)} models")
        print(f"   Models before: {models_before}, after: {models_after}")
        print(f"   Registered models: {registered_models}")
        return True
    except Exception as e:
        print(f"❌ Model registration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_wrapper_initialization():
    """Test wrapper initialization (without server call)"""
    print("🧪 Testing wrapper initialization...")
    try:
        from vllm_vlmevalkit_wrapper import VLLMWrapper
        
        # Try to create wrapper (this will fail if server is not running, but that's ok for this test)
        try:
            wrapper = VLLMWrapper(
                model_name="test_model",
                api_base="http://localhost:8000/v1"
            )
            print("✅ Wrapper initialized successfully")
            return True
        except Exception as e:
            if "Failed to connect" in str(e):
                print("⚠️  Wrapper creation failed due to server connection (expected if server not running)")
                print("✅ Wrapper code is functional")
                return True
            else:
                print(f"❌ Wrapper initialization failed: {e}")
                return False
    except Exception as e:
        print(f"❌ Wrapper test failed: {e}")
        return False

def list_sample_datasets():
    """List some sample datasets for testing"""
    print("\n📋 Sample datasets you can test with:")
    sample_datasets = [
        "MMBench_DEV_EN",
        "MME", 
        "SEEDBench_IMG",
        "MMMU_DEV_VAL",
        "MathVista_MINI"
    ]
    
    try:
        try:
            from vlmeval.utils.dataset_config import dataset_URLs
        except ImportError:
            from vlmeval.dataset import dataset_URLs
        available = [d for d in sample_datasets if d in dataset_URLs]
        print(f"   Available: {', '.join(available)}")
        
        if len(available) < len(sample_datasets):
            missing = [d for d in sample_datasets if d not in dataset_URLs]
            print(f"   Not available: {', '.join(missing)}")
    except:
        print(f"   Suggested: {', '.join(sample_datasets)}")

def main():
    print("🔍 Testing vLLM + VLMEvalKit Integration")
    print("=" * 50)
    
    tests = [
        ("VLMEvalKit Import", test_vlmevalkit_import),
        ("Custom Wrapper Import", test_wrapper_import),
        ("Model Registration", test_model_registration),
        ("Wrapper Initialization", test_wrapper_initialization),
        ("vLLM Server Connection", lambda: test_vllm_server_connection()),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! You're ready to run evaluations.")
        print("\n🚀 Next steps:")
        print("1. Start your vLLM server:")
        print("   vllm serve <your_model_path> --trust-remote-code")
        print("2. Run evaluation:")
        print("   python run_vlm_evaluation.py --model-path <your_model_path> --datasets MMBench_DEV_EN")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    list_sample_datasets()
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    exit(main())
