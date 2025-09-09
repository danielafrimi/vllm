"""
Custom VLM wrapper for integrating vLLM models with VLMEvalKit
This wrapper allows VLMEvalKit to evaluate VLM models served by vLLM
"""

import base64
import io
import os
import time
import logging
from typing import List, Optional, Union

import requests
from PIL import Image
from openai import OpenAI

from vlmeval.smp import *
from vlmeval.api.base import BaseAPI


class VLLMWrapper(BaseAPI):
    """
    Custom wrapper to integrate vLLM-served VLM models with VLMEvalKit
    """
    
    def __init__(self, 
                 model_name: str,
                 api_base: str = "http://localhost:8000/v1",
                 api_key: str = "EMPTY",
                 max_tokens: int = 1024,
                 temperature: float = 0.0,
                 **kwargs):
        """
        Initialize the vLLM wrapper
        
        Args:
            model_name: The model name/path used when starting vLLM server
            api_base: vLLM server base URL
            api_key: API key (can be "EMPTY" for local serving)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize OpenAI client for vLLM
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
        )
        
        # Set default kwargs for VLMEvalKit compatibility
        self.default_kwargs = {
            'max_tokens': max_tokens,
            'temperature': temperature
        }
        
        # Set required attributes for VLMEvalKit BaseAPI compatibility
        self.retry = 3
        self.wait_time = 3
        self.wait = 3  # Same as wait_time
        self.verbose = True
        self.logger = logging.getLogger(__name__)
        self.fail_msg = "Request failed"
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test if vLLM server is accessible"""
        try:
            # Try to get model info
            response = requests.get(f"{self.api_base.rstrip('/v1')}/v1/models")
            response.raise_for_status()
            print(f"✅ Successfully connected to vLLM server at {self.api_base}")
        except Exception as e:
            print(f"❌ Failed to connect to vLLM server: {e}")
            print(f"Make sure vLLM server is running at {self.api_base}")
            raise
    
    def use_custom_prompt(self, dataset):
        """Whether to use custom prompt for specific dataset"""
        return False
    
    def build_prompt(self, line, dataset):
        """Build prompt for the given dataset line"""
        if dataset is None:
            dataset = ''
        
        tgt_path = self.dump_image(line, dataset)
        
        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        
        msgs = []
        if hint is not None:
            question = hint + '\n' + question
            
        msgs.append(dict(type='text', value=question))
        msgs.append(dict(type='image', value=tgt_path))
        
        return msgs
    
    def generate_inner(self, inputs, **kwargs):
        """
        Generate response using vLLM server
        
        Args:
            inputs: List of input messages with text and image
            **kwargs: Additional generation parameters
            
        Returns:
            Tuple of (ret_code, answer, log) as expected by VLMEvalKit
        """
        # Process inputs to OpenAI format
        messages = self._process_inputs(inputs)
        
        try:
            # Make API call to vLLM server
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature),
                top_p=kwargs.get('top_p', 1.0),
            )
            
            response = completion.choices[0].message.content
            answer = response.strip() if response else ""
            
            # Return format expected by VLMEvalKit: (ret_code, answer, log)
            return 0, answer, {"success": True, "model": self.model_name}
            
        except Exception as e:
            error_msg = f"Error during generation: {e}"
            self.logger.error(error_msg)
            # Return error format: (ret_code, answer, log)
            return 1, "", {"success": False, "error": error_msg}
    
    def _process_inputs(self, inputs):
        """Convert VLMEvalKit inputs to OpenAI chat format"""
        messages = []
        content = []
        
        for inp in inputs:
            if inp['type'] == 'text':
                content.append({
                    "type": "text", 
                    "text": inp['value']
                })
            elif inp['type'] == 'image':
                # Convert image to base64 or use URL
                image_content = self._process_image(inp['value'])
                content.append(image_content)

                ## todo add here support for video
        
        messages.append({
            "role": "user",
            "content": content
        })
        
        return messages
    
    def _process_image(self, image_path):
        """Process image for API call"""
        if image_path.startswith('http'):
            # URL image
            return {
                "type": "image_url",
                "image_url": {"url": image_path}
            }
        else:
            # Local image file - convert to base64
            try:
                with open(image_path, 'rb') as image_file:
                    image_data = image_file.read()
                    base64_image = base64.b64encode(image_data).decode('utf-8')
                    
                # Detect image format
                image_format = image_path.split('.')[-1].lower()
                if image_format in ['jpg', 'jpeg']:
                    mime_type = 'image/jpeg'
                elif image_format == 'png':
                    mime_type = 'image/png'
                else:
                    mime_type = 'image/jpeg'  # default
                
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                }
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                return {
                    "type": "text",
                    "text": f"[Error loading image: {image_path}]"
                }


# Factory function to create model instances
def create_vllm_model(model_name: str, **kwargs):
    """Factory function to create VLLMWrapper instances"""
    return VLLMWrapper(model_name=model_name, **kwargs)


# Example usage and model definitions
VLLM_MODELS = {
    # Add your model configurations here
    'nano_nemotron_vl': {
        'model_name': 'NemotronH_Nano_VL',  # This should match your actual model path
        'api_base': 'http://localhost:8000/v1',
        'max_tokens': 1024,
        'temperature': 0.0
    },
}


def get_vllm_model(model_key: str, **override_kwargs):
    """Get a vLLM model instance by key"""
    if model_key not in VLLM_MODELS:
        raise ValueError(f"Model {model_key} not found. Available models: {list(VLLM_MODELS.keys())}")
    
    config = VLLM_MODELS[model_key].copy()
    config.update(override_kwargs)
    
    return VLLMWrapper(**config)
