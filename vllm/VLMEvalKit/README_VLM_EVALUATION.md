# VLM Evaluation with vLLM + VLMEvalKit Integration

This guide explains how to evaluate Vision-Language Models (VLMs) served by vLLM using VLMEvalKit, a comprehensive evaluation framework for large vision-language models.

## Overview

This integration allows you to:
- Serve any VLM model using vLLM for high-performance inference
- Evaluate the model on 50+ vision-language benchmarks through VLMEvalKit
- Get standardized metrics and comparisons with other VLMs

## Architecture Flow

```
┌─────────────────┐    HTTP API    ┌──────────────────┐    Integration    ┌─────────────────┐
│   vLLM Server   │ ◄──────────── │  VLLMWrapper     │ ◄─────────────── │   VLMEvalKit    │
│                 │                │  (Custom Bridge) │                   │                 │
│ • Model Loading │                │ • API Translation│                   │ • Dataset Loading│
│ • GPU Inference │                │ • Format Convert │                   │ • Evaluation     │
│ • OpenAI API    │                │ • Error Handling │                   │ • Metrics        │
└─────────────────┘                └──────────────────┘                   └─────────────────┘
```


Ensure your VLM model is in a format compatible with vLLM:
- Hugging Face format (recommended)
- Local model files with proper config.json
- Model should support multimodal inputs (images + text)

### 3. Integration Files

You need these key files (located in `vllm/VLMEvalKit/` directory):

#### A. `vllm_vlmevalkit_wrapper.py` - The Bridge Component
This file contains the `VLLMWrapper` class that:
- Inherits from VLMEvalKit's `BaseAPI`
- Handles communication with vLLM server via OpenAI-compatible API
- Converts VLMEvalKit format to OpenAI chat completion format
- Processes images (base64 encoding, URL handling)
- Returns results in VLMEvalKit expected format

#### B. VLMEvalKit Configuration
The integration modifies VLMEvalKit's config to register your vLLM model:
```python
# Added to vlmeval/config.py
vllm_models = {
    'nano_nemotron_vl_vllm': partial(  # This is the model name you use in commands
        VLLMWrapper,
        model_name="/home/dafrimi/projects/models/working_13p41",  # This matches your vLLM serve path
        api_base="http://localhost:8000/v1",
        max_tokens=1024,
        temperature=0.0
    ),
}
```

**Important Model Name Configuration**: 
- The **key** (`nano_nemotron_vl_vllm`) is what you use in VLMEvalKit commands
- The **`model_name`** parameter should match exactly what you serve with vLLM
- In your case: 
  - vLLM serve command: `vllm serve /home/dafrimi/projects/models/working_13p41`
  - VLMEvalKit model name: `nano_nemotron_vl_vllm`
  - Wrapper model_name: `/home/dafrimi/projects/models/working_13p41`

## Usage Flow

### Step 1: Start vLLM Server

```bash
# Your exact command (working example)
vllm serve /home/dafrimi/projects/models/working_13p41 \
    --runner generate \
    --max-model-len 8192 \
    --trust-remote-code \
    --gpu-memory-utilization 0.95

# General template
vllm serve /path/to/your/model \
    --runner generate \
    --max-model-len 8192 \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --port 8000
```

**Server Startup Process:**
1. Model loading and weight initialization
2. GPU memory allocation and KV cache setup
3. CUDA graph compilation for optimization
4. API server startup on specified port
5. Ready to accept requests at `http://localhost:8000`

### Step 2: Verify Server is Running

```bash
# Test server connectivity
curl http://localhost:8000/v1/models

# Expected response:
# {
#   "object": "list",
#   "data": [{"id": "/path/to/your/model", "object": "model", ...}]
# }
```

### Step 3: Run Evaluation

#### Quick Test (5 samples)
```bash
python -m vlmeval \
    --data MMBench_DEV_EN \
    --model nano_nemotron_vl_vllm \
    --limit 5 \
    --verbose \
    --work-dir ./results
```

#### Multiple Datasets
```bash
python -m vlmeval \
    --data MMBench_DEV_EN MME SEEDBench_IMG \
    --model nano_nemotron_vl_vllm \
    --limit 50 \
    --verbose
```

#### Full Evaluation
```bash
python -m vlmeval \
    --data MMBench_DEV_EN \
    --model nano_nemotron_vl_vllm \
    --verbose
```

## Evaluation Process Flow

### 1. **Dataset Loading**
```
VLMEvalKit loads dataset → Downloads if needed → Processes to standard format
```

### 2. **Sample Processing**
For each sample:
```
Image + Question → VLLMWrapper → OpenAI API format → vLLM Server → Response → VLMEvalKit format
```

**Detailed Flow:**
1. **Input Processing**: VLMEvalKit provides image path + question text
2. **Format Conversion**: Wrapper converts to OpenAI chat completion format:
   ```json
   {
     "messages": [{
       "role": "user",
       "content": [
         {"type": "text", "text": "What is in this image?"},
         {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
       ]
     }]
   }
   ```
3. **API Call**: HTTP POST to vLLM server `/v1/chat/completions`
4. **Response Processing**: Extract text response and return to VLMEvalKit
5. **Progress Tracking**: Real-time progress bar showing completion status

### 3. **Evaluation & Scoring**
```
Model Responses → Answer Extraction → Metric Calculation → Final Scores
```

## Available Datasets

### Popular Benchmarks
- **MMBench_DEV_EN**: Multi-modal reasoning benchmark
- **MME**: Comprehensive evaluation with 14 subtasks
- **SEEDBench_IMG**: Image understanding benchmark
- **MMMU_DEV_VAL**: Multi-modal multi-university benchmark
- **MathVista**: Mathematical reasoning with visuals
- **AI2D**: Diagram understanding
- **ChartQA**: Chart and graph comprehension

### List All Available Datasets
```bash
python -c "from vlmeval.dataset import SUPPORTED_DATASETS; print('\n'.join(sorted(SUPPORTED_DATASETS)))"
```

## Configuration Options

### VLLMWrapper Parameters
```python
VLLMWrapper(
    model_name="/path/to/model",      # Model path or name served by vLLM
    api_base="http://localhost:8000/v1",  # vLLM server endpoint
    api_key="EMPTY",                  # API key (can be empty for local)
    max_tokens=1024,                  # Maximum tokens to generate
    temperature=0.0,                  # Sampling temperature
)
```

### vLLM Server Parameters
```bash
vllm serve MODEL_PATH \
    --runner generate \               # Use generate runner for VLMs
    --max-model-len 8192 \           # Context length
    --trust-remote-code \            # Allow custom model code
    --gpu-memory-utilization 0.95 \  # GPU memory usage
    --tensor-parallel-size 2 \       # Multi-GPU (if available)
    --port 8000 \                    # Server port
    --dtype auto                     # Automatic dtype selection
```

### VLMEvalKit Parameters
```bash
python -m vlmeval \
    --data DATASET_NAME \            # Dataset(s) to evaluate on
    --model MODEL_NAME \             # Registered model name
    --limit N \                      # Limit samples (for testing)
    --verbose \                      # Verbose output
    --work-dir ./results \           # Output directory
    --mode all \                     # 'all' or 'infer' (inference only)
    --retry 3                        # Retry failed requests
```

## Results and Output

### Output Structure
```
./results/
├── MODEL_NAME_DATASET.xlsx         # Main results file
├── MODEL_NAME_DATASET_eval.xlsx    # Evaluation metrics
└── logs/                           # Detailed logs
```

### Result Interpretation
- **Overall Score**: Aggregate performance across all tasks
- **Sub-task Scores**: Performance on specific capabilities
- **Accuracy Metrics**: Exact match, fuzzy match scores
- **Error Analysis**: Failed samples and error types

## Troubleshooting

### Common Issues

#### 1. **Server Connection Failed**
```bash
# Error: Connection refused
# Solution: Ensure vLLM server is running
curl http://localhost:8000/v1/models
```

#### 2. **Out of Memory**
```bash
# Reduce GPU memory usage or model length
vllm serve MODEL_PATH --gpu-memory-utilization 0.8 --max-model-len 4096
```

#### 3. **Model Loading Issues**
```bash
# Ensure trust-remote-code if using custom models
vllm serve MODEL_PATH --trust-remote-code
```

#### 4. **Slow Inference**
```bash
# Enable optimizations
vllm serve MODEL_PATH --runner generate --gpu-memory-utilization 0.95
```

### Debug Mode
```bash
# Enable detailed logging
python -m vlmeval --data DATASET --model MODEL --verbose --limit 5
```

## Performance Tips

### 1. **Server Optimization**
- Use `--runner generate` for VLMs
- Set `--gpu-memory-utilization 0.9+` for better memory usage
- Enable tensor parallelism for multi-GPU setups

### 2. **Evaluation Optimization**
- Start with `--limit` for testing
- Use `--mode infer` to skip evaluation if only generating responses
- Run multiple datasets in parallel if you have resources

### 3. **Memory Management**
- Monitor GPU memory usage during evaluation
- Adjust `--max-model-len` based on your dataset requirements
- Consider batch size adjustments in vLLM config

## Example: Complete Workflow

```bash
# 1. Navigate to VLMEvalKit directory
cd vllm/VLMEvalKit

# 2. Start vLLM server (from project root)
cd ../..
vllm serve /home/dafrimi/projects/models/working_13p41 \
    --runner generate \
    --max-model-len 8192 \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 &

# 3. Wait for server to start (check logs)
sleep 30

# 4. Test connection
curl http://localhost:8000/v1/models

# 5. Run quick evaluation
cd vllm/VLMEvalKit
python -m vlmeval \
    --data MMBench_DEV_EN \
    --model nano_nemotron_vl_vllm \
    --limit 10 \
    --verbose

# 6. Run full evaluation
python -m vlmeval \
    --data MMBench_DEV_EN MME SEEDBench_IMG \
    --model nano_nemotron_vl_vllm \
    --verbose \
    --work-dir ./evaluation_results
```

## Advanced Usage

### Custom Model Registration

To add your own model, modify the VLMEvalKit config:

```python
# In vlmeval/config.py (around line 1460)
vllm_models = {
    'your_custom_model_vllm': partial(  # ← This name goes in --model parameter
        VLLMWrapper,
        model_name="/your/model/path",  # ← This must match your vLLM serve path
        api_base="http://localhost:8000/v1",
        max_tokens=2048,
        temperature=0.1,
    ),
}
```

**Example for your setup:**
```bash
# 1. Your vLLM serve command
vllm serve /home/dafrimi/projects/models/working_13p41 --runner generate

# 2. Your config entry
'nano_nemotron_vl_vllm': partial(
    VLLMWrapper,
    model_name="/home/dafrimi/projects/models/working_13p41",  # Must match serve path
    api_base="http://localhost:8000/v1",
)

# 3. Your evaluation command
python -m vlmeval --model nano_nemotron_vl_vllm --data MMBench_DEV_EN
```

### Multi-GPU Setup
```bash
# Tensor parallelism across 2 GPUs
vllm serve MODEL_PATH --tensor-parallel-size 2

# Pipeline parallelism (for very large models)
vllm serve MODEL_PATH --pipeline-parallel-size 2
```

### Batch Evaluation
```bash
# Evaluate multiple models
for model in model1_vllm model2_vllm; do
    python -m vlmeval --data MMBench_DEV_EN --model $model --verbose
done
```

## Integration Benefits

1. **High Performance**: vLLM's optimized inference engine
2. **Standardized Evaluation**: Consistent metrics across models
3. **Comprehensive Benchmarks**: 50+ vision-language datasets
4. **Easy Comparison**: Compare with other VLMs on leaderboards
5. **Scalable**: Support for multi-GPU and large models
6. **Flexible**: Easy to add new models and datasets

## Conclusion

This integration provides a powerful way to evaluate VLMs with minimal setup overhead. The combination of vLLM's inference optimization and VLMEvalKit's comprehensive evaluation suite enables thorough assessment of vision-language model capabilities across diverse tasks and domains.

For questions or issues, refer to:
- [vLLM Documentation](https://docs.vllm.ai/)
- [VLMEvalKit Repository](https://github.com/open-compass/VLMEvalKit)
