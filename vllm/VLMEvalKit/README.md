# VLMEvalKit Integration for vLLM

This directory contains the complete integration between vLLM and VLMEvalKit for evaluating Vision-Language Models.

## 📁 Files Overview

### Core Integration Files
- **`vllm_vlmevalkit_wrapper.py`** - Main wrapper class that bridges vLLM server and VLMEvalKit
- **`vlmeval_vllm_config.py`** - Configuration helper for registering vLLM models with VLMEvalKit

### Evaluation Scripts
- **`run_vllm_evaluation_simple.py`** - Simple script for running evaluations



## Quick Start

1. **Start vLLM Server:**
   ```bash
   # From project root
   vllm serve /path/to/your/model --runner generate --trust-remote-code 
   ```

2. **Run Evaluation:**
   ```bash
   # From this directory
   python -m vlmeval --data MMBench_DEV_EN --model nano_nemotron_vl_vllm --limit 10
   ```
