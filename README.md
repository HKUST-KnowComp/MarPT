## Experiment Implementation Guide

This repository implements the 3-stage experiment described in the paper.

## Directory Structure

- `/generator`: Unified LLM generation module (calls vLLM-served models via OpenAI-compatible API)
- `/risk`: Tests the baseline PT parameters
- `/marker`: Implements experiment for marker mappings
- `/risk_marker`: Replaces numeric probabilities with epistemic markers and runs PT parameter measurement again

## Experiment Steps

### 1. Setup
```bash
pip install -r requirements.txt
```

### 2. Start vLLM Server

Start a vLLM server with your target model. For example:
```bash
vllm serve Qwen/Qwen3-32B --port 8000
```

The experiments expect an OpenAI-compatible API at `http://localhost:8000/v1` by default.
You can override this by setting the `VLLM_BASE_URL` environment variable:
```bash
export VLLM_BASE_URL=http://your-server:8000/v1
```

### 3. Stage Selection
Navigate to the directory corresponding to the stage you want to run. For example, to run baseline PT measurement:
```bash
cd risk
```

### 4. Directory Contents
Each stage directory has the following structure:
```
risk/
    |---plot/
    |---processed/
    |---result/
    analyze.py
    elicitation.py
    mle.py
    prompt.py
    values_probs.py
```

### 5. Model Configuration
1. Open `elicitation.py`
2. Edit `models_to_test` to select your target model(s)
3. Optionally adjust `base_url`, `sample_num`, and `batchsize`

### 6. Running the Experiment
```bash
python elicitation.py
```
Results will be saved under `result/`

### 7. Processing Results
```bash
python process.py > result.txt
```
- Measured parameters will be saved in `processed/`
- Results will also be shown in `result.txt`

### 8. Customizing Marker Substitution (Stage 3 only)
To customize marker substitution:
1. Open `risk_marker/prompt.py`
2. Modify the function `safe_sub` as needed

### Notes

- The `generator/` module provides a unified `VLLMGenerator` class used by all three stages.
  - `risk/` and `risk_marker/` use **chat completion** (`/v1/chat/completions`) with multi-turn conversations.
  - `marker/` uses **text completion** (`/v1/completions`) with raw prompts.
- For **Qwen3** models, thinking mode is automatically disabled via `chat_template_kwargs`.
- For **QwQ** models, a system message suppressing reasoning is automatically prepended.
