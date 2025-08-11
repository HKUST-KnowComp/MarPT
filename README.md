## Experiment Implementation Guide

This repository implements the 3-stage experiment described in the paper.

## Directory Structure

- `/risk`: Tests the baseline PT parameters
- `/marker`: Implements experiment for marker mappings
- `/risk_marker`: Replaces numeric probabilities with epistemic markers and runs PT parameter measurement again

## Experiment Steps

### 1. Setup
```bash
pip install -r requirements.txt
```

### 2. Stage Selection
Navigate to the directory corresponding to the stage you want to run. For example, to run baseline PT measurement:
```bash
cd /risk
```

### 3. Directory Contents
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

### 4. Model Configuration
1. Open `elicitation.py`
2. Go to line 256
3. Select your model to test
4. If your model requires an access token:
   ```bash
   huggingface-cli login
   ```
   Then input your token when prompted

### 5. Running the Experiment
```bash
python elicitation.py
```
Results will be saved under `/result`

### 6. Processing Results
```bash
python process.py > result.txt
```
- Measured parameters will be saved in `/processed`
- Results will also be shown in `result.txt`

### 7. Customizing Marker Substitution (Stage 3 only)
To customize marker substitution:
1. Open `/risk_marker/prompt.py`
2. Go to line 72
3. Modify the function `safe_sub` as needed
```
