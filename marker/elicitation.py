'''
This is the main file for testing PT parameters
'''

from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt import gen_batch_prompt, intro_text, format_lottery
from marker import markers, probs
from tqdm import tqdm
import numpy as np
import torch
import re
import os
import json
import random

os.environ["HF_HOME"] = "/project/rwangcn/tmp"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

# Global config
sample_number = None
model_name = None
output_file = None
model = None
tokenizer = None

def setup(model_path, sample_num, output_filename):
    global model, tokenizer, sample_number, model_name, output_file
    model_name = model_path
    sample_number = sample_num
    output_file = output_filename
    
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    if tokenizer.chat_template is None and hasattr(tokenizer, "default_chat_template"):
        tokenizer.default_chat_template = "{% for message in messages %}{{message['content']}}{% endfor %}"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

def save(filename, responses):    
    existing_data = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            existing_data = json.load(f)
    
    if isinstance(existing_data, list) and isinstance(responses, list):
        combined_data = existing_data + responses
    else:
        combined_data = [existing_data, responses] if existing_data else responses
    
    with open(filename, 'w') as f:
        json.dump(combined_data, f, indent=4)

def elicit(marker_idx, batchsize=16):
    global model, tokenizer, model_name, output_file
    
    # Generate batch sequences and prompts
    full_seqs, batch_prompts = gen_batch_prompt(markers[marker_idx], batchsize)
    
    # Initialize batch data structures
    batch_responses = []
    for i in range(batchsize):
        batch_responses.append([])

    responses = []
    for i in range(batchsize):
        responses.append([])
    
    # For each lottery round
    for round_idx in range(len(probs)):
        input_texts = []
        
        # Compose input text for each sequence
        for i in range(batchsize):
            current_prompt = ""
            
            # Add intro text only for first round
            if round_idx == 0:
                current_prompt += intro_text
            
            # Add previous questions and responses
            for j in range(round_idx):
                current_prompt += batch_prompts[i][j+1]  # Skip intro
                if j < len(batch_responses[i]):
                    if batch_responses[i][j] in ['K', 'U']:
                        current_prompt += f"\n[{batch_responses[i][j]}]\n"
                    else:
                        current_prompt += f"\n[ ]\n"
            
            # Add current question
            current_prompt += batch_prompts[i][round_idx+1]  # +1 to skip intro
            
            input_texts.append(current_prompt)
        
        # Tokenize batch inputs
        inputs = tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(model.device)
        
        # Generate responses
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=8,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                # temparature = 0.7
            )
        
        # Process responses
        for i in range(batchsize):
            # Extract new tokens
            output_tokens = outputs[i][inputs.input_ids.shape[1]:]
            resp_text = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
            
            # Parse choice
            choice_match = re.search(r'\[(K|U)\]', resp_text)
            if choice_match:
                choice = choice_match.group(1)
            # elif len(resp_text) >= 2 and resp_text[0] == '[':
            #     choice = resp_text[1] if resp_text[1] in ['K','U'] else ''
            else:
                choice = 'None'
                
            batch_responses[i].append(choice)
            responses[i].append(resp_text)
    
    # Prepare results for saving
    results = []
    for i in range(batchsize):
        result = {
            "model": model_name,
            "sequence": full_seqs[i],
            "marker": markers[marker_idx],
            "choices": batch_responses[i],
            "outputs": responses[i]
        }

        sorted_pairs = sorted(zip(result["sequence"], result["choices"], result["outputs"]), key=lambda x: x[0])
        sorted_sequence, sorted_choices, sorted_outputs = zip(*sorted_pairs)

        # result["sequence"] = list(sorted_sequence)

        del result["sequence"]
        del result['outputs']
        result["choices"] = list(sorted_choices)
        result["outputs"] = list(sorted_outputs)

        if 'None' in result['choices']:
            continue

        results.append(result)
    
    save(output_file, [results])
    return results

if __name__ == "__main__":
    models_to_test = [
        # "meta-llama/Llama-3.1-8B-Instruct",
        # "Qwen/Qwen2.5-7B-Instruct"
        # "Qwen/Qwen2.5-14B-Instruct"
        # "Qwen/Qwen2.5-32B-Instruct"
        # "mistralai/Mistral-7B-Instruct-v0.3"
        # "microsoft/phi-4"
        # "allenai/OLMo-2-1124-7B-Instruct"
        # "Qwen/QwQ-32B"
        "Qwen/Qwen3-32B"
    ]
    
    output_dir = "/project/rwangcn/MUvBE/marker/result"
    sample_num = 1024
    batchsize = 1

    for model_name in models_to_test:
        model_simple_name = model_name.replace("/", "_")
        output_file = os.path.join(output_dir, f"{model_simple_name}.json")
        
        setup(model_name, sample_number, output_file)

        marker_left = range(len(markers))
        total = len(marker_left) * sample_num
        
        with tqdm(total=total, desc="Overall Progress") as pbar:
            for marker_idx in marker_left:
                for run_id in range(sample_num):
                    # print(f"=== Model: {model_name}, Run: {run_id + 1} ===")
                    elicit(marker_idx=marker_idx, batchsize=batchsize)
                    print(f"Responses saved to {output_file}")
                    pbar.update(1)