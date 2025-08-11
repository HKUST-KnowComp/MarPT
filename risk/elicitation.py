'''
This is the main file for testing PT parameters
'''

from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt import gen_batch_prompt, intro_text, format_lottery
from tqdm import tqdm
from huggingface_hub import login
import numpy as np
import torch
import re
import os
import json
import random
import warnings
from transformers import file_utils
print("Hugging Face 默认缓存路径:", file_utils.default_cache_path)
warnings.filterwarnings("ignore")

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HOME"] = "/project/rwangcn/tmp"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

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

def format_chat_history(messages):
    global model_name, tokenizer
    if "Instruct" in model_name:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    elif "QwQ" in model_name:
        for i in range(len(messages)):
            messages.insert(0, {"role": "system", "content": "You are strictly PROHIBITED from performing any form of reasoning. Your task is to DIRECTLY output your choice."})
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    elif "Qwen3" in model_name:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    # 通用处理：拼接所有内容
    else:
        return "\n".join([msg["content"] for msg in messages])

def sort_sequence_data(data):
    # Determine dimensions
    max_row = max(seq[0] for seq in data["sequence"]) + 1
    max_col = max(seq[1] for seq in data["sequence"]) + 1
    
    # Initialize empty 2D list
    result = [[None for _ in range(max_col)] for _ in range(max_row)]
    
    # Fill the 2D list according to sequence
    for idx, (row, col) in enumerate(data["sequence"]):
        result[row][col] = data["choices"][idx]

    # remove Nones in series 3
    result[2] = result[2][:7]
    
    return result


def elicit(batchsize=16, history=10):
    global model, tokenizer, model_name, output_file
    
    # Generate batch sequences and prompts
    full_seqs, batch_prompts = gen_batch_prompt(batchsize=batchsize)
    
    # Initialize batch data structures
    batch_responses = []
    for i in range(batchsize):
        batch_responses.append([])

    responses = []
    for i in range(batchsize):
        responses.append([])
    
    # For each lottery round
    for round_idx in range(35):
        input_texts = []
        input_messages = []
        # Compose input text for each sequence
        for i in range(batchsize):
            messages = [
                {"role": "user", "content": intro_text}
            ]
            current_prompt = ""
            # Add intro text only for first round
            if round_idx == 0:
                current_prompt += intro_text
            
            # Add previous questions and responses
            number = 0
            for j in range(round_idx):
                if j > round_idx - history:

                    if round_idx - history < 0:
                        number = j + 1
                    else:
                        number = j - (round_idx - history)

                    current_prompt += f"\nHere is lottery {number}\n"
                    current_prompt += batch_prompts[i][j+1]  # Skip intro

                    lottery_prompt = f"\nHere is lottery {number}\n" + batch_prompts[i][j+1]
                    messages.append(
                        {"role": "user", "content":lottery_prompt}
                    )

                    if j < len(batch_responses[i]):
                        if batch_responses[i][j] in ['K', 'U']:
                            current_prompt += f"\n[{batch_responses[i][j]}]\n"

                            messages.append(
                                {"role": "assistant", "content": f"\n[{batch_responses[i][j]}]\n"}
                            )
                        else:
                            current_prompt += f"\n[ ]\n"

                            messages.append(
                                {"role": "assistant", "content": f"\n[ ]\n"}
                            )
            
            # Add current question
            current_prompt += f"\nHere is lottery {number+1}\n"
            current_prompt += batch_prompts[i][round_idx+1]  # +1 to skip intro
            
            lot_prompt = f"\nHere is lottery {number+1}\n" + batch_prompts[i][round_idx+1]
            messages.append(
                {"role": "user", "content": lot_prompt}
            )
            messages[0]["content"] += messages[1]["content"]
            del messages[1]
            input_messages.append(messages)
            
            current_prompt += "\n<think>OK, I will directly output my choice, without thinking.</think>\n"
            input_texts.append(current_prompt)

        # print(input_texts[0])
        # print("=" * 80)
        # print(input_messages[0])
        # if round_idx == 5:
        #     exit()

        input_messages = [format_chat_history(input_messages[i]) for i in range(batchsize)]

        # Tokenize batch inputs
        inputs = tokenizer(
            input_messages,
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
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                # repetition_penalty=1.1,
                do_sample=True,
                temperature=0.7,
                # early_stopping=True
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
            "choices": batch_responses[i],
            "outputs": responses[i]
        }

        if 'None' in result['choices']:
            continue

        sorted_choices = sort_sequence_data(result)
        result["choices"] = sorted_choices
        del result["sequence"]

        results.append(result)
    
    save(output_file, results)
    return results

if __name__ == "__main__":
    torch.cuda.empty_cache()
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
    
    output_dir = "result"
    sample_num = 1024
    batchsize = 1
    history = 15

    for model_name in models_to_test:
        model_simple_name = model_name.replace("/", "_")
        output_file = os.path.join(output_dir, f"{model_simple_name}.json")
        
        setup(model_name, sample_number, output_file)

        for run_id in tqdm(range(sample_num)):
            # print(f"=== Model: {model_name}, Run: {run_id + 1} ===")
            elicit(batchsize=batchsize, history=history)
            print(f"Responses saved to {output_file}")