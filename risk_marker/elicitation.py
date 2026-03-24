'''
This is the main file for testing PT parameters
'''

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator import VLLMGenerator
from prompt import gen_batch_prompt, intro_text, format_lottery, to_text
from tqdm import tqdm
import re
import json

generator = None
model_name = None
output_file = None


def setup(model_path, base_url, output_filename):
    global generator, model_name, output_file
    model_name = model_path
    output_file = output_filename
    generator = VLLMGenerator(model_name=model_path, base_url=base_url)
    print(f"Using vLLM endpoint for model: {model_path} at {base_url}")


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


def sort_sequence_data(data):
    max_row = max(seq[0] for seq in data["sequence"]) + 1
    max_col = max(seq[1] for seq in data["sequence"]) + 1
    result = [[None for _ in range(max_col)] for _ in range(max_row)]
    for idx, (row, col) in enumerate(data["sequence"]):
        result[row][col] = data["choices"][idx]
    result[2] = result[2][:7]
    return result


def parse_choice(text):
    """Extract the last [K] or [U] from the response."""
    matches = re.findall(r'\[(K|U)\]', text)
    return matches[-1] if matches else 'None'


def elicit(batchsize=16, history=10):
    global generator, model_name, output_file

    full_seqs, batch_prompts = gen_batch_prompt(batchsize=batchsize)
    batch_prompts = to_text(batch_prompts, model_name=model_name)

    batch_responses = [[] for _ in range(batchsize)]
    responses = [[] for _ in range(batchsize)]

    for round_idx in range(35):
        round_messages = []

        for i in range(batchsize):
            messages = [{"role": "user", "content": intro_text}]

            number = 0
            for j in range(round_idx):
                if j > round_idx - history:
                    if round_idx - history < 0:
                        number = j + 1
                    else:
                        number = j - (round_idx - history)

                    lottery_prompt = (
                        f"\nHere is lottery {number}\n"
                        + batch_prompts[i][j + 1]
                    )
                    messages.append({"role": "user", "content": lottery_prompt})

                    if j < len(batch_responses[i]):
                        if batch_responses[i][j] in ['K', 'U']:
                            messages.append({
                                "role": "assistant",
                                "content": f"\n[{batch_responses[i][j]}]\n"
                            })
                        else:
                            messages.append({
                                "role": "assistant",
                                "content": "\n[ ]\n"
                            })

            lot_prompt = (
                f"\nHere is lottery {number + 1}\n"
                + batch_prompts[i][round_idx + 1]
            )
            messages.append({"role": "user", "content": lot_prompt})

            # Merge intro and first lottery into a single user message
            if len(messages) > 1 and messages[1]["role"] == "user":
                messages[0]["content"] += messages[1]["content"]
                del messages[1]

            round_messages.append(messages)

        resp_texts = generator.chat_batch(
            round_messages, max_tokens=2048, temperature=0.7
        )

        for i in range(batchsize):
            resp_text = resp_texts[i]
            choice = parse_choice(resp_text)
            batch_responses[i].append(choice)
            responses[i].append(resp_text)

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
    models_to_test = [
        # "meta-llama/Llama-3.1-8B-Instruct",
        # "Qwen/Qwen2.5-7B-Instruct",
        # "Qwen/Qwen2.5-14B-Instruct",
        # "Qwen/Qwen2.5-32B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3"
    ]

    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    output_dir = "result"
    sample_num = 2
    batchsize = 1
    history = 15

    for model_name in models_to_test:
        model_simple_name = model_name.replace("/", "_")
        output_file = os.path.join(output_dir, f"{model_simple_name}.json")

        setup(model_name, base_url, output_file)

        for run_id in tqdm(range(sample_num)):
            elicit(batchsize=batchsize, history=history)
            print(f"Responses saved to {output_file}")
