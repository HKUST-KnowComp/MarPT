'''
This is the main file for testing PT parameters
'''

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator import VLLMGenerator
from prompt import gen_batch_prompt, intro_text, format_lottery
from marker import markers, probs
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


def parse_choice(text):
    """Extract the last [K] or [U] from the response."""
    matches = re.findall(r'\[(K|U)\]', text)
    return matches[-1] if matches else 'None'


def elicit(marker_idx, batchsize=16):
    global generator, model_name, output_file

    full_seqs, batch_prompts = gen_batch_prompt(markers[marker_idx], batchsize)

    batch_responses = [[] for _ in range(batchsize)]
    responses = [[] for _ in range(batchsize)]

    for round_idx in range(len(probs)):
        input_texts = []

        for i in range(batchsize):
            current_prompt = ""

            if round_idx == 0:
                current_prompt += intro_text

            for j in range(round_idx):
                current_prompt += batch_prompts[i][j + 1]
                if j < len(batch_responses[i]):
                    if batch_responses[i][j] in ['K', 'U']:
                        current_prompt += f"\n[{batch_responses[i][j]}]\n"
                    else:
                        current_prompt += f"\n[ ]\n"

            current_prompt += batch_prompts[i][round_idx + 1]
            input_texts.append(current_prompt)

        resp_texts = generator.complete_batch(
            input_texts, max_tokens=2048, temperature=0.0
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
            "marker": markers[marker_idx],
            "choices": batch_responses[i],
            "outputs": responses[i]
        }

        sorted_pairs = sorted(
            zip(result["sequence"], result["choices"], result["outputs"]),
            key=lambda x: x[0]
        )
        sorted_sequence, sorted_choices, sorted_outputs = zip(*sorted_pairs)

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
        # "Qwen/Qwen2.5-7B-Instruct",
        # "Qwen/Qwen2.5-14B-Instruct",
        # "Qwen/Qwen2.5-32B-Instruct",
        # "mistralai/Mistral-7B-Instruct-v0.3",
        # "microsoft/phi-4",
        # "allenai/OLMo-2-1124-7B-Instruct",
        # "Qwen/QwQ-32B",
        "Qwen/Qwen3-32B"
    ]

    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    output_dir = "result"
    sample_num = 1024
    batchsize = 1

    for model_name in models_to_test:
        model_simple_name = model_name.replace("/", "_")
        output_file = os.path.join(output_dir, f"{model_simple_name}.json")

        setup(model_name, base_url, output_file)

        marker_left = range(len(markers))
        total = len(marker_left) * sample_num

        with tqdm(total=total, desc="Overall Progress") as pbar:
            for marker_idx in marker_left:
                for run_id in range(sample_num):
                    elicit(marker_idx=marker_idx, batchsize=batchsize)
                    print(f"Responses saved to {output_file}")
                    pbar.update(1)
