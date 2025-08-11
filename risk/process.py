'''
This file grab data from /result and count choices, store in /processed
'''

from mle import perform_MLE
import json
import numpy as np

def load(path):
    with open(path, 'r') as f:
        return list(json.load(f))

def extract_choices(choices):
    series = [[], [], []]
    for index, item in enumerate(choices):
        if index < 256:
            series[0].append(item['choices'][0])
            series[1].append(item['choices'][1])
            series[2].append(item['choices'][2])
    return series

def count_KU(choices):
    count = np.zeros( (len(choices[0]), 2), dtype=int )
    for item in choices:
        for i, choice in enumerate(item):
            if choice == 'K':
                count[i][0] += 1
            elif choice == 'U':
                count[i][1] += 1
    return np.array(count)

def process(filename, output_path, model_name):
    
    result = load(filename)
    choices = extract_choices(result)
    data = [count_KU(choice).tolist() for choice in choices]
    result = perform_MLE(data, model_name=model_name)
    result['data'] = list(data)

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    for input_path in [
        "result/meta-llama_Llama-3.1-8B-Instruct.json",
        "result/mistralai_Mistral-7B-Instruct-v0.3.json",
        "result/Qwen_Qwen2.5-7B-Instruct.json",
        "result/Qwen_Qwen2.5-14B-Instruct.json",
        "result/Qwen_Qwen2.5-32B-Instruct.json",
        "result/Qwen_Qwen3-32B.json"
    ]:
        if "Qwen" in input_path and "7B" in input_path:
            model_name = "Qwen7B"
        elif "Qwen" in input_path and "14B" in input_path:
            model_name = "Qwen14B"
        elif "Qwen2.5" in input_path and "32B" in input_path:
            model_name = "Qwen2.5-32B"
        elif "Qwen3" in input_path and "32B" in input_path:
            model_name = "Qwen3-32B"
        elif "Llama" in input_path and "8B" in input_path:
            model_name = "Llama8B"
        elif "Mistral" in input_path and "7B" in input_path:
            model_name = "Mistral7B"
        print(f"==========Model: {model_name}==========")
        process(input_path, output_path=input_path.replace("result", "processed"), model_name=model_name)
        print()
        print()
