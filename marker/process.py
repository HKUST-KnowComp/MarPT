'''
This file grab data from /result and count choices, store in /processed
'''

import json
import os
import copy

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

def switching_point(data, total):
    data = copy.deepcopy(data)
    data.insert(0, [0, total])
    data.append([total, 0])

    # Avoid 0
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] == 0:
                data[i][j] = 0.1
    
    # Record switching
    cross = []
    for i in range(len(data)-1):
        if data[i][0] / total <= 0.5 and data[i+1][0] / total > 0.5:
            cross.append(i)
    if len(cross) >= 2:
        return "No Switching"

    index = cross[0]

    
    if index == 0:
        low_prob = 0
        high_prob = 5
    elif index == 10:
        low_prob = 95
        high_prob = 100
    else:
        low_prob = 5 + (index-1) * 10
        high_prob = low_prob + 10

    y_low = data[index][0] / total
    y_high = data[index+1][0] / total

    prob_at_05 = low_prob + (0.5 - y_low) * (high_prob - low_prob) / (y_high - y_low) if y_high != y_low else (low_prob + high_prob) / 2

    return f"{prob_at_05:.2f}%"

def process(filename, output_path):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            existing_data = json.load(f)
    
    result = {}
    for i in range(len(existing_data)):
        for j in range(len(existing_data[i])):
            if result.get(existing_data[i][j]['marker'], None) == None:
                result[existing_data[i][j]['marker']] = {
                    "Total": 0,
                    "KU": [[0, 0] for _ in range(10)],
                }
            if result[existing_data[i][j]['marker']]['Total'] < 256:
                result[existing_data[i][j]['marker']]['Total'] += 1
                for index, choice in enumerate(existing_data[i][j]['choices']):
                    if choice == 'K':
                        result[existing_data[i][j]['marker']]['KU'][index][0] += 1
                    elif choice == 'U':
                        result[existing_data[i][j]['marker']]['KU'][index][1] += 1    

    print(f"===Model: {filename[37:]}===")
    for key in result.keys():
        switching = switching_point(result[key]['KU'], result[key]['Total'])
        result[key]['Switching'] = switching 
        print(f"Marker: {key:<20}\tSwitching: {switching:<10}\tSample Number: {result[key]['Total']:<10}")
    
    print()
    print()

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
        process(input_path, output_path=input_path.replace("result", "processed"))
