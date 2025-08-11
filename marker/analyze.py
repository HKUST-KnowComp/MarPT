'''
This file visualize the results under /plot
'''

import json
import numpy as np
import matplotlib.pyplot as plt

def plot_first_number_ratio(data, save_path):
    ratios = data[:, 0] / (data[:, 0] + data[:, 1])
    
    plt.figure(figsize=(10, 6))
    plt.plot(ratios, marker='o', linestyle='-', color='b')

    plt.xticks(range(len(ratios)), range(len(ratios)))
    
    plt.title('Ratio of First Number to Total')
    plt.xlabel('Index')
    plt.ylabel('Ratio')
    plt.grid(True)
    
    plt.ylim(0, 1.1)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {save_path}")

def plot_multiple_first_number_ratios(data_list, labels=None, colors=None, save_path='multi_data_ratio.png', show=False):
    if colors is None:
        colors = ['b', 'g', 'r', 'c', 'm', 'y']
    
    plt.figure(figsize=(12, 6))
    
    for i, data in enumerate(data_list):
        ratios = data[:, 0] / (data[:, 0] + data[:, 1])

        label = labels[i] if labels else f'Data {i+1}'
        
        plt.plot(ratios, 
                marker='o', 
                linestyle='-', 
                color=colors[i % len(colors)],
                label=label
        )
    
    x_ticks = np.linspace(0, 100, 21)  # [0, 5, 10, ..., 100]

    plt.xticks(ticks=np.arange(len(data_list[0])),
            labels=[f"{int(x)}" for x in x_ticks]
    )

    plt.title('Fixed: 50% - $60 and 50% - $40 ; Move: x% - $100')
    plt.xlabel('Probability of getting $100')
    plt.ylabel('Fixed Choice Ratio')
    plt.grid(True)
    plt.legend()
    
    plt.ylim(0, 1.1)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {save_path}")

def plot(filename, save_path):
    with open(filename, 'r') as f:
        result = json.load(f)
        for key in result.keys():
            plot_first_number_ratio(np.array(result[key]['KU']), save_path=save_path+"/"+key.replace(" ", "_"))

if __name__ == "__main__":
    for input_path in [
        "/project/rwangcn/MUvBE/marker/processed/meta-llama_Llama-3.1-8B-Instruct.json",
        "/project/rwangcn/MUvBE/marker/processed/mistralai_Mistral-7B-Instruct-v0.3.json",
        "/project/rwangcn/MUvBE/marker/processed/Qwen_Qwen2.5-7B-Instruct.json",
        "/project/rwangcn/MUvBE/marker/processed/Qwen_Qwen2.5-14B-Instruct.json",
        "/project/rwangcn/MUvBE/marker/processed/Qwen_Qwen2.5-32B-Instruct.json"
    ]:
        output_path = "/project/rwangcn/MUvBE/marker/plot/"
        if "Qwen" in input_path and "7B" in input_path:
            output_path += "Qwen7B"
        elif "Qwen" in input_path and "14B" in input_path:
            output_path += "Qwen14B"
        elif "Qwen" in input_path and "32B" in input_path:
            output_path += "Qwen32B"
        elif "Llama" in input_path and "8B" in input_path:
            output_path += "Llama8B"
        elif "Mistral" in input_path and "7B" in input_path:
            output_path += "Mistral7B"

        plot(input_path, output_path)
    
    