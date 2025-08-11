'''
This file visualize the results under /plot
'''

import json
import numpy as np
import matplotlib.pyplot as plt

def load(path):
    with open(path, 'r') as f:
        return list(json.load(f))

def extract_choices(choices):
    series = [[], [], []]
    for item in choices:
        series[0].append(item['choices'][0])
        series[1].append(item['choices'][1])
        series[2].append(item['choices'][2])
    return series


def count_KU(choices):
    count = np.zeros( (len(choices[0]), 2) )
    for item in choices:
        for i, choice in enumerate(item):
            if choice == 'K':
                count[i][0] += 1
            elif choice == 'U':
                count[i][1] += 1
    return np.array(count)

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

def plot(input_path, output_path):
    f1 = load(input_path)
    choices1 = extract_choices(f1)
    count = [count_KU(choice) for choice in choices1]
    count1, count2, count3 = count
    print(count1)
    print(count2)
    print(count3)
    plot_first_number_ratio(count1, save_path=output_path+"/count1")
    plot_first_number_ratio(count2, save_path=output_path+"/count2")
    plot_first_number_ratio(count3, save_path=output_path+"/count3")

    # count = [x.tolist() if isinstance(x, np.ndarray) else x for x in count]
    # with open('/project/rwangcn/MUvBE/MR_small/data.py', 'w') as f:
    #     f.write(f"data = {count}")

if __name__ == "__main__":
    for input_path in [
        "/project/rwangcn/MUvBE/marker_risk/result/meta-llama_Llama-3.1-8B-Instruct.json",
        "/project/rwangcn/MUvBE/marker_risk/result/mistralai_Mistral-7B-Instruct-v0.3.json",
        "/project/rwangcn/MUvBE/marker_risk/result/Qwen_Qwen2.5-7B-Instruct.json",
        "/project/rwangcn/MUvBE/marker_risk/result/Qwen_Qwen2.5-14B-Instruct.json",
        "/project/rwangcn/MUvBE/marker_risk/result/Qwen_Qwen2.5-32B-Instruct.json"
    ]:
        output_path = "/project/rwangcn/MUvBE/marker_risk/plot/"
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
    