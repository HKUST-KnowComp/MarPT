import json
import matplotlib.pyplot as plt
import numpy as np

def load_data(input_path):
    with open(input_path, 'r') as f:
        result = json.load(f)
        if "Qwen" in input_path and "7B" in input_path:
            model_name = "Qwen7B"
        elif "Qwen" in input_path and "14B" in input_path:
            model_name = "Qwen14B"
        elif "Qwen" in input_path and "32B" in input_path:
            model_name = "Qwen32B"
        elif "Llama" in input_path and "8B" in input_path:
            model_name = "Llama8B"
        elif "Mistral" in input_path and "7B" in input_path:
            model_name = "Mistral7B"

        for key in result.keys():
            result[key] = result[key]['Switching']

        result = {
            'model': model_name,
            'mapping': result
        }

        return result

def load_all(input_paths):
    mapping_data = []
    for input_path in input_paths:
        mapping_data.append(load_data(input_path))
    return mapping_data

def plot_epistemic_markers(data, output_path):
    # 提取模型名称
    models = [entry['model'] for entry in data]
    n_models = len(models)
    
    # 提取所有标记（使用第一个模型的键顺序）
    all_markers = list(data[0]['mapping'].keys())
    n_markers = len(all_markers)
    
    # 分割标记为两组（每组7个）
    markers_top = all_markers[:7]
    markers_bottom = all_markers[7:14]
    
    # 颜色设置
    colors = ['#4e79a7', '#f28e2c', '#59a14f', '#e15759', '#b07aa1']
    
    # 创建图形和子图
    plt.figure(figsize=(18, 10))
    
    # 转换百分比字符串为浮点数
    def parse_percentage(percent_str):
        return round(float(percent_str.replace('%', '')) / 100, 2)

    # 第一组标记（顶部）
    ax1 = plt.subplot(2, 1, 1)
    width = 0.15  # 条状图的宽度
    indices = np.arange(len(markers_top))
    
    for i, model in enumerate(models):
        # 获取当前模型在顶部标记的概率值
        values = [parse_percentage(data[i]['mapping'][marker]) for marker in markers_top]
        # 计算每个条的位置
        positions = indices + i * width - width * (n_models - 1) / 2
        ax1.bar(positions, values, width, label=model, color=colors[i])

        for idx, rect in enumerate(ax1.patches):
            if idx // len(markers_top) == i:  # 只处理当前模型的条形
                height = rect.get_height()
                ax1.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 0.5),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=7, rotation=0)
    
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.set_xticks(indices)
    ax1.set_xticklabels(markers_top, rotation=15, fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.set_title('Epistemic Markers', fontsize=14, pad=15)
    
    # 第二组标记（底部）
    ax2 = plt.subplot(2, 1, 2)
    indices = np.arange(len(markers_bottom))
    
    for i, model in enumerate(models):
        # 获取当前模型在底部标记的概率值
        values = [parse_percentage(data[i]['mapping'][marker]) for marker in markers_bottom]
        positions = indices + i * width - width * (n_models - 1) / 2
        ax2.bar(positions, values, width, label=model, color=colors[i])

        for idx, rect in enumerate(ax2.patches):
            if idx // len(markers_bottom) == i:  # 只处理当前模型的条形
                height = rect.get_height()
                ax2.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 0.5),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=7, rotation=0)
    
    ax2.set_ylabel('Probability', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.set_xticks(indices)
    ax2.set_xticklabels(markers_bottom, rotation=15, fontsize=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    # ax2.set_title('Bottom 7 Epistemic Markers', fontsize=14, pad=15)
    
    # 添加图例
    plt.figlegend(models, loc='lower center', ncol=n_models, 
                 bbox_to_anchor=(0.5, 0), fontsize=10)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(bottom=0.1)
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    

if __name__ == "__main__":
    input_paths = [
        "/project/rwangcn/MUvBE/marker/processed/meta-llama_Llama-3.1-8B-Instruct.json",
        "/project/rwangcn/MUvBE/marker/processed/mistralai_Mistral-7B-Instruct-v0.3.json",
        "/project/rwangcn/MUvBE/marker/processed/Qwen_Qwen2.5-7B-Instruct.json",
        "/project/rwangcn/MUvBE/marker/processed/Qwen_Qwen2.5-14B-Instruct.json",
        "/project/rwangcn/MUvBE/marker/processed/Qwen_Qwen2.5-32B-Instruct.json"
    ]

    data = load_all(input_paths)
    plot_epistemic_markers(data, "epistemic_markers_plot.png")