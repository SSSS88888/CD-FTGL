import os
import re
import numpy as np
import matplotlib.pyplot as plt
import glob

# 设置 matplotlib 参数
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 40  # 设置全局字体大小
Linewidth = 4.5
LabelFontdict = 30  # 图例字体大小
AxisFontdict = 30   # 坐标轴标签字体大小

# 基础参数
DATASET_NAME = 'a9a'  # 可选择: 'HIGGS', 'a9a', 'ijcnn1', 'diabetes'
COMPRESSOR_TYPE = 'top' # 压缩器类型 [rand, quant, gossip, top]
TOPOLOGY = 'random' # 拓扑结构
FEEDBACK_TYPE = 'one-bandit'  # 反馈类型 [full, one-bandit']
L_CHOICES = [100]  # 要筛选的L值列表，例如[100]表示只筛选L=100，[10, 100]表示筛选L=10和L=100

# 构建基础路径
feedback_dir = 'bandit' if FEEDBACK_TYPE != 'full' else 'full'
topology_dir = TOPOLOGY if TOPOLOGY != 'full' else 'fully_connected'
base_path = f"result/grid_search_repeated_n9/{DATASET_NAME}/{COMPRESSOR_TYPE}/{topology_dir}/{feedback_dir}"
BASE_SAVE_DIR = f"result/plots_Lp_comparison"

# 保存目录在绘图时创建（见 plot_best_results）

# 读取SR文件并计算最终值
def get_final_sr(file_path):
    try:
        sr_data = np.load(file_path)
        # 返回最后一个值，即最终的标准化遗憾
        return sr_data[-1] if len(sr_data) > 0 else float('inf')
    except Exception as e:
        print(f"读取文件 {file_path} 失败: {e}")
        return float('inf')

# 从文件名中提取数值参数
def extract_numeric_param(base_name, param_prefix):
    """
    从文件名中提取数值类型参数，支持下划线格式、小数点格式和科学计数法格式
    例如：w0_05、h1000、eta10、eps0_001、L1、L100等
    """
    # 尝试科学计数法格式（如h1e-5）
    sci_match = re.search(r'%s(\d+e[-+]\d+)(?:_|$|\.|[a-zA-Z])' % param_prefix, base_name)
    if sci_match:
        try:
            return float(sci_match.group(1))
        except ValueError:
            pass
    
    # 尝试下划线格式（如w0_05、eps1_0）
    underscore_match = re.search(r'%s(\d+)_(\d+)(?:_|$|\.|[a-zA-Z])' % param_prefix, base_name)
    if underscore_match:
        try:
            return float(f"{underscore_match.group(1)}.{underscore_match.group(2)}")
        except ValueError:
            pass
    
    # 尝试小数点格式或整数格式（如w0.05、h1000、L1、L100）
    decimal_match = re.search(r'%s(\d+(?:\.\d+)?)(?:_|$|\.|[a-zA-Z])' % param_prefix, base_name)
    if decimal_match:
        try:
            return float(decimal_match.group(1))
        except ValueError:
            pass
    
    return None

# 从文件名中提取seed值
def extract_seed(file_name):
    """从文件名中提取seed值"""
    seed_match = re.search(r'seed(\d+)', file_name)
    if seed_match:
        return int(seed_match.group(1))
    return None

# 计算多轮实验数据的平均值、标准差、最大值和最小值
def calculate_mean_and_std(data_list):
    """计算多轮实验数据的平均值、标准差、最大值和最小值"""
    if not data_list:
        return None, None, None, None
    
    # 如果只有一个数据点，可能是已经计算好的平均值数据
    if len(data_list) == 1:
        # 返回原始数据作为平均值，并创建一个全零的标准差数组
        mean_data = data_list[0]
        std_data = np.zeros_like(mean_data)
        max_data = data_list[0]
        min_data = data_list[0]
        return mean_data, std_data, max_data, min_data
    
    # 对于多个数据点，进行常规的平均值和标准差计算
    # 找出最短的数据长度
    min_length = min(len(data) for data in data_list)
    
    # 截断所有数据到相同长度
    truncated_data = [data[:min_length] for data in data_list]
    
    # 转换为numpy数组进行计算
    data_array = np.array(truncated_data)
    
    # 计算平均值、标准差、最大值和最小值
    mean_data = np.mean(data_array, axis=0)
    std_data = np.std(data_array, axis=0)
    max_data = np.max(data_array, axis=0)
    min_data = np.min(data_array, axis=0)
    
    return mean_data, std_data, max_data, min_data

# 找出每个压缩率下的最优参数设置
def find_best_params_for_compression_rates():
    # 存储结果
    best_results = {}
    
    # 遍历所有L'文件夹
    l_prime_dirs = glob.glob(os.path.join(base_path, 'L=*'))
    
    for l_dir in l_prime_dirs:
        # 提取L和L'值
        l_dir_name = os.path.basename(l_dir)
        # 解析L和L'值
        l_value = None
        l_prime_value = None
        
        try:
            # 格式: L=100_L'=50_0
            parts = l_dir_name.split('_')
            for part in parts:
                if part.startswith('L='):
                    l_value = float(part[2:].replace('_', '.'))
                elif part.startswith("L'="):
                    l_prime_value = float(part[3:].replace('_', '.'))
        except Exception as e:
            print(f"解析目录名 {l_dir_name} 失败: {e}")
            continue
        
        if l_value is None or l_prime_value is None:
            continue
        
        # 检查L值是否在L_CHOICES列表中
        if int(l_value) not in L_CHOICES:
            continue
        
        # 遍历所有压缩率 (w值)
        compression_rates = {}
        
        # 查找所有SR文件
        sr_files = glob.glob(os.path.join(l_dir, '*_SR.npy'))
        
        # 按参数分组文件
        param_groups = {}
        
        for sr_file in sr_files:
            # 提取文件名中的参数
            file_name = os.path.basename(sr_file)
            
            # 解析w值 (压缩率)
            w_value = extract_numeric_param(file_name, 'w')
            if w_value is None:
                continue
            
            # 提取seed值
            seed_value = extract_seed(file_name)
            if seed_value is None:
                continue
            
            # 提取其他参数（h, eps等）
            h_value = extract_numeric_param(file_name, 'h')
            eps_value = extract_numeric_param(file_name, 'eps')
            
            # 创建参数组合键
            param_key = (w_value, h_value, eps_value)
            
            # 将文件添加到对应的参数组
            if param_key not in param_groups:
                param_groups[param_key] = {'sr_files': [], 'bits_files': []}
            
            param_groups[param_key]['sr_files'].append(sr_file)
            
            # 生成对应的Bits文件名
            bits_file = sr_file.replace('_SR.npy', '_Bits.npy')
            param_groups[param_key]['bits_files'].append(bits_file)
        
        # 对每个参数组计算平均最终SR值
        for param_key, files_dict in param_groups.items():
            w_value, h_value, eps_value = param_key
            
            # 计算所有seed的平均最终SR值
            final_sr_values = []
            for sr_file in files_dict['sr_files']:
                final_sr = get_final_sr(sr_file)
                if final_sr != float('inf'):
                    final_sr_values.append(final_sr)
            
            if not final_sr_values:
                continue
            
            avg_final_sr = np.mean(final_sr_values)
            
            # 存储每个w值的最佳结果
            if w_value not in compression_rates or avg_final_sr < compression_rates[w_value]['sr']:
                compression_rates[w_value] = {
                    'sr': avg_final_sr,
                    'sr_files': files_dict['sr_files'],
                    'bits_files': files_dict['bits_files'],
                    'h_value': h_value,
                    'eps_value': eps_value
                }
        
        # 存储该L和L'下的压缩率结果
        if l_value not in best_results:
            best_results[l_value] = {}
        
        best_results[l_value][l_prime_value] = compression_rates
    
    return best_results

# 加载并处理数据
def load_and_process_data(file_paths):
    """加载并处理数据，计算平均值、标准差、最大值和最小值"""
    data_list = []
    
    for file_path in file_paths:
        try:
            data = np.load(file_path)
            data_list.append(data)
        except Exception as e:
            print(f"加载文件 {file_path} 失败: {e}")
            continue
    
    # 计算平均值、标准差、最大值和最小值
    mean_data, std_data, max_data, min_data = calculate_mean_and_std(data_list)
    
    return {
        'mean': mean_data,
        'std': std_data,
        'max': max_data,
        'min': min_data
    }

# 绘制不同L和L'的最优结果对比图
def plot_best_results(best_results):
    # 创建保存目录
    save_dir = os.path.join(BASE_SAVE_DIR, DATASET_NAME, COMPRESSOR_TYPE, TOPOLOGY, feedback_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # 收集所有压缩率
    all_w_values = set()
    for l_value, l_prime_results in best_results.items():
        for l_prime_value, compression_rates in l_prime_results.items():
            all_w_values.update(compression_rates.keys())
    
    for w_value in sorted(all_w_values):
        # 绘制不同L和L'的SR曲线对比图（时间序列）
        plt.figure(figsize=(12, 10))
        
        # 收集所有L和L'的SR数据
        sr_data_dict = {}
        max_steps = 0
        
        for l_value, l_prime_results in best_results.items():
            for l_prime_value, compression_rates in l_prime_results.items():
                if w_value in compression_rates:
                    sr_files = compression_rates[w_value]['sr_files']
                    sr_data = load_and_process_data(sr_files)
                    if sr_data['mean'] is not None:
                        label = f'L={int(l_value)}, K={int(l_prime_value)}'
                        sr_data_dict[label] = sr_data
                        if len(sr_data['mean']) > max_steps:
                            max_steps = len(sr_data['mean'])
        
        # 绘制所有SR曲线
        # 按照K值的大小对数据进行排序
        sorted_labels = sorted(sr_data_dict.keys(), key=lambda x: int(x.split('K=')[1]))
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_labels)))
        steps = np.arange(max_steps)
        
        for i, label in enumerate(sorted_labels):
            data = sr_data_dict[label]
            mean_data = data['mean']
            max_data = data['max']
            min_data = data['min']
            
            plt.plot(steps[:len(mean_data)], mean_data, linewidth=Linewidth, color=colors[i], label=label)
            
            # 使用最大值和最小值作为边界绘制阴影
            if max_data is not None and min_data is not None and len(max_data) == len(mean_data) and len(min_data) == len(mean_data):
                plt.fill_between(steps[:len(mean_data)], min_data, max_data, color=colors[i], alpha=0.2)
        
        # 设置图表属性
        plt.xlabel('Communication Rounds', fontsize=AxisFontdict)
        plt.ylabel('Average Loss', fontsize=AxisFontdict)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=LabelFontdict, loc='upper right')
        # 增大刻度标签字体大小
        plt.xticks(fontsize=AxisFontdict)
        plt.yticks(fontsize=AxisFontdict)
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(save_dir, f'Lp_comparison_w{w_value}_sr_curves_{DATASET_NAME}_{COMPRESSOR_TYPE}_{TOPOLOGY}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
        
        # 保存PDF格式
        save_path_pdf = os.path.join(save_dir, f'Lp_comparison_w{w_value}_sr_curves_{DATASET_NAME}_{COMPRESSOR_TYPE}_{TOPOLOGY}.pdf')
        plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path_pdf}")
        
        plt.close()
        
        # 绘制不同L和L'的Bits vs SR曲线对比图
        plt.figure(figsize=(12, 10))
        
        # 收集所有L和L'的Bits和SR数据
        bits_sr_data_dict = {}
        
        for l_value, l_prime_results in best_results.items():
            for l_prime_value, compression_rates in l_prime_results.items():
                if w_value in compression_rates:
                    sr_files = compression_rates[w_value]['sr_files']
                    bits_files = compression_rates[w_value]['bits_files']
                    
                    sr_data = load_and_process_data(sr_files)
                    bits_data = load_and_process_data(bits_files)
                    
                    if sr_data['mean'] is not None and bits_data['mean'] is not None:
                        label = f'L={int(l_value)}, K={int(l_prime_value)}'
                        # 确保数据长度一致
                        min_len = min(len(sr_data['mean']), len(bits_data['mean']))
                        if min_len > 0:
                            bits_sr_data_dict[label] = {
                                'sr': sr_data['mean'][:min_len],
                                'sr_max': sr_data['max'][:min_len],
                                'sr_min': sr_data['min'][:min_len],
                                'bits': bits_data['mean'][:min_len]
                            }
        
        # 绘制所有Bits vs SR曲线
        # 按照K值的大小对数据进行排序
        sorted_labels = sorted(bits_sr_data_dict.keys(), key=lambda x: int(x.split('K=')[1]))
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_labels)))
        
        for i, label in enumerate(sorted_labels):
            data = bits_sr_data_dict[label]
            # 对bits数据进行排序，确保x轴单调递增
            sorted_indices = np.argsort(data['bits'])
            sorted_bits = data['bits'][sorted_indices]
            sorted_sr = data['sr'][sorted_indices]
            sorted_sr_max = data['sr_max'][sorted_indices]
            sorted_sr_min = data['sr_min'][sorted_indices]
            
            plt.plot(sorted_bits, sorted_sr, linewidth=Linewidth, color=colors[i], label=label)
            
            # 使用最大值和最小值作为边界绘制阴影
            if sorted_sr_max is not None and sorted_sr_min is not None:
                plt.fill_between(sorted_bits, sorted_sr_min, sorted_sr_max, color=colors[i], alpha=0.2)
        
        # 设置图表属性
        plt.xlabel('Total Communication Bits (log scale)', fontsize=AxisFontdict)
        plt.ylabel('Average Loss', fontsize=AxisFontdict)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=LabelFontdict, loc='lower left')
        # 增大刻度标签字体大小
        plt.xticks(fontsize=AxisFontdict)
        plt.yticks(fontsize=AxisFontdict)
        plt.xscale('log')  # 设置log刻度
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(save_dir, f'Lp_comparison_w{w_value}_bits_vs_sr_{DATASET_NAME}_{COMPRESSOR_TYPE}_{TOPOLOGY}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
        
        # 保存PDF格式
        save_path_pdf = os.path.join(save_dir, f'Lp_comparison_w{w_value}_bits_vs_sr_{DATASET_NAME}_{COMPRESSOR_TYPE}_{TOPOLOGY}.pdf')
        plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path_pdf}")
        
        plt.close()

def main():
    print("正在查找最优参数设置...")
    best_results = find_best_params_for_compression_rates()
    
    print("\n最优参数设置:")
    for l_value, l_prime_results in best_results.items():
        print(f"\nL={int(l_value)}:")
        for l_prime_value, compression_rates in l_prime_results.items():
            print(f"  K={int(l_prime_value)}:")
            for w_value, result in compression_rates.items():
                print(f"    压缩率 w={w_value}: 平均标准化遗憾 = {result['sr']:.6f} (基于 {len(result['sr_files'])} 个随机种子)")
    
    print("\n正在绘制对比图...")
    plot_best_results(best_results)

if __name__ == "__main__":
    main()
