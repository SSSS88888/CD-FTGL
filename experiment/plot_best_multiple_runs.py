import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 40  # 设置全局字体大小
Linewidth = 4.5
LabelFontdict = 30  # 图例字体大小
AxisFontdict = 30   # 坐标轴标签字体大小

# 算法颜色映射 - 固定每个算法的颜色
ALGO_COLOR_MAP = {
    'CDFTGL': '#1f77b4',      # 蓝色
    'CDFTGL(L=1)': '#d62728',    # 红色
    'TopDOGD': '#2ca02c',      # 绿色
    'DCDOGD': '#ff7f0e'        # 橙色
}

# 算法线型映射 - 固定每个算法的线型
ALGO_LINESTYLE_MAP = {
    'CDFTGL': '-',             # 实线
    'CDFTGL(L=1)': '-',          # 实线
    'TopDOGD': '--',            # 虚线
    'DCDOGD': '--'             # 虚线
}

# 算法名称映射 - 根据反馈类型区分
ALGO_NAME_MAP = {
    'full': {
        'CDFTGL': 'CD-FTGL',
        'CDFTGL(L=1)': 'CD-FTGL(L=1)',
        'TopDOGD': 'Top-DOGD',
        'DCDOGD': 'DC-DOGD'
    },
    'bandit': {
        'CDFTGL': 'CD-FTBL',
        'CDFTGL(L=1)': 'CD-FTBL(L=1)',
        'TopDOGD': 'Top-DOBD-1',
        'DCDOGD': 'DC-DOBD'
    }
}

# 用户指定的参数
# 要加载的数据集名称，None表示所有
DATASET_NAME_TO_LOAD = ['a9a']  # 例如：['HIGGS', 'a9a', 'ijcnn1']
MAX_ROUNDS = 10000
# 要加载的压缩器类型列表，None表示所有
COMPRESSOR_TYPES_TO_LOAD = ['top']  # 例如：['gossip', 'top', 'rand', 'quant']

# 要加载的拓扑结构类型列表，None表示所有
TOPOLOGY_TYPES_TO_LOAD = ['random']  # 例如：['random', 'fully_connected', 'ring']

# 要加载的实验类型列表
FEEDBACK_TYPES_TO_LOAD = ['full', 'bandit']  # 例如：['full', 'bandit']

# 要加载的T值列表，None表示所有
T_VALUES_TO_LOAD = [10000]  # 例如：[1000, 5000, 10000]

# 压缩率列表
COMPRESSION_RATES = [0.05, 0.1, 0.5, 1]

# 算法列表
ALGORITHMS = ['CDFTGL', 'CDFTGL(L=1)', 'TopDOGD', 'DCDOGD']

# 基础目录设置
# 使用绝对路径，确保脚本能够找到参数文件
script_dir = os.path.dirname(os.path.abspath(__file__))
BASE_PARAMS_DIR = os.path.join(script_dir, 'result/repeated_best_algorithms_comparison_n9/best_parameters_tables')
BASE_RESULT_DIR = os.path.join(script_dir, 'result/grid_search_repeated_n9')
BASE_SAVE_DIR = os.path.join(script_dir, 'result/plots_best_multiple_runs_n9')

# 保存目录在绘图时创建（见 plot_and_save / plot_combined_feedback_charts）

# 调试开关
DEBUG = False

# 文件缓存，避免重复加载相同的文件
_file_cache = {}

# 存储结果数据，用于生成表格
results_data = []



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


def get_algorithm_label(algorithm, feedback_type, params):
    """根据算法、反馈类型和参数生成图例标签"""
    return ALGO_NAME_MAP.get(feedback_type, {}).get(algorithm, algorithm)


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


def find_best_param_files(dataset_name, compressor_types=None, topology_types=None):
    """根据参数查找对应的最佳参数文件
    
    Args:
        dataset_name: 数据集名称
        compressor_types: 压缩器类型列表，None表示所有
        topology_types: 拓扑结构类型列表，None表示所有
        
    Returns:
        匹配的参数文件列表
    """
    param_files = []
    
    # 遍历所有参数文件
    for file in os.listdir(BASE_PARAMS_DIR):
        if file.endswith('.csv'):
            # 检查文件名是否包含当前数据集名称
            if dataset_name not in file:
                continue
            
            # 如果指定了压缩器类型，检查文件名是否包含其中任意一个
            if compressor_types:
                if not any(f"comp{comp}" in file for comp in compressor_types):
                    continue
            
            # 如果指定了拓扑结构类型，检查文件名是否包含其中任意一个
            if topology_types:
                if not any(f"topo{topo}" in file for topo in topology_types):
                    continue
            
            param_files.append(os.path.join(BASE_PARAMS_DIR, file))
    
    # 去重，确保每个参数文件只被处理一次
    param_files = list(set(param_files))
    
    return param_files


def load_best_params(file_path):
    """加载最佳参数表格"""
    try:
        df = pd.read_csv(file_path)
        # 清理列名，去除所有空格
        df.columns = df.columns.str.strip()
        # 清理所有字符串列的值，去除空格
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()

        return df
    except Exception as e:
        print(f"Error loading best params file {file_path}: {e}")
        return None


def get_corresponding_files(dataset, compressor, topology, feedback, algorithm, params):
    """
    根据参数精确匹配对应的文件
    """
    # 构建搜索目录
    search_dir = os.path.join(BASE_RESULT_DIR, dataset, compressor, topology, feedback)
    
    if not os.path.exists(search_dir):
        print(f"Directory not found: {search_dir}")
        return []
    
    # 提取参数
    t = int(params['T'])
    w = float(params['w'])
    
    # 处理CDFTGL(L=1)特殊情况：实际是CDFTGL算法，L=1
    actual_algorithm = algorithm
    if algorithm == 'CDFTGL(L=1)':
        actual_algorithm = 'CDFTGL'  # 实际搜索CDFTGL文件
    
    # 根据算法类型提取不同的参数
    if algorithm == 'CDFTGL' or algorithm == 'CDFTGL(L=1)':
        # CDFTGL相关算法使用h_coeff参数
        h_coeff = float(params['h_coeff'])
        if algorithm == 'CDFTGL(L=1)':
            l_value = 1  # CDFTGL(L=1)固定L=1，转换为整数
        else:
            l_value = int(float(params['L']))  # 将浮点数转换为整数
        eps_coeff = params.get('eps_coeff', None)
        eps_coeff = float(eps_coeff) if pd.notna(eps_coeff) and eps_coeff != '' else None
    elif algorithm == 'TopDOGD':
        # TopDOGD算法使用eta_coeff参数，需要L参数
        eta_coeff = float(params['eta_coeff'])
        l_value = int(float(params['L'])) if 'L' in params and pd.notna(params['L']) else None  # 将浮点数转换为整数
        eps_coeff = params.get('eps_coeff', None)
        eps_coeff = float(eps_coeff) if pd.notna(eps_coeff) and eps_coeff != '' else None
    else:  # DCDOGD
        # DCDOGD算法使用eta_coeff参数，不需要L参数
        eta_coeff = float(params['eta_coeff'])
        l_value = None  # DCDOGD算法不需要L参数
        eps_coeff = params.get('eps_coeff', None)
        eps_coeff = float(eps_coeff) if pd.notna(eps_coeff) and eps_coeff != '' else None
    
    # 构建文件名模式
    w_str = f"{w}".replace('.', '_')
    if w_str.endswith('_0'):
        w_str = w_str[:-2]
    
    # 根据算法类型构建精确的文件模式
    if algorithm == 'CDFTGL' or algorithm == 'CDFTGL(L=1)' or algorithm == 'TopDOGD':
        # 需要L参数的算法，确保精确匹配L值
        file_pattern = f"*{actual_algorithm}*_T{t}_w{w_str}*_L{l_value}_*"  # 使用整数形式的L值
    else:  # DCDOGD不需要L参数
        file_pattern = f"*{actual_algorithm}*_T{t}_w{w_str}*"
    
    # 搜索匹配的文件 - 使用file_pattern过滤
    file_pattern_path = os.path.join(search_dir, file_pattern)
    
    all_files = glob(file_pattern_path)
    
    matched_files = []
    
    print(f"Initial files matching pattern: {len(all_files)} in {search_dir}")
    print(f"Search pattern: {file_pattern}")
    
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        
        # 检查是否包含seed信息
        if 'seed' not in file_name:
            continue
        
        # 精确匹配参数
        # 匹配T参数
        t_match = re.search(r'T(\d+)', file_name)
        if not t_match or int(t_match.group(1)) != t:
            continue
        
        # 匹配w参数
        w_val = extract_numeric_param(file_name, 'w')
        if w_val is None or abs(w_val - w) > 1e-6:
            continue
        
        # 严格匹配所有算法的参数
        matched = True
        
        if algorithm == 'CDFTGL' or algorithm == 'CDFTGL(L=1)':
            # CDFTGL相关算法参数匹配
            h_val = extract_numeric_param(file_name, 'h')
            l_val = extract_numeric_param(file_name, 'L')
            
            if h_val is None:
                # print(f"    Skipping (h_val not found in {file_name})")
                matched = False
            elif abs(h_val - h_coeff) > 1e-6:
                # print(f"    Skipping (h mismatch: found {h_val}, expected {h_coeff})")
                matched = False
            
            if l_val is None:
                # print(f"    Skipping (l_val not found in {file_name})")
                matched = False
            elif abs(l_val - l_value) > 1e-6:
                # print(f"    Skipping (L mismatch: found {l_val}, expected {l_value})")
                matched = False
        elif algorithm == 'TopDOGD':
            # TopDOGD算法参数匹配，需要匹配eta和L值
            # 匹配eta参数
            eta_val = extract_numeric_param(file_name, 'eta')
            if eta_val is None:
                # print(f"    Skipping (eta_val not found in {file_name})")
                matched = False
            elif abs(eta_val - eta_coeff) > 1e-6:
                # print(f"    Skipping (eta mismatch: found {eta_val}, expected {eta_coeff})")
                matched = False
            
            # 匹配L参数
            if l_value is not None:
                l_val = extract_numeric_param(file_name, 'L')
                if l_val is None:
                    # print(f"    Skipping (l_val not found in {file_name})")
                    matched = False
                elif abs(l_val - l_value) > 1e-6:
                    # print(f"    Skipping (L mismatch: found {l_val}, expected {l_value})")
                    matched = False
        else:  # DCDOGD
            # DCDOGD算法参数匹配，只需要匹配eta值，不需要L值
            eta_val = extract_numeric_param(file_name, 'eta')
            if eta_val is None:
                # print(f"    Skipping (eta_val not found in {file_name})")
                matched = False
            elif abs(eta_val - eta_coeff) > 1e-6:
                # print(f"    Skipping (eta mismatch: found {eta_val}, expected {eta_coeff})")
                matched = False
        
        # 匹配eps参数（如果存在）
        if eps_coeff is not None and matched:
            eps_val = extract_numeric_param(file_name, 'eps')
            if eps_val is None:
                # print(f"    Skipping (eps_val not found in {file_name})")
                matched = False
            elif abs(eps_val - eps_coeff) > 1e-6:
                # print(f"    Skipping (eps mismatch: found {eps_val}, expected {eps_coeff})")
                matched = False
        
        if matched:
            if DEBUG:
                print(f"    Adding file: {file_name}")
            matched_files.append(file_path)
    
    print(f"Matched {len(matched_files)} files for {algorithm} with params {params}")
    
    return matched_files


def load_and_process_data(files):
    """加载并处理数据，计算平均值、标准差、最大值和最小值"""
    sr_data = []
    bits_data = []
    global _file_cache
    
    for file_path in files:
        try:
            # 使用缓存，避免重复加载相同的文件
            if file_path in _file_cache:
                data = _file_cache[file_path]
            else:
                data = np.load(file_path)
                _file_cache[file_path] = data
            
            # 确定数据类型
            if 'SR' in file_path.upper():
                sr_data.append(data)
            elif 'BITS' in file_path.upper():
                bits_data.append(data)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            continue
    
    # 计算平均值、标准差、最大值和最小值
    sr_mean, sr_std, sr_max, sr_min = calculate_mean_and_std(sr_data)
    bits_mean, bits_std, bits_max, bits_min = calculate_mean_and_std(bits_data)
    
    return {
        'SR': {
            'mean': sr_mean,
            'std': sr_std,
            'max': sr_max,
            'min': sr_min
        },
        'Bits': {
            'mean': bits_mean,
            'std': bits_std,
            'max': bits_max,
            'min': bits_min
        }
    }


def plot_and_save(dataset, compressor, topology, feedback, compression_rate, all_algorithms_data, params):
    """
    将所有算法绘制在同一张图上并保存
    """
    # 创建保存目录
    save_dir = os.path.join(BASE_SAVE_DIR, dataset, compressor, topology, feedback)
    os.makedirs(save_dir, exist_ok=True)
    
    # 绘制SR图表
    plt.figure(figsize=(12, 10))
    
    # 使用指定的颜色映射
    algorithm_color_map = ALGO_COLOR_MAP
    
    # 遍历所有算法数据
    for algorithm, data in all_algorithms_data.items():
        if data['SR']['mean'] is None or len(data['SR']['mean']) == 0:
            continue
        
        # 获取算法显示名称
        algorithm_label = get_algorithm_label(algorithm, feedback, params)
        
        # 设置图表样式
        color = algorithm_color_map.get(algorithm, '#7f7f7f')
        linestyle = ALGO_LINESTYLE_MAP.get(algorithm, '-')
        
        # 根据算法调整通信轮次
        time = np.arange(1, len(data['SR']['mean']) + 1)
        # if algorithm ==  'TopDOGD':
        #     time = time * 2
        sr_mean = data['SR']['mean']
        sr_max = data['SR']['max']
        sr_min = data['SR']['min']
        
        # 确保横坐标截止
        max_rounds = MAX_ROUNDS
        if time.size > 0 and time[-1] > max_rounds:
            cutoff_idx = np.searchsorted(time, max_rounds, side='right')
            time = time[:cutoff_idx]
            sr_mean = sr_mean[:cutoff_idx]
            sr_max = sr_max[:cutoff_idx]
            sr_min = sr_min[:cutoff_idx]
        
        # 确保数据长度一致
        if len(time) != len(sr_mean):
            min_len = min(len(time), len(sr_mean), len(sr_max), len(sr_min))
            time = time[:min_len]
            sr_mean = sr_mean[:min_len]
            sr_max = sr_max[:min_len]
            sr_min = sr_min[:min_len]
        
        # 绘制曲线
        if len(time) > 0 and len(sr_mean) > 0:
            plt.plot(time, sr_mean, 
                    label=algorithm_label,
                    color=color,
                    linestyle=linestyle,
                    linewidth=Linewidth)
            
            # 使用最大值和最小值作为边界绘制阴影
            if sr_max is not None and sr_min is not None and len(sr_max) == len(time) and len(sr_min) == len(time):
                plt.fill_between(time, 
                                sr_min, 
                                sr_max, 
                                color=color, 
                                alpha=0.2)
            
            # 收集w=0.1时每个算法的最后一个值
            if abs(compression_rate - 0.1) < 1e-6:
                final_value = sr_mean[-1]
                results_data.append({
                    'dataset': dataset,
                    'feedback': feedback,
                    'algorithm': algorithm_label,
                    'final_loss': final_value,
                    'comm_rounds': len(sr_mean)
                })
            

    
    # 设置图表属性
    # plt.title(f'Optimal Algorithms Comparison SR ({dataset}, T={params["T"]}, w={compression_rate}, {feedback}, topology={topology})')
    plt.xlabel('Communication Rounds', fontsize=AxisFontdict)
    plt.ylabel('Average Loss', fontsize=AxisFontdict)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=LabelFontdict, loc='upper right')
    # 增大刻度标签字体大小
    plt.xticks(fontsize=AxisFontdict)
    plt.yticks(fontsize=AxisFontdict)
    
    # 根据数据集设置横坐标上界
    # if dataset == 'HIGGS':
    #     plt.xlim(0, 400)
    # elif dataset in ['ijcnn1', 'a9a']:
    #     plt.xlim(0, 700)
    
    # 根据数据集设置纵坐标上界
    if dataset == 'HIGGS':
        plt.ylim(0.325, 0.5)  # 只设置上限，不设置下限
    elif dataset in ['ijcnn1', 'a9a']:
        plt.ylim(auto=True, top=0.8)  # 设置上限为0.8，下限自适应
    
    # 保存图表
    bandit_suffix = "_bandit" if feedback == 'bandit' else ""
    # 保存PNG格式
    save_path = os.path.join(save_dir, f"optimal_comparison_SR_T{params['T']}_w{compression_rate}_comp{compressor}_topo{topology}{bandit_suffix}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved SR plot to {save_path}")
    # 保存PDF格式
    save_path_pdf = os.path.join(save_dir, f"optimal_comparison_SR_T{params['T']}_w{compression_rate}_comp{compressor}_topo{topology}{bandit_suffix}.pdf")
    plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight')
    print(f"Saved SR plot to {save_path_pdf}")
    plt.close()
    
    # 绘制Bits vs SR图表
    plt.figure(figsize=(12, 10))
    
    # 使用指定的颜色映射
    algorithm_color_map = ALGO_COLOR_MAP
    
    # 遍历所有算法数据
    for algorithm, data in all_algorithms_data.items():
        if data['SR']['mean'] is None or data['Bits']['mean'] is None or len(data['SR']['mean']) == 0 or len(data['Bits']['mean']) == 0:
            continue
        
        sr_data = data['SR']['mean']
        sr_max_data = data['SR']['max']
        sr_min_data = data['SR']['min']
        bits_data = data['Bits']['mean']
        
        # 根据数据集设置截断长度
        # if dataset == 'HIGGS':
        #     max_rounds = 400
        # elif dataset in ['ijcnn1', 'a9a']:
        #     max_rounds = 700
        # else:
        #     max_rounds = MAX_ROUNDS
        max_rounds = MAX_ROUNDS
        # 首先截断SR数据到指定的max_rounds
        if len(sr_data) > max_rounds:
            sr_data = sr_data[:max_rounds]
            sr_max_data = sr_max_data[:max_rounds]
            sr_min_data = sr_min_data[:max_rounds]
        
        # 然后截断Bits数据，确保与截断后的SR数据长度一致
        min_len = min(len(sr_data), len(bits_data), len(sr_max_data), len(sr_min_data))
        if min_len == 0:
            continue
        
        sr_mean = sr_data[:min_len]
        sr_max = sr_max_data[:min_len]
        sr_min = sr_min_data[:min_len]
        bits_mean = bits_data[:min_len]
        
        # 获取算法显示名称
        algorithm_label = get_algorithm_label(algorithm, feedback, params)
        
        # 设置图表样式
        color = algorithm_color_map.get(algorithm, '#7f7f7f')
        linestyle = ALGO_LINESTYLE_MAP.get(algorithm, '-')
        
        # 绘制曲线
        if len(bits_mean) > 0 and len(sr_mean) > 0:
            # 对bits_mean进行排序，确保x轴单调递增
            sorted_indices = np.argsort(bits_mean)
            sorted_bits = bits_mean[sorted_indices]
            sorted_sr = sr_mean[sorted_indices]
            sorted_sr_max = sr_max[sorted_indices]
            sorted_sr_min = sr_min[sorted_indices]
            
            plt.plot(sorted_bits, sorted_sr, 
                    label=algorithm_label,
                    color=color,
                    linestyle=linestyle,
                    linewidth=Linewidth)
            
            # 使用最大值和最小值作为边界绘制阴影
            if sr_max is not None and sr_min is not None:
                plt.fill_between(sorted_bits, 
                                sorted_sr_min, 
                                sorted_sr_max, 
                                color=color, 
                                alpha=0.2)
    
    # 设置图表属性
    # plt.title(f'Optimal Algorithms Comparison: Bits vs SR ({dataset}, T={params["T"]}, w={compression_rate}, {feedback}, topology={topology})')
    plt.xlabel('Total Communication Bits (log scale)', fontsize=AxisFontdict)
    plt.ylabel('Average Loss', fontsize=AxisFontdict)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=LabelFontdict, loc='lower left')
    plt.xscale('log')
    # 增大刻度标签字体大小
    plt.xticks(fontsize=AxisFontdict)
    plt.yticks(fontsize=AxisFontdict)
    
    # 根据数据集设置纵坐标上界
    if dataset == 'HIGGS':
        plt.ylim(0.325, 0.5)  # 只设置上限，不设置下限
    elif dataset in ['ijcnn1', 'a9a']:
        plt.ylim(auto=True, top=0.8)  # 设置上限为0.8，下限自适应
    
    # 保存图表
    bandit_suffix = "_bandit" if feedback == 'bandit' else ""
    # 保存PNG格式
    save_path = os.path.join(save_dir, f"optimal_comparison_Bits_vs_SR_T{params['T']}_w{compression_rate}_comp{compressor}_topo{topology}{bandit_suffix}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved Bits vs SR plot to {save_path}")
    # 保存PDF格式
    save_path_pdf = os.path.join(save_dir, f"optimal_comparison_Bits_vs_SR_T{params['T']}_w{compression_rate}_comp{compressor}_topo{topology}{bandit_suffix}.pdf")
    plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight')
    print(f"Saved Bits vs SR plot to {save_path_pdf}")
    plt.close()


def plot_combined_feedback_charts(dataset, compressor, topology, compression_rate, all_feedback_data, params):
    """
    将所有算法的full和bandit数据绘制在同一张图上并保存
    """
    # 创建保存目录，不包含反馈类型在路径中
    save_dir = os.path.join(BASE_SAVE_DIR, dataset, compressor, topology)
    os.makedirs(save_dir, exist_ok=True)
    
    # 绘制SR图表
    plt.figure(figsize=(12, 10))
    
    # 使用指定的颜色映射
    algorithm_color_map = ALGO_COLOR_MAP
    
    # 遍历所有算法
    for algorithm in ALGORITHMS:
        # 获取算法的颜色
        color = algorithm_color_map.get(algorithm, '#7f7f7f')
        
        # 绘制full数据（实线）
        if 'full' in all_feedback_data and algorithm in all_feedback_data['full']:
            data = all_feedback_data['full'][algorithm]
            if data['SR']['mean'] is not None and len(data['SR']['mean']) > 0:
                # 根据算法调整通信轮次
                time = np.arange(1, len(data['SR']['mean']) + 1)
                sr_mean = data['SR']['mean']
                sr_max = data['SR']['max']
                sr_min = data['SR']['min']
                
                # 根据数据集设置截断长度
                # if dataset == 'HIGGS':
                #     max_rounds = 400
                # elif dataset in ['ijcnn1', 'a9a']:
                #     max_rounds = 700
                # else:
                #     max_rounds = MAX_ROUNDS
                max_rounds = MAX_ROUNDS
                
                # 确保横坐标截止
                if time.size > 0 and time[-1] > max_rounds:
                    cutoff_idx = np.searchsorted(time, max_rounds, side='right')
                    time = time[:cutoff_idx]
                    sr_mean = sr_mean[:cutoff_idx]
                    sr_max = sr_max[:cutoff_idx]
                    sr_min = sr_min[:cutoff_idx]
                
                # 确保数据长度一致
                if len(time) != len(sr_mean):
                    min_len = min(len(time), len(sr_mean), len(sr_max), len(sr_min))
                    time = time[:min_len]
                    sr_mean = sr_mean[:min_len]
                    sr_max = sr_max[:min_len]
                    sr_min = sr_min[:min_len]
                
                # 获取full反馈类型的算法显示名称
                algorithm_label_full = get_algorithm_label(algorithm, 'full', params)
                
                # 绘制曲线
                if len(time) > 0 and len(sr_mean) > 0:
                    plt.plot(time, sr_mean, 
                            label=algorithm_label_full,
                            color=color,
                            linestyle='-',
                            linewidth=Linewidth)
                    
                    # 使用最大值和最小值作为边界绘制阴影
                    if sr_max is not None and sr_min is not None and len(sr_max) == len(time) and len(sr_min) == len(time):
                        plt.fill_between(time, 
                                        sr_min, 
                                        sr_max, 
                                        color=color, 
                                        alpha=0.2)
        
        # 绘制bandit数据（虚线）
        if 'bandit' in all_feedback_data and algorithm in all_feedback_data['bandit']:
            data = all_feedback_data['bandit'][algorithm]
            if data['SR']['mean'] is not None and len(data['SR']['mean']) > 0:
                # 根据算法调整通信轮次
                time = np.arange(1, len(data['SR']['mean']) + 1)
                sr_mean = data['SR']['mean']
                sr_max = data['SR']['max']
                sr_min = data['SR']['min']
                
                # 根据数据集设置截断长度
                # if dataset == 'HIGGS':
                #     max_rounds = 400
                # elif dataset in ['ijcnn1', 'a9a']:
                #     max_rounds = 700
                # else:
                #     max_rounds = MAX_ROUNDS
                
                # 确保横坐标截止
                if time.size > 0 and time[-1] > max_rounds:
                    cutoff_idx = np.searchsorted(time, max_rounds, side='right')
                    time = time[:cutoff_idx]
                    sr_mean = sr_mean[:cutoff_idx]
                    sr_max = sr_max[:cutoff_idx]
                    sr_min = sr_min[:cutoff_idx]
                
                # 确保数据长度一致
                if len(time) != len(sr_mean):
                    min_len = min(len(time), len(sr_mean), len(sr_max), len(sr_min))
                    time = time[:min_len]
                    sr_mean = sr_mean[:min_len]
                    sr_max = sr_max[:min_len]
                    sr_min = sr_min[:min_len]
                
                # 获取bandit反馈类型的算法显示名称
                algorithm_label_bandit = get_algorithm_label(algorithm, 'bandit', params)
                
                # 绘制曲线
                if len(time) > 0 and len(sr_mean) > 0:
                    plt.plot(time, sr_mean, 
                            label=algorithm_label_bandit,
                            color=color,
                            linestyle='--',
                            linewidth=Linewidth)
                    
                    # 使用最大值和最小值作为边界绘制阴影
                    if sr_max is not None and sr_min is not None and len(sr_max) == len(time) and len(sr_min) == len(time):
                        plt.fill_between(time, 
                                        sr_min, 
                                        sr_max, 
                                        color=color, 
                                        alpha=0.2)
    
    # 设置图表属性
    # plt.title(f'Optimal Algorithms Comparison SR ({dataset}, T={params["T"]}, w={compression_rate}, topology={topology})')
    plt.xlabel('Communication Rounds', fontsize=AxisFontdict)
    plt.ylabel('Average Loss', fontsize=AxisFontdict)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=LabelFontdict, loc='upper right')
    # 增大刻度标签字体大小
    plt.xticks(fontsize=AxisFontdict)
    plt.yticks(fontsize=AxisFontdict)
    
    # 根据数据集设置横坐标上界
    # if dataset == 'HIGGS':
    #     plt.xlim(0, 400)
    # elif dataset in ['ijcnn1', 'a9a']:
    #     plt.xlim(0, 700)
    
    # 根据数据集设置纵坐标上界
    if dataset == 'HIGGS':
        plt.ylim(0.325, 0.5)  # 只设置上限，不设置下限
    elif dataset in ['ijcnn1', 'a9a']:
        plt.ylim(auto=True, top=0.8)  # 设置上限为0.8，下限自适应
    
    # 保存图表
    # 保存PNG格式
    save_path = os.path.join(save_dir, f"optimal_comparison_SR_T{params['T']}_w{compression_rate}_comp{compressor}_topo{topology}_combined.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined SR plot to {save_path}")
    # 保存PDF格式
    save_path_pdf = os.path.join(save_dir, f"optimal_comparison_SR_T{params['T']}_w{compression_rate}_comp{compressor}_topo{topology}_combined.pdf")
    plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight')
    print(f"Saved combined SR plot to {save_path_pdf}")
    plt.close()
    
    # 绘制Bits vs SR图表
    plt.figure(figsize=(12, 10))
    
    # 使用指定的颜色映射
    algorithm_color_map = ALGO_COLOR_MAP
    
    # 遍历所有算法
    for algorithm in ALGORITHMS:
        # 获取算法的颜色
        color = algorithm_color_map.get(algorithm, '#7f7f7f')
        
        # 绘制full数据（实线）
        if 'full' in all_feedback_data and algorithm in all_feedback_data['full']:
            data = all_feedback_data['full'][algorithm]
            if data['SR']['mean'] is not None and data['Bits']['mean'] is not None and len(data['SR']['mean']) > 0 and len(data['Bits']['mean']) > 0:
                sr_data = data['SR']['mean']
                sr_max_data = data['SR']['max']
                sr_min_data = data['SR']['min']
                bits_data = data['Bits']['mean']
                
                # 根据数据集设置截断长度
                # if dataset == 'HIGGS':
                #     max_rounds = 400
                # elif dataset in ['ijcnn1', 'a9a']:
                #     max_rounds = 700
                # else:
                #     max_rounds = MAX_ROUNDS
                max_rounds = MAX_ROUNDS
                
                # 首先截断SR数据到指定的max_rounds
                if len(sr_data) > max_rounds:
                    sr_data = sr_data[:max_rounds]
                    sr_max_data = sr_max_data[:max_rounds]
                    sr_min_data = sr_min_data[:max_rounds]
                
                # 然后截断Bits数据，确保与截断后的SR数据长度一致
                min_len = min(len(sr_data), len(bits_data), len(sr_max_data), len(sr_min_data))
                if min_len > 0:
                    sr_mean = sr_data[:min_len]
                    sr_max = sr_max_data[:min_len]
                    sr_min = sr_min_data[:min_len]
                    bits_mean = bits_data[:min_len]
                    
                    # 获取full反馈类型的算法显示名称
                    algorithm_label_full = get_algorithm_label(algorithm, 'full', params)
                    
                    # 绘制曲线
                    if len(bits_mean) > 0 and len(sr_mean) > 0:
                        # 对bits_mean进行排序，确保x轴单调递增
                        sorted_indices = np.argsort(bits_mean)
                        sorted_bits = bits_mean[sorted_indices]
                        sorted_sr = sr_mean[sorted_indices]
                        sorted_sr_max = sr_max[sorted_indices]
                        sorted_sr_min = sr_min[sorted_indices]
                        
                        plt.plot(sorted_bits, sorted_sr, 
                                label=algorithm_label_full,
                                color=color,
                                linestyle='-',
                                linewidth=Linewidth)
                        
                        # 使用最大值和最小值作为边界绘制阴影
                        if sr_max is not None and sr_min is not None:
                            plt.fill_between(sorted_bits, 
                                            sorted_sr_min, 
                                            sorted_sr_max, 
                                            color=color, 
                                            alpha=0.2)
        
        # 绘制bandit数据（虚线）
        if 'bandit' in all_feedback_data and algorithm in all_feedback_data['bandit']:
            data = all_feedback_data['bandit'][algorithm]
            if data['SR']['mean'] is not None and data['Bits']['mean'] is not None and len(data['SR']['mean']) > 0 and len(data['Bits']['mean']) > 0:
                sr_data = data['SR']['mean']
                sr_max_data = data['SR']['max']
                sr_min_data = data['SR']['min']
                bits_data = data['Bits']['mean']
                
                # 根据数据集设置截断长度
                # if dataset == 'HIGGS':
                #     max_rounds = 400
                # elif dataset in ['ijcnn1', 'a9a']:
                #     max_rounds = 700
                # else:
                #     max_rounds = MAX_ROUNDS
                max_rounds = MAX_ROUNDS
                # 首先截断SR数据到指定的max_rounds
                if len(sr_data) > max_rounds:
                    sr_data = sr_data[:max_rounds]
                    sr_max_data = sr_max_data[:max_rounds]
                    sr_min_data = sr_min_data[:max_rounds]
                
                # 然后截断Bits数据，确保与截断后的SR数据长度一致
                min_len = min(len(sr_data), len(bits_data), len(sr_max_data), len(sr_min_data))
                if min_len > 0:
                    sr_mean = sr_data[:min_len]
                    sr_max = sr_max_data[:min_len]
                    sr_min = sr_min_data[:min_len]
                    bits_mean = bits_data[:min_len]
                    
                    # 获取bandit反馈类型的算法显示名称
                    algorithm_label_bandit = get_algorithm_label(algorithm, 'bandit', params)
                    
                    # 绘制曲线
                    if len(bits_mean) > 0 and len(sr_mean) > 0:
                        # 对bits_mean进行排序，确保x轴单调递增
                        sorted_indices = np.argsort(bits_mean)
                        sorted_bits = bits_mean[sorted_indices]
                        sorted_sr = sr_mean[sorted_indices]
                        sorted_sr_max = sr_max[sorted_indices]
                        sorted_sr_min = sr_min[sorted_indices]
                        
                        plt.plot(sorted_bits, sorted_sr, 
                                label=algorithm_label_bandit,
                                color=color,
                                linestyle='--',
                                linewidth=Linewidth)
                        
                        # 使用最大值和最小值作为边界绘制阴影
                        if sr_max is not None and sr_min is not None:
                            plt.fill_between(sorted_bits, 
                                            sorted_sr_min, 
                                            sorted_sr_max, 
                                            color=color, 
                                            alpha=0.2)
    
    # 设置图表属性
    # plt.title(f'Optimal Algorithms Comparison: Bits vs SR ({dataset}, T={params["T"]}, w={compression_rate}, topology={topology})')
    plt.xlabel('Total Communication Bits (log scale)', fontsize=AxisFontdict)
    plt.ylabel('Average Loss', fontsize=AxisFontdict)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=LabelFontdict, loc='lower left')
    plt.xscale('log')
    # 增大刻度标签字体大小
    plt.xticks(fontsize=AxisFontdict)
    plt.yticks(fontsize=AxisFontdict)
    
    # 根据数据集设置纵坐标上界
    if dataset == 'HIGGS':
        plt.ylim(0.325, 0.5)  # 只设置上限，不设置下限
    elif dataset in ['ijcnn1', 'a9a']:
        plt.ylim(auto=True, top=0.8)  # 设置上限为0.8，下限自适应
    
    # 保存图表
    # 保存PNG格式
    save_path = os.path.join(save_dir, f"optimal_comparison_Bits_vs_SR_T{params['T']}_w{compression_rate}_comp{compressor}_topo{topology}_combined.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined Bits vs SR plot to {save_path}")
    # 保存PDF格式
    save_path_pdf = os.path.join(save_dir, f"optimal_comparison_Bits_vs_SR_T{params['T']}_w{compression_rate}_comp{compressor}_topo{topology}_combined.pdf")
    plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight')
    print(f"Saved combined Bits vs SR plot to {save_path_pdf}")
    plt.close()



def main():
    # 遍历每个数据集
    for dataset in DATASET_NAME_TO_LOAD:
        print(f"\nProcessing dataset: {dataset}")

        # 添加生成记录，避免重复生成
        generated_images = set()

        # 遍历每个压缩器类型
        for compressor in COMPRESSOR_TYPES_TO_LOAD:
            print(f"\nProcessing compressor: {compressor}")

            # 遍历每个拓扑结构类型
            for topology in TOPOLOGY_TYPES_TO_LOAD:
                print(f"\nProcessing topology: {topology}")

                # 为当前拓扑结构查找对应的参数文件
                topology_param_files = find_best_param_files(
                    dataset, 
                    compressor_types=[compressor], 
                    topology_types=[topology]
                )

                if not topology_param_files:
                    print(f"No param files found for topology {topology}")
                    continue

                # 遍历当前拓扑结构的参数文件
                for param_file in topology_param_files:
                    print(f"  Processing param file: {param_file}")

                    # 加载最佳参数
                    df = load_best_params(param_file)
                    if df is None:
                        continue

                    # 为每个压缩率保存反馈类型数据的字典
                    compression_rate_feedback_data = {}

                    # 遍历每个反馈类型
                    for feedback in FEEDBACK_TYPES_TO_LOAD:
                        print(f"  Processing feedback: {feedback}")

                        # 遍历每个压缩率
                        for compression_rate in COMPRESSION_RATES:
                            print(f"  Processing compression rate: {compression_rate}")

                            # 过滤当前配置下的最佳参数
                            filtered_params = df[
                                (df['dataset'] == dataset) &
                                (abs(df['w'] - compression_rate) < 1e-6) &
                                (df['data_type'] == feedback)
                            ]

                            # 过滤指定的T值
                            if T_VALUES_TO_LOAD:
                                filtered_params = filtered_params[filtered_params['T'].isin(T_VALUES_TO_LOAD)]


                            if filtered_params.empty:
                                print(f"  No parameters found for {dataset}, {feedback}, w={compression_rate}")
                                continue

                            # 收集所有算法的数据
                            all_algorithms_data = {}
                            reference_params = None

                            # 遍历每个算法
                            for algorithm in ALGORITHMS:
                                # 过滤当前算法的最佳参数
                                if algorithm == 'CDFTGL(L=1)':
                                    # CDFTGL(L=1)使用CDFTGL的参数，L固定为1
                                    algo_params = filtered_params[filtered_params['algorithm'] == 'CDFTGL']
                                else:
                                    algo_params = filtered_params[filtered_params['algorithm'] == algorithm]

                                if algo_params.empty:
                                    print(f"  No parameters found for algorithm {algorithm}")
                                    continue

                                # 只取第一行参数
                                params = algo_params.iloc[0].to_dict()

                                # 保存参考参数用于图表标题
                                if reference_params is None:
                                    reference_params = params

                                # 查找对应的文件
                                matched_files = get_corresponding_files(dataset, compressor, topology, feedback, algorithm, params)

                                if not matched_files:
                                    print(f"  No files found for {algorithm} with params {params}")
                                    continue

                                print(f"  Found {len(matched_files)} files for {algorithm}")

                                # 加载并处理数据
                                data = load_and_process_data(matched_files)

                                # 保存算法数据
                                all_algorithms_data[algorithm] = data

                            # 如果没有找到任何算法数据，跳过
                            if not all_algorithms_data:
                                print(f"  No data found for any algorithm")
                                continue

                            # 检查是否已经生成过该图片
                            image_key = (dataset, compressor, topology, feedback, compression_rate, reference_params['T'])
                            if image_key in generated_images:
                                print(f"  Skipping already generated image for {image_key}")
                                continue
                            generated_images.add(image_key)

                            # 绘制并保存图表，将所有算法绘制在同一张图上（按反馈类型分开）
                            plot_and_save(dataset, compressor, topology, feedback, compression_rate, all_algorithms_data, reference_params)

                            # 保存当前压缩率下的反馈类型数据，用于后续绘制综合图表
                            if compression_rate not in compression_rate_feedback_data:
                                compression_rate_feedback_data[compression_rate] = {
                                    'reference_params': reference_params
                                }
                            compression_rate_feedback_data[compression_rate][feedback] = all_algorithms_data

                    # 遍历每个压缩率，绘制综合图表
                    for compression_rate, feedback_data in compression_rate_feedback_data.items():
                        reference_params = feedback_data['reference_params']

                        # 只处理包含full和bandit数据的情况
                        if 'full' in feedback_data and 'bandit' in feedback_data:
                            print(f"\n  Generating combined chart for compression rate: {compression_rate}")

                            # 收集所有反馈类型的数据
                            all_feedback_data = {
                                'full': feedback_data['full'],
                                'bandit': feedback_data['bandit']
                            }

                            # 检查是否已经生成过该图片
                            combined_image_key = (dataset, compressor, topology, 'combined', compression_rate, reference_params['T'])
                            if combined_image_key in generated_images:
                                print(f"  Skipping already generated combined image for compression rate {compression_rate}")
                                continue
                            generated_images.add(combined_image_key)

                            # 绘制并保存综合图表，将full和bandit数据绘制在同一张图上
                            plot_combined_feedback_charts(dataset, compressor, topology, compression_rate, all_feedback_data, reference_params)

    generate_results_table(results_data)

def generate_results_table(results_data):
    """生成结果表格"""
    if not results_data:
        print("No data to generate table")
        return
    
    print("\n" + "="*80)
    print("**Table 1: Average Loss vs. Communication Rounds across Three Tasks**")
    print("="*80)
    
    # 按反馈类型和算法分组
    bandit_data = [item for item in results_data if item['feedback'] == 'bandit']
    full_data = [item for item in results_data if item['feedback'] == 'full']
    
    # 提取所有数据集
    datasets = sorted(list(set(item['dataset'] for item in results_data)))
    
    # 提取所有算法
    bandit_algorithms = sorted(list(set(item['algorithm'] for item in bandit_data)))
    full_algorithms = sorted(list(set(item['algorithm'] for item in full_data)))
    
    # 打印表格头部
    header = "| Setting | Algorithm | Comm. Rounds |"
    for dataset in datasets:
        header += f" {dataset} Loss |"
    print(header)
    
    separator = "| :--- | :--- | :--- |"
    for _ in datasets:
        separator += " :--- |"
    print(separator)
    
    # 打印Bandit部分
    print("| **Bandit** | **CD-FTBL (Ours)** | **~2,000** |", end="")
    for dataset in datasets:
        # 查找CD-FTBL在该数据集的结果
        cdftbl_result = next((item for item in bandit_data if item['algorithm'] == 'CD-FTBL' and item['dataset'] == dataset), None)
        if cdftbl_result:
            print(f" **{cdftbl_result['final_loss']:.4f}** |", end="")
        else:
            print(" - |", end="")
    print()
    
    # 打印Top-DOBD-1
    print("| | Top-DOBD-1 (Yang2026) | 10,000 |", end="")
    for dataset in datasets:
        topdogd_result = next((item for item in bandit_data if 'Top-DOBD-1' in item['algorithm'] and item['dataset'] == dataset), None)
        if topdogd_result:
            print(f" {topdogd_result['final_loss']:.4f} |", end="")
        else:
            print(" - |", end="")
    print()
    
    # 打印DC-DOBD
    print("| | DC-DOBD (Tu2022) | 10,000 |", end="")
    for dataset in datasets:
        dcdogd_result = next((item for item in bandit_data if 'DC-DOBD' in item['algorithm'] and item['dataset'] == dataset), None)
        if dcdogd_result:
            print(f" {dcdogd_result['final_loss']:.4f} |", end="")
        else:
            print(" - |", end="")
    print()
    
    # 打印Full Info部分
    print("| **Full Info** | **CD-FTGL (Ours)** | **10,000** |", end="")
    for dataset in datasets:
        cdftgl_result = next((item for item in full_data if item['algorithm'] == 'CD-FTGL' and item['dataset'] == dataset), None)
        if cdftgl_result:
            print(f" **{cdftgl_result['final_loss']:.4f}** |", end="")
        else:
            print(" - |", end="")
    print()
    
    # 打印Top-DOGD
    print("| | Top-DOGD (Yang2026) | 10,000 |", end="")
    for dataset in datasets:
        topdogd_result = next((item for item in full_data if 'Top-DOGD' in item['algorithm'] and item['dataset'] == dataset), None)
        if topdogd_result:
            print(f" {topdogd_result['final_loss']:.4f} |", end="")
        else:
            print(" - |", end="")
    print()
    
    # 打印DC-DOGD
    print("| | DC-DOGD (Tu2022) | 10,000 |", end="")
    for dataset in datasets:
        dcdogd_result = next((item for item in full_data if 'DC-DOGD' in item['algorithm'] and item['dataset'] == dataset), None)
        if dcdogd_result:
            print(f" {dcdogd_result['final_loss']:.4f} |", end="")
        else:
            print(" - |", end="")
    print()
    
    print("="*80)


# 生成结果表格

if __name__ == '__main__':
    main()
