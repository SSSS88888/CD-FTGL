import os
import re
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import copy
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 绘图参数
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
Linewidth = 2.5
LabelFontdict = 12
AxisFontdict = 12
# 线型列表
LINESTYLES = ['-', '--', '-.', ':']
# 颜色列表
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 创建结果目录基础路径 - 添加repeated标识
BASE_SAVE_DIR = os.path.join(script_dir, 'result', 'repeated_best_algorithms_comparison_n9')
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

def parse_filename(file_path):
    """解析文件名，提取算法名称、参数、指标类型和数据集名称"""
    base_name = os.path.basename(file_path)
    
    # 提取算法名称
    if ('CDFTGL' in base_name) or ('CD-FTGL' in base_name):
        algorithm = 'CDFTGL'
    elif ('TopDOGD' in base_name) or ('Top-DOGD' in base_name):
        algorithm = 'TopDOGD'
    elif ('DCDOGD' in base_name) or ('DC-DOGD' in base_name):
        algorithm = 'DCDOGD'
    else:
        algorithm = 'Unknown'
    
    # 提取数据集名称、压缩器类型和拓扑结构 - 只支持grid_search_repeated_n9目录
    # 目录结构：grid_search_repeated_n9/dataset_name/compress_type/topology_type/feedback_type
    dataset_name = None
    compressor_type = None
    topology_type = None
    
    # 直接解析目录结构，不使用正则表达式，更可靠
    path_parts = file_path.split(os.sep)
    
    # 查找grid_search_repeated_n9在路径中的位置
    for i, part in enumerate(path_parts):
        if part == 'grid_search_repeated_n9':
            # 确保有足够的后续路径部分
            if i + 4 < len(path_parts):
                dataset_name = path_parts[i+1]
                compressor_type = path_parts[i+2]
                topology_type = path_parts[i+3]
            break
    
    # 如果通过路径分割没有找到，尝试使用正则表达式
    if not dataset_name:
        # 尝试从grid_search_repeated_n9目录提取
        grid_repeated_pattern = r'(grid_search_repeated_n9)/(\w+)/(\w+)/(\w+)/(\w+)'
        grid_repeated_match = re.search(grid_repeated_pattern, file_path)
        if grid_repeated_match:
            dataset_name = grid_repeated_match.group(2)
            compressor_type = grid_repeated_match.group(3)
            topology_type = grid_repeated_match.group(4)
        else:
            # 尝试旧的目录结构（没有compressor和topology）
            dataset_match = re.search(r'(grid_search_repeated_n9)/(\w+)', file_path)
            if dataset_match:
                dataset_name = dataset_match.group(2)
    
    # 提取时间轮次T
    t_match = re.search(r'T(\d+)', base_name)
    time_round = int(t_match.group(1)) if t_match else None
    
    # 提取压缩率w（支持w0.05与w0_05两种格式）
    compression_rate = None
    w_match = re.search(r'w(?:(\d+)_(\d+)|(\d+(?:\.[0-9]+)?))', base_name)
    if w_match:
        if w_match.group(3):
            try:
                compression_rate = float(w_match.group(3))
            except ValueError:
                compression_rate = None
        else:
            try:
                compression_rate = float(f"{w_match.group(1)}.{w_match.group(2)}")
            except ValueError:
                compression_rate = None
    
    # 提取h值 - 支持h0_001格式、h1e-5格式和普通数字格式
    h_value = None
    if algorithm == 'CDFTGL':
        # 首先尝试匹配科学计数法格式（如h1e-5或h1e-05）
        h_match = re.search(r'h(\d+e[-+]\d+)', base_name)
        if h_match:
            try:
                h_value = float(h_match.group(1))
                # 调试输出
                # if '1e-5' in h_match.group(1) or '1e-05' in h_match.group(1):
                #     print(f"成功提取h1e-5或h1e-05格式: {h_value} from {base_name}")
            except ValueError:
                h_value = None
        else:
            # 尝试匹配下划线格式（如h0_001）
            h_match = re.search(r'h(\d+)_(\d+)', base_name)
            if h_match:
                try:
                    h_value = float(f"{h_match.group(1)}.{h_match.group(2)}")
                except ValueError:
                    h_value = None
            else:
                # 尝试匹配普通数字格式（如h10）
                h_match = re.search(r'h(\d+)', base_name)
                if h_match:
                    try:
                        h_value = float(h_match.group(1))
                    except ValueError:
                        h_value = None
    
    # 提取eta值 - 支持eta0_001格式、eta1e-5格式和普通数字格式
    eta_value = None
    if algorithm in ['TopDOGD', 'DCDOGD']:
        # 首先尝试匹配科学计数法格式（如eta1e-5）
        eta_match = re.search(r'eta(\d+e[-+]\d+)', base_name)
        if eta_match:
            try:
                eta_value = float(eta_match.group(1))
            except ValueError:
                eta_value = None
        else:
            # 尝试匹配下划线格式（如eta0_001）
            eta_match = re.search(r'eta(\d+)_(\d+)', base_name)
            if eta_match:
                try:
                    eta_value = float(f"{eta_match.group(1)}.{eta_match.group(2)}")
                except ValueError:
                    eta_value = None
            else:
                # 尝试匹配普通数字格式（如eta10）
                eta_match = re.search(r'eta(\d+)', base_name)
                if eta_match:
                    try:
                        eta_value = float(eta_match.group(1))
                    except ValueError:
                        eta_value = None
    
    # 提取L值（支持Lauto、L10、L10.5、L10_5）
    l_value = None
    # 首先检查是否为auto
    if 'Lauto' in base_name:
        l_value = 'auto'
    else:
        # 尝试匹配带小数点的格式（如L10.5）
        l_match_dec = re.search(r'L(\d+(?:\.[0-9]+)?)', base_name)
        if l_match_dec:
            try:
                l_value = float(l_match_dec.group(1))
            except ValueError:
                l_value = None
        else:
            # 尝试匹配带下划线的格式（如L10_5）
            l_match_us = re.search(r'L(\d+)_(\d+)', base_name)
            if l_match_us:
                try:
                    l_value = float(f"{l_match_us.group(1)}.{l_match_us.group(2)}")
                except ValueError:
                    l_value = None
    
    # 提取eps参数值（支持eps0_001格式）
    eps_value = None
    # 尝试匹配带下划线的格式（如eps0_001）
    eps_match = re.search(r'eps(\d+)_(\d+)', base_name)
    if eps_match:
        try:
            eps_value = float(f"{eps_match.group(1)}.{eps_match.group(2)}")
        except ValueError:
            eps_value = None
    else:
        # 尝试匹配带小数点的格式（如eps0.001）
        eps_match_dec = re.search(r'eps(\d+(?:\.[0-9]+)?)', base_name)
        if eps_match_dec:
            try:
                eps_value = float(eps_match_dec.group(1))
            except ValueError:
                eps_value = None
        else:
            # 尝试匹配科学计数法格式（如eps1e-3）
            eps_match_sci = re.search(r'eps(\d+e[-+]\d+)', base_name)
            if eps_match_sci:
                try:
                    eps_value = float(eps_match_sci.group(1))
                except ValueError:
                    eps_value = None
    
    # 提取指标类型
    if 'SR' in base_name:
        metric = 'SR'
    elif 'Bits' in base_name:
        metric = 'Bits'
    else:
        metric = 'Unknown'
    
    # 根据文件路径判断是否为bandit数据
    is_bandit = 'bandit' in file_path.lower()
    
    return {
        'algorithm': algorithm,
        'time_round': time_round,
        'compression_rate': compression_rate,
        'h_value': h_value,
        'eta_value': eta_value,
        'l_value': l_value,
        'metric': metric,
        'is_bandit': is_bandit,
        'dataset_name': dataset_name,
        'eps_value': eps_value,
        'compressor_type': compressor_type,
        'topology_type': topology_type
    }

# 文件缓存，避免重复加载相同的文件
_file_cache = {}

# 已处理文件计数，用于进度显示
_processed_files = 0
_total_files = 0

def load_result_file(file_path):
    """加载结果文件，返回完整的性能指标数组
    使用缓存机制避免重复加载相同文件，提高性能
    """
    global _processed_files
    
    # 检查缓存
    if file_path in _file_cache:
        return _file_cache[file_path]
    
    try:
        # 使用mmap_mode='r'可以提高大文件加载效率（只读模式）
        data = np.load(file_path, mmap_mode='r')
        # 转换为numpy数组并缓存
        data_array = np.array(data)
        _file_cache[file_path] = data_array
        
        # 更新进度
        _processed_files += 1
        if _total_files > 0 and _processed_files % 10 == 0:
            print(f"Progress: {_processed_files}/{_total_files} files loaded ({_processed_files/_total_files*100:.1f}%)")
            
        return data_array
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def load_all_results(base_dir='result', max_workers=8, dataset_name=None, compressor_types=None, topology_types=None, feedback_types=None):
    """加载所有结果文件，只从grid_search_repeated_n9目录读取
    
    优化：使用线程池并行加载文件，提高性能
    
    Args:
        base_dir: 基础目录
        max_workers: 最大工作线程数
        dataset_name: 要加载的数据集名称，None表示所有，可以是单个字符串或列表
        compressor_types: 要加载的压缩器类型列表，None表示所有
        topology_types: 要加载的拓扑结构类型列表，None表示所有
        feedback_types: 要加载的反馈类型列表，None表示所有
    
    Returns:
        所有加载的结果列表
    """
    start_time = time.time()
    result_files = []
    
    # 只处理grid_search_repeated_n9目录
    grid_search_type = 'grid_search_repeated_n9'
    grid_search_dir = os.path.join(base_dir, grid_search_type)
    if os.path.exists(grid_search_dir):
        # 如果指定了数据集名称，只搜索该数据集目录
        if dataset_name:
            # 确保dataset_name是列表
            dataset_names = dataset_name if isinstance(dataset_name, list) else [dataset_name]
            
            # 遍历所有指定的数据集
            for ds_name in dataset_names:
                dataset_dir = os.path.join(grid_search_dir, ds_name)
                
                # 获取所有压缩器类型的列表
                if compressor_types:
                    compressors = compressor_types
                else:
                    # 获取该数据集目录下的所有压缩器类型
                    if os.path.exists(dataset_dir):
                        compressors = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
                    else:
                        compressors = []
                
                # 遍历所有压缩器类型
                for compressor in compressors:
                    compressor_dir = os.path.join(dataset_dir, compressor)
                    
                    # 获取所有拓扑结构类型的列表
                    if topology_types:
                        topologies = topology_types
                    else:
                        # 获取该压缩器目录下的所有拓扑结构类型
                        if os.path.exists(compressor_dir):
                            topologies = [d for d in os.listdir(compressor_dir) if os.path.isdir(os.path.join(compressor_dir, d))]
                        else:
                            topologies = []
                    
                    # 遍历所有拓扑结构类型
                    for topology in topologies:
                        topology_dir = os.path.join(compressor_dir, topology)
                        
                        # 获取所有反馈类型的列表
                        if feedback_types:
                            feedbacks = feedback_types
                        else:
                            # 获取该拓扑结构目录下的所有反馈类型
                            if os.path.exists(topology_dir):
                                feedbacks = [d for d in os.listdir(topology_dir) if os.path.isdir(os.path.join(topology_dir, d))]
                            else:
                                feedbacks = []
                        
                        # 遍历所有反馈类型
                        for feedback in feedbacks:
                            feedback_dir = os.path.join(topology_dir, feedback)
                            # 搜索该反馈类型目录下的所有.npy文件
                            npy_pattern = os.path.join(feedback_dir, '*.npy')
                            npy_files = glob(npy_pattern)
                            result_files.extend(npy_files)
        else:
            # 如果没有指定数据集名称，使用原来的递归搜索
            filtered_pattern = os.path.join(grid_search_dir, '**/*.npy')
            grid_files = glob(filtered_pattern, recursive=True)
            result_files.extend(grid_files)
    
    # 去重并排序，确保文件处理顺序一致
    result_files = sorted(list(set(result_files)))
    
    print(f"Found {len(result_files)} result files to load")
    
    # 更新全局计数器用于进度显示
    global _total_files, _processed_files
    _total_files = len(result_files)
    _processed_files = 0
    
    results = []
    
    # 定义处理单个文件的函数
    def process_file(file_path):
        file_info = parse_filename(file_path)
        data = load_result_file(file_path)
        
        if data is not None:
            return {
                'file_path': file_path,
                'info': file_info,
                'data': data
            }
        return None
    
    # 使用线程池并行处理文件
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_file = {executor.submit(process_file, file_path): file_path for file_path in result_files}
        
        # 处理完成的任务
        for future in as_completed(future_to_file):
            result = future.result()
            if result is not None:
                results.append(result)
    
    end_time = time.time()
    
    # 对结果列表进行排序，确保返回结果的顺序一致
    results.sort(key=lambda x: x['file_path'])
    
    print(f"Loaded {len(results)} results in {end_time - start_time:.2f} seconds")
    
    return results

def find_best_parameters(results, dataset_name, time_round, compression_rate, l_value, data_type='full', compressor_type=None, topology_type=None):
    """
    为每个算法找到最佳参数设置
    最佳参数定义为在最后一个时间步具有最小SR值的参数组合
    对于grid_search_repeated_n9目录中的结果，计算多次运行的平均值
    
    优化：使用更高效的筛选和分组方法，减少重复计算
    
    Args:
        results: 所有结果列表
        dataset_name: 数据集名称
        time_round: 时间轮次T
        compression_rate: 压缩率w
        l_value: L值
        data_type: 数据类型，'full'或'bandit'
        compressor_type: 压缩器类型，None表示所有
        topology_type: 拓扑结构类型，None表示所有
    """
    is_bandit = (data_type == 'bandit')
    
    # 使用生成器表达式替代列表推导式，减少内存使用
    # 一次性筛选符合条件的SR结果，避免多次遍历
    sr_results = []
    for r in results:
        info = r['info']
        # 提前检查关键条件，快速过滤
        if (info['metric'] == 'SR' and 
            info['dataset_name'] == dataset_name and 
            info['time_round'] == time_round and 
            info['compression_rate'] == compression_rate and 
            info['is_bandit'] == is_bandit):
            # 如果指定了compressor_type，需要检查compressor_type
            if compressor_type is not None and info['compressor_type'] != compressor_type:
                continue
            # 如果指定了topology_type，需要检查topology_type
            if topology_type is not None and info['topology_type'] != topology_type:
                continue
            # 如果指定了L值，并且不是DCDOGD算法，需要检查L值
            if l_value is None or info['algorithm'] == 'DCDOGD' or info['l_value'] == l_value:
                # 只有在bandit场景下才检查eps值
                if is_bandit:
                    # 设置允许的eps值列表
                    allowed_eps_values = [0.1, 1, 10]
                    # 检查eps_value是否在允许列表中
                    if info['eps_value'] in allowed_eps_values:
                        sr_results.append(r)
                else:
                    # full场景下不需要检查eps值
                    sr_results.append(r)
    
    # 提前返回空结果
    if not sr_results:
        return {}
    
    # 使用字典进行算法分组，避免多次遍历
    algo_results_map = {}
    for r in sr_results:
        algo = r['info']['algorithm']
        if algo not in algo_results_map:
            algo_results_map[algo] = []
        algo_results_map[algo].append(r)
    
    best_params = {}
    
    for algorithm, algo_results in algo_results_map.items():
        # 为每个参数组合分组
        param_combos = {}
        
        # 预计算参数键和repeated标志
        for result in algo_results:
            # 提前提取文件路径
            file_path = result['file_path']
            # 一次性获取所有需要的信息
            info = result['info']
            
            # 判断是否为repeated结果
            is_repeated = 'grid_search_repeated_n9' in file_path
            
            # 获取参数组合键
            if algorithm == 'CDFTGL':
                param_key = (info['h_value'], info['eps_value'])
            else:  # TopDOGD, DCDOGD
                param_key = (info['eta_value'], info['eps_value'])
            
            # 更新参数组合
            if param_key not in param_combos:
                param_combos[param_key] = {
                    'results': [],
                    'is_repeated': is_repeated
                }
            else:
                param_combos[param_key]['is_repeated'] |= is_repeated  # 逻辑或运算更高效
            
            param_combos[param_key]['results'].append(result)
        
        # 为每个参数组合计算最终SR值
        param_final_sr = {}
        
        for param_key, combo in param_combos.items():
            combo_results = combo['results']
            if not combo_results:
                continue
            
            if combo['is_repeated']:
                # 优化：一次性收集所有有效数据，避免多次循环
                valid_results = [r for r in combo_results if len(r['data']) > 0]
                if valid_results:
                    # 一次性计算平均值，避免多次访问列表
                    final_srs = [r['data'][-1] for r in valid_results]
                    avg_final_sr = np.mean(final_srs)
                    
                    # 处理不同长度的data数组
                    data_list = [r['data'] for r in valid_results]
                    # 找到最短数组长度
                    min_length = min(len(data) for data in data_list)
                    # 截断所有数组到最短长度
                    truncated_data = [data[:min_length] for data in data_list]
                    # 转换为numpy数组并计算平均值
                    data_arrays = np.array(truncated_data)
                    avg_data = np.mean(data_arrays, axis=0)
                    
                    param_final_sr[param_key] = {
                        'sr': avg_final_sr,
                        'result': valid_results[0],
                        'avg_data': avg_data
                    }
            else:
                # 对于非repeated结果，直接找到最佳结果
                best_sr = float('inf')
                best_result = None
                
                for result in combo_results:
                    data = result['data']
                    if len(data) > 0:
                        final_sr = data[-1]
                        if final_sr < best_sr:
                            best_sr = final_sr
                            best_result = result
                
                if best_result is not None:
                    param_final_sr[param_key] = {
                        'sr': best_sr,
                        'result': best_result,
                        'avg_data': best_result['data']
                    }
        
        # 找到SR最小的参数组合
        if param_final_sr:
            # 使用min函数的key参数进行一次性查找，添加平局处理规则
            # 当SR值相同时，按参数键排序，确保选择一致的参数组合
            best_param = min(param_final_sr.items(), key=lambda x: (x[1]['sr'], x[0]))
            # 创建结果对象，避免深拷贝整个对象，只复制必要信息
            best_result = {
                'file_path': best_param[1]['result']['file_path'],
                'info': best_param[1]['result']['info'],
                'data': best_param[1]['avg_data']
            }
            best_params[algorithm] = best_result
    
    return best_params

def find_optimal_l_value(results, dataset_name, time_round, compression_rate, data_type, compressor_type, topology_type):
    """为每个算法找到各自的最优L值（基于最后一个时间步的SR值）
    
    Args:
        results: 所有结果列表
        dataset_name: 数据集名称
        time_round: 时间轮次T
        compression_rate: 压缩率w
        data_type: 数据类型，'full'或'bandit'
        compressor_type: 压缩器类型
        topology_type: 拓扑结构类型
        
    Returns:
        字典，键为算法名称，值为该算法的最优L值；没有L值的算法（如DCDOGD）不会出现在字典中
    """
    # 获取所有有L值的算法
    is_bandit = (data_type == 'bandit')
    # 与find_best_parameters函数保持一致的eps值列表
    allowed_eps_values = [0.1, 1, 10]
    
    # 获取所有有L值的算法
    algorithms_with_l = set(r['info']['algorithm'] for r in results 
                          if r['info']['dataset_name'] == dataset_name
                          and r['info']['time_round'] == time_round
                          and r['info']['compression_rate'] == compression_rate
                          and r['info']['is_bandit'] == is_bandit
                          and r['info']['compressor_type'] == compressor_type
                          and r['info']['topology_type'] == topology_type
                          and r['info']['l_value'] is not None
                          and r['info']['metric'] == 'SR'
                          and (not is_bandit or r['info']['eps_value'] in allowed_eps_values))
    
    # 提取所有不同的L值
    all_l_values = set(r['info']['l_value'] for r in results 
                      if r['info']['dataset_name'] == dataset_name
                      and r['info']['time_round'] == time_round
                      and r['info']['compression_rate'] == compression_rate
                      and r['info']['is_bandit'] == is_bandit
                      and r['info']['compressor_type'] == compressor_type
                      and r['info']['topology_type'] == topology_type
                      and r['info']['l_value'] is not None
                      and r['info']['metric'] == 'SR'
                      and (not is_bandit or r['info']['eps_value'] in allowed_eps_values))
    
    optimal_l_values = {}
    
    # 为每个算法找到最优L值
    for algorithm in algorithms_with_l:
        # DCDOGD没有L参数，跳过
        if algorithm == 'DCDOGD':
            continue
        
        best_l = None
        best_sr = float('inf')  # SR值越小越好
        
        # 根据算法类型确定L值搜索范围
        if algorithm == 'CDFTGL' or algorithm == 'TopDOGD':
            # 限定L值搜索范围为[10, 100, 1000]
            allowed_l_values = [10, 100, 1000]
        else:
            # 其他算法使用所有可用的L值
            allowed_l_values = all_l_values
        
        # 只考虑在allowed_l_values和all_l_values中的交集
        l_values_to_search = [l for l in allowed_l_values if l in all_l_values]
        
        for l in l_values_to_search:
            # 查找该算法在该L值下的最佳参数
            best_params = find_best_parameters(results, dataset_name, time_round, compression_rate, l, data_type, compressor_type, topology_type)
            
            if best_params and algorithm in best_params:
                result = best_params[algorithm]
                if len(result['data']) > 0:
                    final_sr = result['data'][-1]
                    # 如果这是目前找到的最佳SR值，更新最佳L值
                    if final_sr < best_sr:
                        best_sr = final_sr
                        best_l = l
        
        if best_l is not None:
            optimal_l_values[algorithm] = best_l
    
    return optimal_l_values

def save_best_parameters_table(all_best_params, save_dir, dataset_name=None, compressor_types=None, topology_types=None, feedback_types=None):
    """
    将所有情况的最优参数保存为表格形式
    full模式保存dataset, algorithm, T, w, L, data_type, h_coeff/eta_coeff
    bandit模式额外保存eps_coeff
    
    Args:
        all_best_params: 所有情况的最佳参数
        save_dir: 保存目录
        dataset_name: 数据集名称，用于命名文件，可以是单个字符串或列表
        compressor_types: 压缩器类型列表，用于命名文件
        topology_types: 拓扑结构类型列表，用于命名文件
        feedback_types: 反馈类型列表，用于命名文件
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 准备表格数据
    table_data = []
    
    for key, best_params in all_best_params.items():
        # 处理不同长度的键
        if len(key) == 8:
            # 新的键格式：(dataset_name, data_type, t, w, l, compressor, topology, algo)
            key_dataset_name, data_type, t, w, l, compressor, topology, algo = key
        elif len(key) == 7:
            # 旧的键格式：(dataset_name, data_type, t, w, l, compressor, topology)
            key_dataset_name, data_type, t, w, l, compressor, topology = key
        else:
            # 更旧的键格式：(dataset_name, data_type, t, w, l)
            key_dataset_name, data_type, t, w, l = key
            compressor = None
            topology = None
        
        for algorithm, result in best_params.items():
            info = result['info']
            row = {
                'dataset': key_dataset_name,
                'algorithm': algorithm,
                'T': t,
                'w': w,
                'L': l if l is not None else '',
                'data_type': data_type,
                'compressor_type': compressor or info.get('compressor_type', ''),
                'topology_type': topology or info.get('topology_type', '')
            }
            
            # 根据算法类型添加系数
            if algorithm == 'CDFTGL' or algorithm == 'CDFTGL(L=1)':
                row['h_coeff'] = info['h_value']
            else:
                row['eta_coeff'] = info['eta_value']
            
            # bandit模式添加eps_coeff
            if data_type == 'bandit':
                row['eps_coeff'] = info['eps_value']
            
            table_data.append(row)
    
    if not table_data:
        print("No best parameters to save as table")
        return
    
    # 创建DataFrame
    df = pd.DataFrame(table_data)
    
    # 为每个数据集生成单独的表格文件
    unique_datasets = df['dataset'].unique()
    
    for current_dataset in unique_datasets:
        # 过滤当前数据集的数据
        dataset_df = df[df['dataset'] == current_dataset]
        
        # 保存每个压缩器类型和拓扑结构组合的独立文件，只保存_all.csv
        if topology_types:
            for topology in topology_types:
                # 过滤当前拓扑结构的数据
                topology_df = dataset_df[dataset_df['topology_type'] == topology]
                if not topology_df.empty:
                    if compressor_types:
                        # 为每个压缩器类型单独保存文件
                        for compressor in compressor_types:
                            # 过滤当前压缩器类型的数据
                            compressor_df = topology_df[topology_df['compressor_type'] == compressor]
                            if not compressor_df.empty:
                                # 构建文件名前缀，包含当前数据集、压缩器类型和拓扑结构的信息
                                file_prefix = f"repeated_best_parameters_{current_dataset}"
                                # 只使用当前压缩器类型作为文件名的一部分
                                file_prefix += f"_comp{compressor}"
                                # 只使用当前拓扑结构作为文件名的一部分
                                file_prefix += f"_topo{topology}"
                                
                                # 保存汇总的csv文件
                                summary_file_path = os.path.join(save_dir, f"{file_prefix}_all.csv")
                                compressor_df.to_csv(summary_file_path, index=False)
                                print(f"Summary repeated best parameters table saved to {summary_file_path}")
                    else:
                        # 如果没有指定压缩器类型，保存所有压缩器类型的数据
                        file_prefix = f"repeated_best_parameters_{current_dataset}"
                        # 只使用当前拓扑结构作为文件名的一部分
                        file_prefix += f"_topo{topology}"
                        
                        # 保存汇总的csv文件
                        summary_file_path = os.path.join(save_dir, f"{file_prefix}_all.csv")
                        topology_df.to_csv(summary_file_path, index=False)
                        print(f"Summary repeated best parameters table saved to {summary_file_path}")
        else:
            # 如果没有指定拓扑结构，保存所有拓扑结构的数据
            if compressor_types:
                # 为每个压缩器类型单独保存文件
                for compressor in compressor_types:
                    # 过滤当前压缩器类型的数据
                    compressor_df = dataset_df[dataset_df['compressor_type'] == compressor]
                    if not compressor_df.empty:
                        # 构建文件名前缀，包含当前数据集和压缩器类型的信息
                        file_prefix = f"repeated_best_parameters_{current_dataset}"
                        # 只使用当前压缩器类型作为文件名的一部分
                        file_prefix += f"_comp{compressor}"
                        
                        # 保存汇总的csv文件
                        summary_file_path = os.path.join(save_dir, f"{file_prefix}_all.csv")
                        compressor_df.to_csv(summary_file_path, index=False)
                        print(f"Summary repeated best parameters table saved to {summary_file_path}")
            else:
                # 如果没有指定压缩器类型和拓扑结构，保存所有数据
                file_prefix = f"repeated_best_parameters_{current_dataset}"
                
                # 保存汇总的csv文件
                summary_file_path = os.path.join(save_dir, f"{file_prefix}_all.csv")
                dataset_df.to_csv(summary_file_path, index=False)
                print(f"Summary repeated best parameters table saved to {summary_file_path}")






def main():
    """主函数，加载所有结果并比较最佳参数性能"""
    # 1. 让用户指定要加载的数据集、压缩方法、拓扑类型、反馈类型
    # 这里可以根据需要修改为命令行参数或配置文件读取
    # 当前使用硬编码方式，用户可以直接修改这些变量
    
    # 要加载的数据集名称，None表示所有
    DATASET_NAME_TO_LOAD = ['ijcnn1']  # 例如：['HIGGS', 'a9a', 'ijcnn1']
    
    # 要加载的压缩器类型列表，None表示所有
    COMPRESSOR_TYPES_TO_LOAD = ['top']  # 例如：['top', 'gossip', 'quant', 'rand']
    
    # 要加载的拓扑结构类型列表，None表示所有
    TOPOLOGY_TYPES_TO_LOAD = ['random']  # 例如：['random', 'fully_connected', 'ring']
    
    # 要加载的反馈类型列表，None表示所有
    FEEDBACK_TYPES_TO_LOAD = ['full','bandit']  # 例如：['full', 'bandit']
    
    # 2. 加载指定的结果文件
    print(f"Loading results for dataset: {DATASET_NAME_TO_LOAD}")
    print(f"Loading results for compressor types: {COMPRESSOR_TYPES_TO_LOAD}")
    print(f"Loading results for topology types: {TOPOLOGY_TYPES_TO_LOAD}")
    print(f"Loading results for feedback types: {FEEDBACK_TYPES_TO_LOAD}")
    
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建正确的结果目录路径
    result_dir = os.path.join(script_dir, 'result')
    
    all_results = load_all_results(
        base_dir=result_dir,
        dataset_name=DATASET_NAME_TO_LOAD,
        compressor_types=COMPRESSOR_TYPES_TO_LOAD,
        topology_types=TOPOLOGY_TYPES_TO_LOAD,
        feedback_types=FEEDBACK_TYPES_TO_LOAD
    )
    
    if not all_results:
        print("No result files found.")
        return
    
    # 用于存储所有情况的最佳参数
    all_best_params = {}
    
    # 获取所有不同的时间轮次、压缩率、L值、数据集名称、压缩器类型和拓扑结构
    time_rounds = set(r['info']['time_round'] for r in all_results if r['info']['time_round'] is not None)
    compression_rates = set(r['info']['compression_rate'] for r in all_results if r['info']['compression_rate'] is not None)
    l_values = set(r['info']['l_value'] for r in all_results if r['info']['l_value'] is not None)
    dataset_names = set(r['info']['dataset_name'] for r in all_results if r['info']['dataset_name'] is not None)
    all_compressor_types = set(r['info']['compressor_type'] for r in all_results if r['info']['compressor_type'] is not None)
    all_topology_types = set(r['info']['topology_type'] for r in all_results if r['info']['topology_type'] is not None)
    
    print(f"Found time rounds: {time_rounds}")
    print(f"Found compression rates: {compression_rates}")
    print(f"Found L values: {l_values}")
    print(f"Found datasets: {dataset_names}")
    print(f"Found compressor types: {all_compressor_types}")
    print(f"Found topology types: {all_topology_types}")
    
    # 用户指定要比较的压缩器类型和拓扑结构
    # 可以根据需要修改这些列表，只比较指定的类型
    COMPRESSOR_TYPES_TO_COMPARE = COMPRESSOR_TYPES_TO_LOAD  # 默认比较所有加载的压缩器类型
    TOPOLOGY_TYPES_TO_COMPARE = TOPOLOGY_TYPES_TO_LOAD  # 默认比较所有加载的拓扑结构
    
    # 示例：只比较特定的压缩器类型和拓扑结构
    # COMPRESSOR_TYPES_TO_COMPARE = ['gossip', 'top']
    # TOPOLOGY_TYPES_TO_COMPARE = ['random', 'fully_connected']
    
    print(f"\nWill compare compressor types: {COMPRESSOR_TYPES_TO_COMPARE}")
    print(f"Will compare topology types: {TOPOLOGY_TYPES_TO_COMPARE}")
    
    # 处理指定的反馈类型
    data_types = FEEDBACK_TYPES_TO_LOAD
    # 如果需要只处理一种类型，可以取消下面这行的注释
    # data_types = ['full']
    
    # 为每个数据集、T、w、compressor_type、topology_type组合和数据类型生成比较图表
    for dataset_name in dataset_names:
        print(f"\nProcessing dataset: {dataset_name}")
        for data_type in data_types:
            print(f"\nGenerating charts for {data_type} data...")
            for t in time_rounds:
                for w in compression_rates:
                    # 处理用户指定的压缩器类型
                    for compressor in COMPRESSOR_TYPES_TO_COMPARE:
                        # 处理用户指定的拓扑结构
                        for topology in TOPOLOGY_TYPES_TO_COMPARE:
                            print(f"\nProcessing {dataset_name}, T={t}, w={w}, compressor={compressor}, topology={topology}, data_type={data_type}...")
                            
                            # 1. 找到每个算法各自的最优L值
                            print("Finding optimal L values for each algorithm...")
                            optimal_l_values = find_optimal_l_value(all_results, dataset_name, t, w, data_type, compressor, topology)
                            
                            print(f"Found optimal L values: {optimal_l_values}")
                            
                            # 2. 为每个算法保存其最优L值下的结果
                            print("Finding best parameters for each algorithm with their optimal L values...")
                            # 对于有L值的算法，为每个算法分别查找其最优L值下的最佳参数
                            for algo, optimal_l in optimal_l_values.items():
                                # 获取该算法在其最优L值下的最佳参数
                                algo_optimal_params = find_best_parameters(all_results, dataset_name, t, w, optimal_l, data_type, compressor, topology)
                                
                                if algo_optimal_params and algo in algo_optimal_params:
                                    print(f"Found best parameters for {algo} with optimal L={optimal_l}")
                                    # 存储该算法的最优参数
                                    all_best_params[(dataset_name, data_type, t, w, optimal_l, compressor, topology, algo)] = {algo: algo_optimal_params[algo]}
                            
                            # 3. 为CDFTGL算法专门处理L=1的情况
                            print("Finding best parameters for CDFTGL with L=1...")
                            # 获取所有算法在L=1时的最佳参数
                            all_l1_params = find_best_parameters(all_results, dataset_name, t, w, 1, data_type, compressor, topology)
                            
                            if all_l1_params and 'CDFTGL' in all_l1_params:
                                print("Found best parameters for CDFTGL with L=1")
                                # 只保存CDFTGL的L=1结果，并将算法名改为CDFTGL(L=1)
                                cdftgl_l1_result = all_l1_params['CDFTGL'].copy()
                                # 创建一个新的最佳参数字典，只包含CDFTGL(L=1)
                                l1_best_params = {
                                    'CDFTGL(L=1)': cdftgl_l1_result
                                }
                                # 存储CDFTGL(L=1)的最佳参数
                                all_best_params[(dataset_name, data_type, t, w, 1, compressor, topology, 'CDFTGL(L=1)')] = l1_best_params
                            
                            # 4. 处理DCDOGD算法（没有L参数）
                            print("Finding best parameters for DCDOGD...")
                            # 为DCDOGD单独查找最佳参数（不指定L值）
                            dcdogd_params = find_best_parameters(all_results, dataset_name, t, w, None, data_type, compressor, topology)
                            
                            if dcdogd_params and 'DCDOGD' in dcdogd_params:
                                print("Found best parameters for DCDOGD")
                                # 只保存DCDOGD的结果
                                all_best_params[(dataset_name, data_type, t, w, None, compressor, topology, 'DCDOGD')] = {'DCDOGD': dcdogd_params['DCDOGD']}
    
    # 保存所有最佳参数为表格
    print("\n保存所有最佳参数为表格...")
    params_save_dir = os.path.join(BASE_SAVE_DIR, "best_parameters_tables")
    save_best_parameters_table(
        all_best_params, 
        params_save_dir,
        dataset_name=DATASET_NAME_TO_LOAD,
        compressor_types=COMPRESSOR_TYPES_TO_LOAD,
        topology_types=TOPOLOGY_TYPES_TO_LOAD,
        feedback_types=FEEDBACK_TYPES_TO_LOAD
    )
    
    print("\n所有最佳参数表格保存完成!")

if __name__ == "__main__":
    main()
