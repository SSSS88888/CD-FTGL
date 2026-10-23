import sys
import os
import numpy as np
import itertools
import time
import argparse

# 将上级目录添加到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 所有路径都以本脚本所在目录为基准，因此可以在任意目录下运行
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SCRIPT_DIR, 'result')
DATA_DIR = os.path.join(SCRIPT_DIR, os.pardir, 'data')

# 导入所需的模型类和参数类
from parameter_TopDOGD import TopDOGDParameters
from parameter_CDFTGL import CDFTGLParameters
from parameter_DCDOGD import DCDOGDParameters
from Top_DOGD import TopDOGD
from CD_FTGL import CDFTGL
from DC_DOGD import DCDOGD



# 基础实验参数
BATCH_SIZE = 1
STRONG_CONV = 0
AGENT_NUM = 50
EDGE_NUM = 100
G_lipschitz = 10
R_bound = 10
r_bound = 9
M_bound = 10
TOPOLOGY = 'full'
COMPRESSOR_TYPE = 'top' # 压缩器类型 [rand, quant, gossip, top]
DISTRIBUTION_TYPE = 'random' # 分布类型 [label, random]
DATASET_NAME = 'ijcnn1'   # 可选择: 'HIGGS', 'a9a', 'ijcnn1', 'diabetes'
FEEDBACK_TYPE = 'one-bandit'  # 反馈类型 [full, one-bandit]

# 固定的五个随机数种子
SEEDS = [42]
# SEEDS=[42]

# 要运行的算法列表，可选择: ['CDFTGL', 'TopDOGD', 'DCDOGD']
ALGORITHMS_TO_RUN = ['CDFTGL', 'TopDOGD', 'DCDOGD']

# 创建结果目录，包含数据集子目录、压缩器类型、拓扑结构和反馈类型子目录
# 将拓扑结构为"full"的情况转换为"fully connected"，以与full information区分
topology_dir = TOPOLOGY if TOPOLOGY != 'full' else 'fully_connected'
feedback_dir = f"bandit" if FEEDBACK_TYPE != 'full' else f"full"
os.makedirs(os.path.join(RESULT_DIR, 'grid_search_repeated_n50', DATASET_NAME, COMPRESSOR_TYPE, topology_dir, feedback_dir), exist_ok=True)

# 网格搜索参数设置
T_VALUES = [10000]  # 时间步数 [10000, 50000]
w_VALUES = [0.1, 0.5]  # 压缩率参数
# w_VALUES = [0.05, 0.1, 0.5]
# w_VALUES = [0.1, 0.5]

# CD-FTGL特定参数
# h_coeff_VALUES = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]  # h_coeff范围
h_coeff_VALUES = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
# Top-DOGD和DC-DOGD特定参数
eta_coeff_VALUES = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000, 100000]  # eta_coeff范围

# eps_manual_VALUES = [0.1, 1, 10]  # eps_manual范围
eps_manual_VALUES = [1]
# L_VALUES = ['auto', 1, 10, 100, 1000]  # L参数，'auto'表示自动计算
# L_VALUES = [1, 10, 100, 1000]
L_VALUES = [10, 100]

# 加载数据集
def load_dataset(DATASET_NAME):
    if DATASET_NAME == 'HIGGS':
        A = np.load(os.path.join(DATA_DIR, 'data_HIGGS', 'A_HIGGS.npy'))
        y = np.load(os.path.join(DATA_DIR, 'data_HIGGS', 'y_HIGGS.npy'))
        print("# Number of features: 28")
        print(f"# Number of samples: 10999999")
    elif DATASET_NAME == 'a9a':
        A = np.load(os.path.join(DATA_DIR, 'data_a9a', 'A_a9a.npy'))
        y = np.load(os.path.join(DATA_DIR, 'data_a9a', 'y_a9a.npy'))
        print("# Number of features: 123")
        print(f"# Number of samples: 32561")
    elif DATASET_NAME == 'ijcnn1':
        A = np.load(os.path.join(DATA_DIR, 'data_ijcnn1', 'A_ijcnn1.npy'))
        y = np.load(os.path.join(DATA_DIR, 'data_ijcnn1', 'y_ijcnn1.npy'))
        print("# Number of features: 22")
        print(f"# Number of samples: 141691")
    elif DATASET_NAME == 'diabetes':
        A = np.load(os.path.join(DATA_DIR, 'data_diabetes', 'A_diabetes.npy'))
        y = np.load(os.path.join(DATA_DIR, 'data_diabetes', 'y_diabetes.npy'))
        print("# Number of features: 8")
        print(f"# Number of samples: 70692")
    else:
        raise ValueError(f"不支持的数据集: {DATASET_NAME}，可选数据集: 'HIGGS', 'a9a', 'ijcnn1', 'diabetes'")
    
    # 创建测试集
    np.random.seed(42)  # 设置随机种子确保可重复性
    num = A.shape[0]
    # 对于大型数据集，采样10%的数据作为测试集已足够准确反映模型性能
    test_size = min(int(0.1 * num), 10000)  # 最多采样10000个样本
    A_test, y_test = None, None
    test_num = 0
    if test_size > 0:
        # 随机选择测试集索引
        indices = np.random.permutation(num)
        test_indices = indices[:test_size]
        # 创建测试集
        A_test = A[test_indices]
        y_test = y[test_indices]
        test_num = test_size
        print(f"# Test set size: {test_num}")
    else:
        print("# No test set created")
    
    return A, y, A_test, y_test, test_num

# 创建CDFTGL参数
def create_CDFTGL_params(T, w, h_coeff, eps_manual, L_value):
    # 处理L参数 - 根据用户要求，当L_value='auto'时设置为None
    L_manual = None if L_value == 'auto' else L_value
    
    return CDFTGLParameters(
        alpha=STRONG_CONV,
        compressor_type=COMPRESSOR_TYPE,
        w=w,
        s=2,
        topology=TOPOLOGY,
        agent_num=AGENT_NUM,
        edge_num=EDGE_NUM,
        batch_size=BATCH_SIZE,
        label=f'CD-FTGL-T{T}-w{w}-h{h_coeff}-L{L_value}',
        distribution_type=DISTRIBUTION_TYPE,
        G=G_lipschitz, 
        R=R_bound,
        r=r_bound,
        h_coeff=h_coeff,
        # 始终传递eps_coeff参数，不再根据反馈类型条件设置
        eps_coeff=1,
        T=T,
        feedback_type=FEEDBACK_TYPE,
        manual_params=True,
        L_manual=L_manual,  # 当L_value='auto'时这里为None，会触发自动计算
        L_prime_manual=L_manual / 5,  # 可直接指定L'值
        eps_manual=eps_manual,
    )

# 创建TopDOGD参数
def create_TopDOGD_params(T, w, eta_coeff, eps_manual, L_value):
    # 处理L参数 - 根据用户要求，当L_value='auto'时设置为None
    L1_manual = None if L_value == 'auto' else L_value
    # L2_manual = None if L_value == 'auto' else L_value
    
    return TopDOGDParameters(
        alpha=STRONG_CONV,
        compressor_type=COMPRESSOR_TYPE,
        w=w,
        s=2,
        topology=TOPOLOGY,
        agent_num=AGENT_NUM,
        edge_num=EDGE_NUM,
        batch_size=BATCH_SIZE,
        label=f'Top-DOGD-T{T}-w{w}-eta{eta_coeff}-L{L_value}',
        distribution_type=DISTRIBUTION_TYPE,
        G=G_lipschitz,  # 初始设置
        R=R_bound,
        r=r_bound,
        eta_coeff=eta_coeff,
        # 始终传递eps_coeff参数，不再根据反馈类型条件设置
        eps_coeff=1,
        T=T,
        feedback_type=FEEDBACK_TYPE,
        manual_params=True,
        # eta_manual=0.01,
        L1_manual=L1_manual,  # 当L_value='auto'时这里为None，会触发自动计算
        # L2_manual=L2_manual  # 当L_value='auto'时这里为None，会触发自动计算
        # gamma_manual=0.26,
        eps_manual=eps_manual,
    )

# 创建DCDOGD参数 - 修改：移除L_value参数
def create_DCDOGD_params(T, w, eta_coeff, eps_manual):
    return DCDOGDParameters(
        alpha=STRONG_CONV,
        compressor_type=COMPRESSOR_TYPE,
        w=w,
        s=2,
        topology=TOPOLOGY,
        agent_num=AGENT_NUM,
        edge_num=EDGE_NUM,
        batch_size=BATCH_SIZE,
        label=f'DCDOGD-T{T}-w{w}-eta{eta_coeff}',
        distribution_type=DISTRIBUTION_TYPE,
        L=G_lipschitz, 
        D=2 * R_bound,
        R=R_bound,
        r=r_bound,
        C=M_bound,
        T=T,
        feedback_type=FEEDBACK_TYPE,
        tune=True,
        # gamma=0.26,
        eta_coeff=eta_coeff,
        # 始终传递eps_coeff参数，DC_DOGD类内部会正确处理
        eps_coeff=1,
        eps_manual=eps_manual,
        c=1
    )

# 生成唯一的文件名
def generate_filename(model_type, T, w, coeff_name, coeff_value, L_value=None, eps_manual=None, seed=None):
    # 替换文件名中的特殊字符
    w_str = str(w).replace('.', '_')
    coeff_str = str(coeff_value).replace('.', '_')
    
    # 根据反馈类型选择不同的保存目录
    feedback_dir = 'bandit' if FEEDBACK_TYPE != 'full' else 'full'
    # 将拓扑结构为"full"的情况转换为"fully connected"，以与full information区分
    topology_dir = TOPOLOGY if TOPOLOGY != 'full' else 'fully_connected'
    
    # 构建基础文件名
    if model_type == 'DCDOGD':
        # DCDOGD模型不需要L参数
        base_name = f"{model_type}_T{T}_w{w_str}_{coeff_name}{coeff_str}"
    else:
        # 其他模型需要L参数
        L_str = str(L_value).replace('.', '_') if L_value is not None else 'None'
        base_name = f"{model_type}_T{T}_w{w_str}_{coeff_name}{coeff_str}_L{L_str}"
    
    # 只在bandit场景下添加eps信息和bandit标识
    if FEEDBACK_TYPE != 'full':
        # 添加bandit标识
        base_name = f"{base_name}_bandit"
        # 添加eps_manual信息
        if eps_manual is not None:
            eps_str = str(eps_manual).replace('.', '_')
            base_name = f"{base_name}_eps{eps_str}"
    
    # 添加seed信息
    if seed is not None:
        base_name = f"{base_name}_seed{seed}"
    
    # 修改为保存到数据集目录下，包含压缩器类型和拓扑结构
    return os.path.join(RESULT_DIR, 'grid_search_repeated_n50', DATASET_NAME, COMPRESSOR_TYPE, topology_dir, feedback_dir, base_name)

# 定义任务类型
class ExperimentTask:
    def __init__(self, model_type, params_func, params_args, filename_args):
        self.model_type = model_type
        self.params_func = params_func
        self.params_args = params_args
        self.filename_args = filename_args

# 运行网格搜索
def run_grid_search(start_task=1, end_task=None):
    A, y, A_test, y_test, test_num = load_dataset(DATASET_NAME)
    
    # 记录开始时间
    start_time = time.time()
    
    # 生成所有任务列表
    all_tasks = []
    
    # 根据用户要求，调整实验计数逻辑，考虑每个参数组合会运行5次不同的种子
    num_seeds = len(SEEDS)  # 种子数量
    
    print("\n生成所有任务列表...")
    
    # 1. CD-FTGL 任务生成
    if "CDFTGL" in ALGORITHMS_TO_RUN:
        print("生成CD-FTGL任务...")
        if FEEDBACK_TYPE != 'full':
            # bandit场景：遍历所有eps_manual值
            for T, w, h_coeff, eps_manual, L_value in itertools.product(
                    T_VALUES, w_VALUES, h_coeff_VALUES, eps_manual_VALUES, L_VALUES):
                
                # 对每个参数组合遍历五个种子
                for seed in SEEDS:
                    task = ExperimentTask(
                        model_type="CDFTGL",
                        params_func=create_CDFTGL_params,
                        params_args=(T, w, h_coeff, eps_manual, L_value),
                        filename_args=("CDFTGL", T, w, "h", h_coeff, L_value, eps_manual, seed)
                    )
                    all_tasks.append(task)
        else:
            # full场景：不遍历eps_manual值，固定使用一个值
            eps_manual = 1.0  # 任意固定值
            for T, w, h_coeff, L_value in itertools.product(
                    T_VALUES, w_VALUES, h_coeff_VALUES, L_VALUES):
                
                # 对每个参数组合遍历五个种子
                for seed in SEEDS:
                    task = ExperimentTask(
                        model_type="CDFTGL",
                        params_func=create_CDFTGL_params,
                        params_args=(T, w, h_coeff, eps_manual, L_value),
                        filename_args=("CDFTGL", T, w, "h", h_coeff, L_value, eps_manual, seed)
                    )
                    all_tasks.append(task)
    
    # 2. Top-DOGD 任务生成
    if "TopDOGD" in ALGORITHMS_TO_RUN:
        print("生成Top-DOGD任务...")
        if FEEDBACK_TYPE != 'full':
            # bandit场景：遍历所有eps_manual值
            for T, w, eta_coeff, eps_manual, L_value in itertools.product(
                    T_VALUES, w_VALUES, eta_coeff_VALUES, eps_manual_VALUES, L_VALUES):
                
                # 对每个参数组合遍历五个种子
                for seed in SEEDS:
                    task = ExperimentTask(
                        model_type="TopDOGD",
                        params_func=create_TopDOGD_params,
                        params_args=(T, w, eta_coeff, eps_manual, L_value),
                        filename_args=("TopDOGD", T, w, "eta", eta_coeff, L_value, eps_manual, seed)
                    )
                    all_tasks.append(task)
        else:
            # full场景：不遍历eps_manual值，固定使用一个值
            eps_manual = 1.0  # 任意固定值
            for T, w, eta_coeff, L_value in itertools.product(
                    T_VALUES, w_VALUES, eta_coeff_VALUES, L_VALUES):
                
                # 对每个参数组合遍历五个种子
                for seed in SEEDS:
                    task = ExperimentTask(
                        model_type="TopDOGD",
                        params_func=create_TopDOGD_params,
                        params_args=(T, w, eta_coeff, eps_manual, L_value),
                        filename_args=("TopDOGD", T, w, "eta", eta_coeff, L_value, eps_manual, seed)
                    )
                    all_tasks.append(task)
    
    # 3. DCDOGD 任务生成
    if "DCDOGD" in ALGORITHMS_TO_RUN:
        print("生成DCDOGD任务...")
        if FEEDBACK_TYPE != 'full':
            # bandit场景：遍历所有eps_manual值
            for T, w, eta_coeff, eps_manual in itertools.product(
                    T_VALUES, w_VALUES, eta_coeff_VALUES, eps_manual_VALUES):
                
                # 对每个参数组合遍历五个种子
                for seed in SEEDS:
                    task = ExperimentTask(
                        model_type="DCDOGD",
                        params_func=create_DCDOGD_params,
                        params_args=(T, w, eta_coeff, eps_manual),
                        filename_args=("DCDOGD", T, w, "eta", eta_coeff, None, eps_manual, seed)
                    )
                    all_tasks.append(task)
        else:
            # full场景：不遍历eps_manual值，固定使用一个值
            eps_manual = 1.0  # 任意固定值
            # DCDOGD不需要L_value参数
            for T, w, eta_coeff in itertools.product(
                    T_VALUES, w_VALUES, eta_coeff_VALUES):
                
                # 对每个参数组合遍历五个种子
                for seed in SEEDS:
                    task = ExperimentTask(
                        model_type="DCDOGD",
                        params_func=create_DCDOGD_params,
                        params_args=(T, w, eta_coeff, eps_manual),
                        filename_args=("DCDOGD", T, w, "eta", eta_coeff, None, eps_manual, seed)
                    )
                    all_tasks.append(task)
    
    # 计算总任务数
    total_tasks = len(all_tasks)
    print(f"总任务数: {total_tasks}")
    
    # 根据起始和结束索引筛选任务
    if end_task is None:
        end_task = total_tasks
    
    # 转换为0-based索引
    start_idx = start_task - 1
    end_idx = end_task - 1
    
    # 确保索引在有效范围内
    start_idx = max(0, start_idx)
    end_idx = min(total_tasks - 1, end_idx)
    
    # 筛选任务
    filtered_tasks = all_tasks[start_idx:end_idx + 1]
    num_filtered_tasks = len(filtered_tasks)
    
    print(f"\n筛选任务范围: 任务 {start_task} 到 {end_task}")
    print(f"实际执行任务: 任务 {start_idx + 1} 到 {end_idx + 1} ({num_filtered_tasks} 个任务)")
    
    # 执行筛选后的任务
    completed_tasks = 0
    
    for task_idx, task in enumerate(filtered_tasks):
        # 设置随机种子
        seed = task.filename_args[-1]  # 最后一个参数是seed
        np.random.seed(seed)
        
        # 生成参数
        params = task.params_func(*task.params_args)
        
        # 生成文件名
        filename_base = generate_filename(*task.filename_args)
        
        # 检查文件是否已存在
        sr_file = f"{filename_base}_SR.npy"
        bits_file = f"{filename_base}_Bits.npy"
        
        if os.path.exists(sr_file) and os.path.exists(bits_file):
            completed_tasks += 1
            print(f"跳过已存在的实验: {params.label}_seed{seed} ({task_idx + 1}/{num_filtered_tasks}) [全局: {start_idx + task_idx + 1}/{total_tasks}]")
            continue
        
        try:
            # 创建模型实例
            if task.model_type == "CDFTGL":
                model = CDFTGL(params, do_prints=True)
            elif task.model_type == "TopDOGD":
                model = TopDOGD(params, do_prints=True)
            elif task.model_type == "DCDOGD":
                model = DCDOGD(params, do_prints=True)
            else:
                raise ValueError(f"未知的模型类型: {task.model_type}")
            
            # 运行优化
            print(f"运行实验: {params.label}_seed{seed} ({task_idx + 1}/{num_filtered_tasks}) [全局: {start_idx + task_idx + 1}/{total_tasks}]")
            model.optimize(A, y, A_test, y_test, test_num, rounds=1)
            
            # 保存结果
            np.save(sr_file, model.avg_m_regret / np.arange(1, len(model.avg_m_regret)+1))
            np.save(bits_file, model.avg_trans_bit)
            
        except Exception as e:
            print(f"实验 {params.label}_seed{seed} 失败: {str(e)} ({task_idx + 1}/{num_filtered_tasks}) [全局: {start_idx + task_idx + 1}/{total_tasks}]")
        
        completed_tasks += 1
    
    # 计算并显示总耗时
    total_time = (time.time() - start_time) / 3600
    print(f"\n网格搜索完成! 总耗时: {total_time:.2f} 小时")
    print(f"执行任务: {completed_tasks}/{num_filtered_tasks}")
    print(f"总任务数: {total_tasks}")


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='运行网格搜索，支持任务分片')
    parser.add_argument('--start', type=int, default=1, help='起始任务索引（从1开始）')
    parser.add_argument('--end', type=int, default=None, help='结束任务索引（包括该索引）')
    args = parser.parse_args()
    
    # 运行网格搜索，传入任务索引范围
    run_grid_search(args.start, args.end)

if __name__ == "__main__":
    main()
