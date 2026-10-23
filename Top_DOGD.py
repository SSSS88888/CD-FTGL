from parameter_TopDOGD import TopDOGDParameters
from logistic_base import LogisticBase
from compressor import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt  # 新增导入
from tqdm import tqdm  # 新增导入

def choose_on_shpere(d):
    vec = np.random.randn(d) 
    return vec/np.linalg.norm(vec)

class TopDOGD(LogisticBase):
    def __init__(self, params: TopDOGDParameters, do_prints=False):
        p = params
        super().__init__(p)
        self.compressor = self.__choose_compressor(p.compressor_type, p.w, p.s)
        self.W = self.__create_matrix(p.agent_num, p.edge_num, p.topology)
        self.w = p.w
        self.do_prints = do_prints
        
        # 初始化测试集相关属性为None，稍后在optimize方法中初始化
        self.A_test = None
        self.y_test = None
        self.test_num = None
        
    def __choose_compressor(self, type, w, s):            
        if type == 'no':
            self.params.w = 1
            return RandGossip(1)
        if type == 'rand':
            return RandSparse(w)
        if type == 'top':
            return TopSparse(w)
        if type == 'gossip':
            return RandGossip(w)
        if type == 'quant':
            return RandQuant(w)

    def __create_matrix(self, n, m, topology):
        # 创建邻接矩阵A
        if n == 1:
            self.degree = np.ones(n)
            return np.eye(1)
        if topology == 'ring':
            if n == 2:
                W = np.array(((0, 1), (1, 0)))
                self.degree = np.ones(n)
            else:
                value = 0.5
                W = np.zeros((n, n))
                np.fill_diagonal(W[1:, :], value)
                np.fill_diagonal(W[:, 1:], value)
                W[0, n - 1] = value
                W[n - 1, 0] = value
                self.degree = np.ones(n) * 2
        elif topology == 'random':
            if n == 50:
                # 生成随机正则图，每个节点的度数固定为 d=4
                # 总边数 m = (50 * 4) / 2 = 100
                d = 4 
                G = nx.random_regular_graph(d, n, seed=42)
            else:
                # 维持原有逻辑，针对小规模节点使用 gnm 随机图
                G = nx.gnm_random_graph(n, m, seed=42)
            # G = nx.gnm_random_graph(n, m, seed=42)
            # 确保生成的图是连通的，否则算法无法收敛
            if not nx.is_connected(G):
                raise ValueError("Generated graph is not connected. Try a larger m or a different seed.")
            # assert nx.is_connected(G)
            W = nx.to_numpy_array(G)
            self.degree = np.zeros(n)
            for i in nx.nodes(G):
                sum = 0
                self.degree[i] = nx.degree(G, i)
                for j in iter(nx.neighbors(G, i)):
                    weight = 1 / (max(nx.degree(G, i), nx.degree(G, j)))
                    W[i][j] = weight
                    sum += weight
                W[i][i] = 1 - sum
        elif topology == 'full':
            W = np.ones((n, n)) / n
            self.degree = np.ones(n) * n
        return W

    def optimize(self, A, y, A_test=None, y_test=None, test_num=0, rounds=1):
        p = self.params
        # bs = p.batch_size
        self.agent_num = p.agent_num
        num, d = A.shape  # num 是样本总数
        self.T = p.T
        self.batch_size = p.batch_size
        self.alpha = p.alpha
        
        # 使用外部传入的测试集
        self.A_test = A_test
        self.y_test = y_test
        self.test_num = test_num
        
        # 根据手动开关决定参数来源
        if p.manual_params:
            # 先执行自动计算以获取所有参数（包括可能被手动覆盖的h/L/gamma）
            if hasattr(p, 'calculate_params'):
                delta_auto, beta_auto, gamma_auto, L1_auto, L2_auto, eta_auto, eps_auto = p.calculate_params(self.W, p.w, d)
                self.delta = delta_auto  # 保留自动计算的delta
                self.beta = beta_auto    # 保留自动计算的beta
                self.eps = eps_auto      # 保留自动计算的eps
                # 对h/L/gamma使用手动值（如果提供），否则使用自动计算值
                self.eta = p.eta_manual if p.eta_manual is not None else eta_auto
                self.L1 = p.L1_manual if p.L1_manual is not None else L1_auto
                self.L2 = p.L2_manual if p.L2_manual is not None else L2_auto
                self.gamma = p.gamma if p.gamma is not None else gamma_auto
                
                # 参数安全检查，确保eta_coeff和eps_coeff不为None
                self.eta_coeff = getattr(p, 'eta_coeff', 1.0)
                self.eps_coeff = getattr(p, 'eps_coeff', 1.0)
            else:
                # 如果没有calculate_params方法但启用了manual_params，需要手动提供所有参数
                raise ValueError("当 manual_params=True 时，参数对象p必须实现calculate_params方法")
        else:
            # 自动计算所有参数
            if hasattr(p, 'calculate_params'):
                self.delta, self.beta, self.gamma, self.L1, self.L2, self.eta, self.eps = p.calculate_params(self.W, p.w, d)
                
                # 参数安全检查，确保eta_coeff和eps_coeff不为None
                self.eta_coeff = getattr(p, 'eta_coeff', 1.0)
                self.eps_coeff = getattr(p, 'eps_coeff', 1.0)

        self.L = self.L1 + self.L2
        # 根据定理设置总轮数 B = T/(L1+L2)
        
        self.B = self.T // self.L
        B = self.B  # 简化变量名
        p.epoch = B  # 重定义epoch为总轮数
        if self.do_prints:
            self.print_params()
        
        # 初始化时间记录数组
        self.time_history = []

        # 主循环对应算法中的z=1到B（带进度条）
        # for m in tqdm(range(1, B + 1), desc="Training Progress", total=B, disable=not self.do_prints):
        self.compressor.total_bit = 0
        
        # 第一次调用train方法获取结果长度，然后初始化avg数组
        m_regret, bits = self.train(A, y, B)
        
        # 根据实际返回的长度初始化结果记录数组
        result_length = len(m_regret)
        self.avg_m_regret = np.zeros(result_length)
        self.avg_trans_bit = np.zeros(result_length)
        
        # 累加第一次的结果
        self.avg_m_regret += m_regret
        self.avg_trans_bit += bits
        
        # 继续执行剩余的轮次
        for _ in range(1, rounds):
            self.compressor.total_bit = 0
            m_regret, bits = self.train(A, y, B)
            self.avg_m_regret += m_regret
            self.avg_trans_bit += bits
        
        # 计算平均值
        self.avg_m_regret /= rounds
        self.avg_trans_bit /= rounds
        self.is_complete = True
        
        # 打印最终的x值
        # print("Algorithm finished. Final x values:")
        # print(self.x)

    def train(self, A, y, B):
        """
        Top-DOGD 训练：修复通信记录逻辑，保证损失/比特长度严格一致
        """
        # 基本维度与批次划分（保留原逻辑，无修改）
        num, d = A.shape
        bs = self.batch_size
        n = self.agent_num
        L1, L2 = self.L1, self.L2
        L = L1 + L2
        eta =  self.eta

        if not hasattr(self, 'R'):
            self.R = np.inf

        # 获取分配类型，默认为'random'（保留原逻辑，无修改）
        distribution_type = getattr(self.params, 'distribution_type', 'random')
        
        # ========== 数据划分逻辑完全保留，无修改 ==========
        if distribution_type == 'label':
            unique_labels = np.unique(y)
            label_count = len(unique_labels)
            assert label_count <= n, f"标签数量({label_count})超过节点数量({n})"
            nodes_per_label = []
            remaining_nodes = n
            for i in range(label_count):
                if i == label_count - 1:
                    nodes_per_label.append(remaining_nodes)
                else:
                    nodes = remaining_nodes // (label_count - i)
                    nodes_per_label.append(max(1, nodes))
                    remaining_nodes -= nodes
            label_to_agents = {}
            current_node = 0
            print("Top-DOGD：节点与标签映射关系:")
            for i, label in enumerate(unique_labels):
                num_nodes = nodes_per_label[i]
                label_to_agents[label] = list(range(current_node, current_node + num_nodes))
                print(f"Top-DOGD：标签 {label} 分配给节点: {label_to_agents[label]}")
                current_node += num_nodes
            agent_indices = [[] for _ in range(n)]
            for idx, label in enumerate(y):
                if label in label_to_agents:
                    agents = label_to_agents[label]
                    agent_idx = agents[idx % len(agents)]
                    agent_indices[agent_idx].append(idx)
                else:
                    agent_indices[0].append(idx)
            for i in range(n):
                np.random.shuffle(agent_indices[i])
            max_samples = max(len(indices) for indices in agent_indices)
            data_epoch = max(1, max_samples // bs)
            indice_data = np.zeros((n, data_epoch, bs), dtype='int64')
            for i in range(n):
                node_indices = agent_indices[i]
                node_len = len(node_indices)
                for e in range(data_epoch):
                    start_idx = (e * bs) % node_len
                    end_idx = start_idx + bs
                    if end_idx > node_len:
                        batch_indices = node_indices[start_idx:] + node_indices[:end_idx - node_len]
                    else:
                        batch_indices = node_indices[start_idx:end_idx]
                    indice_data[i, e] = batch_indices
        else:
            data_epoch = max(1, num // (n * bs))
            shuffled = np.random.permutation(num)
            indice_data = np.zeros((n, data_epoch, bs), dtype='int64')
            start = 0
            for i in range(n):
                for e in range(data_epoch):
                    end = start + bs
                    if end > num:
                        start = 0
                        end = bs
                    indice_data[i, e] = shuffled[start:end]
                    start = end
        ptr = np.zeros(n, dtype=int)  # 每个 agent 的批次指针

        # ========== ✅ 修改1：废弃固定数组，改用动态列表存储通信数据（核心） ==========
        self.x = np.zeros((d, n), dtype=np.float64)
        self.x_hat = np.zeros((d, n), dtype=np.float64)
        self.y = np.zeros((d, n), dtype=np.float64)
        self.y_hat = np.zeros((d, n), dtype=np.float64)
        self.s = np.zeros((d, n), dtype=np.float64)
        self.z = np.zeros((d, n), dtype=np.float64)
        self.q = np.zeros((d, n), dtype=np.float64)
        self.q_r = np.zeros((d, n), dtype=np.float64)
        self.res = np.zeros((d, n), dtype=np.float64)
        comm_loss_list = []    # 仅存储【通信时】的节点损失 (通信次数, n)
        comm_bit_list = []     # 仅存储【通信时】的累计比特数 (通信次数,)

        # 主循环：块 m=1..B（保留原算法逻辑，仅修改记录部分）
        for m in range(1, B + 1):
            x_fixed = self.x.copy()
            z_temp = np.zeros((d, n), dtype=np.float64)

            if self.alpha > 0:
                eta = 1 / (self.alpha * (m * self.L + 8))
            if m >= 2:
                y_k = x_fixed - eta * self.z
                hat_y_k = self.x_hat.copy()
                s_k = self.s.copy()
            else:
                y_k = None
                hat_y_k = None
                s_k = None

            # 阶段一：L1 次本地步
            for k in range(L1):
                X_batch_i = [None] * n
                y_batch_i = [None] * n
                for i in range(n):
                    e_data = ptr[i]
                    batch_idx = indice_data[i, e_data]
                    X_b = A[batch_idx, :]
                    y_b = y[batch_idx]
                    X_batch_i[i] = X_b
                    y_batch_i[i] = y_b
                    x_i_now, grad_i = self.grad_oracle(x_fixed[:, i], X_b, y_b)
                    z_temp[:, i] += grad_i
                    ptr[i] += 1
                    if ptr[i] >= data_epoch:
                        ptr[i] = 0

                # ========== ✅ 修改2：阶段一通信块内，强绑定记录损失+比特 ==========
                if m >= 2:
                    diff = y_k - hat_y_k
                    for i in range(n):
                        self.q[:, i] = self.compressor.compress(diff[:, i], self.degree[i])
                    hat_y_kp1 = hat_y_k + self.q
                    s_kp1 = s_k + self.W.dot(self.q.T).T
                    y_kp1 = y_k + self.gamma * (s_kp1 - hat_y_kp1)
                    y_k, hat_y_k, s_k = y_kp1, hat_y_kp1, s_kp1
                    
                    # ✅ 通信发生 → 立即记录【累计比特】+【当前损失】，无过滤
                    comm_bit_list.append(self.compressor.total_bit)
                    current_loss = np.zeros(n, dtype=np.float64)
                    for i in range(n):
                        if self.A_test is not None and self.y_test is not None:
                            current_loss[i] = self.loss(x_i_now, self.A_test, self.y_test) / self.test_num
                        else:
                            current_loss[i] = self.loss(x_i_now, A, y) / num
                    comm_loss_list.append(current_loss)

            # 阶段一结束
            if m >= 2:
                y_end = y_k.copy()
                y_hat_end = hat_y_k.copy()
            else:
                y_end = x_fixed.copy()
                y_hat_end = self.y_hat.copy()

            # 阶段二：L2 次本地步
            if m >= 2:
                r_target = self.proj(y_end, self.R) - y_end
                r_acc = np.zeros((d, n), dtype=np.float64)
            else:
                r_target = None
                r_acc = np.zeros((d, n), dtype=np.float64)

            for k in range(L2):
                X_batch_i = [None] * n
                y_batch_i = [None] * n
                for i in range(n):
                    e_data = ptr[i]
                    batch_idx = indice_data[i, e_data]
                    X_b = A[batch_idx, :]
                    y_b_ = y[batch_idx]
                    X_batch_i[i] = X_b
                    y_batch_i[i] = y_b_
                    x_i_now, grad_i = self.grad_oracle(x_fixed[:, i], X_b, y_b_)
                    z_temp[:, i] += grad_i
                    ptr[i] += 1
                    if ptr[i] >= data_epoch:
                        ptr[i] = 0

                # ========== ✅ 修改3：阶段二通信块内，强绑定记录损失+比特 ==========
                if m >= 2:
                    diff_r = r_target - r_acc
                    for i in range(n):
                        self.q_r[:, i] = self.compressor.compress(diff_r[:, i], self.degree[i])
                    r_acc = r_acc + self.q_r
                    
                    # ✅ 通信发生 → 立即记录【累计比特】+【当前损失】，无过滤
                    comm_bit_list.append(self.compressor.total_bit)
                    current_loss = np.zeros(n, dtype=np.float64)
                    for i in range(n):
                        if self.A_test is not None and self.y_test is not None:
                            current_loss[i] = self.loss(x_i_now, self.A_test, self.y_test) / self.test_num
                        else:
                            current_loss[i] = self.loss(x_i_now, A, y) / num
                    comm_loss_list.append(current_loss)

            # 块末统一更新（保留原逻辑，无修改）
            self.z = z_temp.copy()
            if m >= 2:
                self.x = self.proj(y_end, self.R)
                self.x_hat = y_hat_end + r_acc
                self.y = y_end.copy()
                self.y_hat = y_hat_end.copy()
                self.s = s_k.copy()
            else:
                self.x = self.proj(y_end, self.R)

        # ========== ✅ 修改4：动态列表转数组，直接返回，废弃所有事后筛选逻辑 ==========
        comm_loss_array = np.array(comm_loss_list) if comm_loss_list else np.array([])
        comm_bit_array = np.array(comm_bit_list) if comm_bit_list else np.array([])
        m_regret = self.max_regret(comm_loss_array)
        
        # ✅ 强制校验长度一致性（可选，调试用）
        assert len(m_regret) == len(comm_bit_array), f"长度不一致！regret={len(m_regret)}, bit={len(comm_bit_array)}"
        return m_regret, comm_bit_array


    def grad_oracle(self, x, A, y):
        p = self.params
        d = A.shape[1]
        if p.feedback_type == 'full':
            g = self.grad(x,A,y)
            x_now = x
        if p.feedback_type == 'one-bandit':
            u = choose_on_shpere(d)  
            g = d / self.eps * self.loss(x+self.eps * u,A,y) * u
            x_now = x+self.eps * u
        assert not np.isnan(g).any()
        return x_now,g

    def max_regret(self, loss_history):
        """计算每个通信时间步的最大遗憾值（累计损失的最大值）
        ✅ 新输入：loss_history → (通信次数, n)，仅包含通信时的损失，无无效数据
        ✅ 输出长度 ≡ 通信次数 ≡ 比特数组长度
        """
        if len(loss_history) == 0:
            return np.array([])
        
        # 直接计算累计损失 + 节点维度最大值，一步到位
        sum_loss = np.cumsum(loss_history, axis=0)  # 按通信步累计损失 (通信次数, n)
        max_regret = np.max(sum_loss, axis=1)       # 取每个步的节点最大损失 (通信次数,)
        return max_regret

    def proj(self,x,R):
        if  np.linalg.norm(x) > R:
            return x / np.linalg.norm(x) * R
        else:
            return x    

    def print_params(self):
        p = self.params
        print('Top-DOGD optimization start.')
        print('Graph topology: {}'.format(p.topology))
        print('Agent number: {}, B: {}'.format(p.agent_num, self.B))
        print("------------Parameters-------------")
        print(" |delta   |{:>20.5f}|".format(self.delta))
        print(" |beta    |{:>20.5f}|".format(self.beta))
        print(" |w       |{:>20.5f}|".format(p.w))
        print(" |s       |{:>20.5f}|".format(p.s))
        print(" |eps     |{:>20.5f}|".format(self.eps))   
        print(" |eta     |{:>20.5f}|".format(self.eta))      
        print(" |gamma   |{:>20.5f}|".format(self.gamma))
        print(" |L1       |{:>20.5f}|".format(self.L1))
        print(" |L2       |{:>20.5f}|".format(self.L2))
        print("-----------------------------------")
