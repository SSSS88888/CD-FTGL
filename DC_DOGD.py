from parameter_DCDOGD import DCDOGDParameters
from logistic_base import LogisticBase
from compressor import *
import networkx as nx
import numpy as np

def choose_on_shpere(d):
    vec = np.random.randn(d) 
    return vec/np.linalg.norm(vec)

class  DCDOGD(LogisticBase):
    def __init__(self, params: DCDOGDParameters, do_prints = False):
        p = params
        super().__init__(p)
        self.compressor = self.__choose_compressor(p.compressor_type,p.w,p.s)
        self.W= self.__create_matrix(p.agent_num, p.edge_num, p.topology)
        self.do_prints = do_prints
        
        # 这些属性将在optimize方法中初始化，用于测试集抽样
        self.A_test = None
        self.y_test = None
        self.test_num = None
        
    def __choose_compressor(self,type,w,s):            
        if type == 'no':
            self.params.w = 1
            return RandGossip(1)
        if type == 'rand':
            # 存储实际使用的w值到params中
            self.params.w = w
            return RandSparse(w)
        if type == 'top':
            # 存储实际使用的w值到params中
            self.params.w = w
            return TopSparse(w)
        if type == 'gossip':
            # 存储实际使用的w值到params中
            self.params.w = w
            return RandGossip(w)
        if type == 'quant':
            # 对于量化压缩器，w是用于计算s的参数
            return RandQuant(w)

    def __create_matrix(self,n,m,topology):
        if n==1:
            self.degree = np.ones(n)
            return np.eye(1)
        if topology == 'ring':
            if n == 2:
                W = np.array(( (0,1),(1,0)   ))
                self.degree = np.ones(n)
            else:
                value = 0.5
                W = np.zeros( (n, n) )
                np.fill_diagonal(W[1:,:], value )
                np.fill_diagonal(W[:, 1:], value )
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
                self.degree[i] = nx.degree(G,i)
                for j in iter(nx.neighbors(G,i)):
                    weight = 1/( max(nx.degree(G,i) , nx.degree(G,j) )  )
                    W[i][j] = weight
                    sum += weight
                W[i][i] = 1 - sum
        elif topology == 'full':
            W = np.ones((n,n)) / n
            self.degree = np.ones(n) * n
        return W

    def __calculate_config(self,d):
        par = self.params
        if par.compressor_type == 'quant':
            self.compressor.update_w(d)
        W = self.W
        
        if par.agent_num == 1:
            delta = 1
        else:
            delta = 1 - np.partition( np.abs(np.linalg.eigvalsh(W)), -2)[-2]
        beta = np.linalg.norm(np.eye(par.agent_num)- W,ord=2)
        assert delta > 1e-16 and delta <=1
        assert beta >= 0 and beta <=2
        
        w = self.compressor.w
        par.beta = beta
        par.delta = delta
        
        # 使用params中的w值确保一致性，而不是直接使用compressor的w
        # 这样在不同压缩器类型中都能使用正确的w值
        w = self.params.w

        if par.tune == False or par.gamma == None:
            # 确保w在有效范围内
            if w <= 0 or w > 1:
                print(f"警告: w值 {w} 超出有效范围(0,1]，将使用默认值1")
                w = 1
                self.params.w = 1
                self.compressor.w = 1
                
            # gamma = 3 * (delta ** 3) * (w ** 2) * (w + 1) / (
            #             48 * (delta ** 2 + 18 * delta * (beta ** 2) + 36 * (beta ** 2)) * (beta ** 2) * (w + 2) * (
            #                 1 - w) + 4 * (delta ** 2) * (beta + 1) * beta * (w + 2) * (1 - w) * w + 6 * (
            #                         delta ** 3) * w)
            gamma = (3 * delta**3 * w**3 * (w + 1)) / (
                        2 * delta**2 * w 
                        + 9 * beta**2 * (delta + 2) * (w - w**2) 
                        + 24 * beta**2 * (w + 2) 
                        + w * (w + 2) * (1 - w) * (beta**2 + 2 * beta) 
                        + 12 * beta**2 * (w + 2) * (1 - w)
                    )

            par.gamma = gamma
            print("gamma of DC-DOGD:",gamma)
        else:
            gamma = par.gamma

        H = 4 * 3 ** 0.5 * (par.agent_num ** 0.5 + 2 * 3 **0.5 / (gamma * delta) + 1 ) * (1 + 1/(gamma * delta ) + 1/ w)
        (R,D,C,T,L,r,alpha) = (par.R, par.D, par.C, par.T,par.L,par.r,par.alpha)
        print("H:",H)
        if alpha>0:
            # c = 16 / (3 * gamma * delta)
            c = 16 / (3 * delta)
        else:
            # c = 8 / (3 * gamma * delta)
            c = 8 / (3 * delta)

        if par.c == None:
            par.c = c
        else:
            c=par.c

        if par.feedback_type == 'full':
            eps = 0
            if par.tune == True:
                if alpha > 0:
                    eta = par.eta_coeff * 1/alpha
                else:
                    eta = par.eta_coeff * D/L
            else:
                if alpha > 0:
                    eta = 1/alpha
                else:
                    eta = D/L
            
        if par.feedback_type == 'one-bandit':
            if par.tune == True:
                if alpha > 0:
                    # 按原理论值计算无法实现，因此使用系数调整
                    # eps = par.eps_coeff * ( (H*d ** 2 * C **2 * np.log(T+c) )/( (L+C/r) *alpha * T ) )**(1/3)
                    # eps = par.eps_coeff * d ** (2/3) * par.T ** (-1/3) * (math.log(par.T)) ** (1/3)
                    eps = par.eps_manual
                    eta = par.eta_coeff * 1/alpha
                else:
                     # 按原理论值计算无法实现，因此使用系数调整
                    # eps = par.eps_coeff * ((1+4*H) *R*d*C / (2 * ( L + C/r)))**0.5 * (T+c) ** 0.25 / T ** 0.5
                    # eps = par.eps_coeff * d ** 0.25 * par.T ** (-0.25)
                    eps = par.eps_manual
                    eta = par.eta_coeff * (2 * R * eps) / (d * C)
            else:
                if alpha > 0:
                    eps = ( (H*d ** 2 * C **2 * np.log(T+c) )/( (L+C/r) *alpha * T ) )**(1/3)
                    eta = 1/alpha
                else:
                    eps = ((1+4*H) *R*d*C / (2 * ( L + C/r)))**0.5 * (T+c) ** 0.25 / T ** 0.5
                    eta = (2 * R * eps) / (d * C)
                   
        if par.feedback_type == 'two-bandit':
            if par.tune == True:
                if alpha > 0:
                    eps = par.eps_coeff * np.log(T) / T
                    eta = par.eta_coeff * 1/alpha
                else:
                    eps = par.eps_coeff * 1  / T** 0.5
                    eta = par.eta_coeff * (2 * R) / (d * L)
            else:
                if alpha > 0:
                    eps = np.log(T) / T
                    eta = 1/alpha
                else:
                    eps = 1  / T** 0.5
                    eta = (2 * R) / (d * L)

        par.eps = eps
        par.eta = eta

    def stepsize(self,t):
        p = self.params
        if p.alpha > 0:
            return p.eta / (t+p.c)
        else:
            return p.eta / (t+p.c) ** 0.5

    def grad_oracle(self,x,A,y):
        p = self.params
        d = A.shape[1]
        if p.feedback_type == 'full':
            g = self.grad(x,A,y)
            x_now = x
        if p.feedback_type == 'one-bandit':
            u = choose_on_shpere(d)  
            g = d / p.eps * self.loss(x+p.eps * u,A,y) * u
            x_now = x+p.eps * u
        if p.feedback_type == 'two-bandit':
            u = choose_on_shpere(d)
            g =  d / (2 * p.eps) * ( self.loss(x+ p.eps * u,A,y) - self.loss(x- p.eps *u,A,y)  ) * u
            x_now =  [x+p.eps * u, x-p.eps * u]
        assert not np.isnan(g).any()
        return x_now,g

    def print_params(self):
        p = self.params
        print('DC-DOGD optimization start.')
        print("------------核心参数-------------")
        print(" | 参数名            | 值                    |")
        print("-------------------------------------------")
        print(" | 网络拓扑          | {:>20} |".format(p.topology))  # 改为右对齐
        print(" | 智能体数量        | {:>20d} |".format(p.agent_num))  # 改为右对齐（整数）
        print(" | 总通信轮数        | {:>20d} |".format(p.T))  # 改为右对齐（整数）
        print(" | 反馈类型          | {:>20} |".format(p.feedback_type))  # 改为右对齐
        print(" | 压缩器类型        | {:>20} |".format(p.compressor_type))  # 改为右对齐
        print(" | gamma             | {:>20.5f} |".format(p.gamma))  # 保持右对齐（浮点）
        print(" | c                 | {:>20.5f} |".format(p.c))  # 保持右对齐（浮点）
        print(" | eps               | {:>20.5f} |".format(p.eps))  # 保持右对齐（浮点）
        print(" | eta               | {:>20.5f} |".format(p.eta))  # 保持右对齐（浮点）
        print(" | delta             | {:>20.5f} |".format(p.delta))  # 保持右对齐（浮点）
        print(" | beta              | {:>20.5f} |".format(p.beta))  # 保持右对齐（浮点）
        print("-------------------------------------------")

    def optimize(self, A, y, A_test=None, y_test=None, test_num=0, rounds=1):
        """
        调用 train() 进行 CD-FTGL 训练。总通信轮数为 p.T，数据分配逻辑在 train() 内使用循环指针实现。

        参数:
            A: np.ndarray, shape=(num, d)  —— 全体特征矩阵
            y: np.ndarray, shape=(num,) or (num,1)  —— 全体标签
            A_test: np.ndarray, shape=(test_num, d)  —— 外部采样的测试集特征矩阵
            y_test: np.ndarray, shape=(test_num,) or (test_num,1)  —— 外部采样的测试集标签
            test_num: int  —— 测试集样本数量
            rounds: int  —— 重复训练轮数，用于平均结果

        假设 self.params 中包含：
            batch_size, agent_num, distribution_type, eps, r, D, gamma, T（总通信轮数）等。
        """
        p = self.params
        bs = p.batch_size
        n = p.agent_num
        num, d = A.shape
        self.__calculate_config(d)
        
        # 使用外部传入的测试集
        self.A_test = A_test
        self.y_test = y_test
        self.test_num = test_num

        if self.do_prints:
            self.print_params()

        # 分配平均结果的容器
        T = p.T  # 总通信轮数
        self.avg_trans_bit = np.zeros(T, dtype=np.float64)
        self.avg_m_regret = np.zeros(T, dtype=np.float64)

        # 多轮训练循环
        for _ in range(rounds):
            self.compressor.total_bit = 0
            m_regret, trans_bits = self.train(A, y)
            self.avg_m_regret += m_regret
            self.avg_trans_bit += trans_bits

        self.avg_m_regret /= rounds
        self.avg_trans_bit /= rounds
        self.is_complete = True


    def train(self, A, y):
        """
        在内部根据循环指针分配数据并执行 CD-FTGL 训练，总通信轮数为 p.T。

        参数:
            A: np.ndarray, shape=(num, d)  —— 全体特征矩阵
            y: np.ndarray, shape=(num,) or (num,1)  —— 全体标签

        返回:
            m_regret: np.ndarray, shape=(p.T,)  —— 每轮通信结束时的累计最大遗憾
            trans_bit_history: np.ndarray, shape=(p.T,)  —— 每轮通信结束时的累计通信比特数
        """
        p = self.params
        n = p.agent_num
        bs = p.batch_size
        num, d = A.shape
        T = p.T
        # 获取分配类型，默认为'random'
        distribution_type = getattr(p, 'distribution_type', 'random')
        
        # 数据划分：每个 agent 的批次索引（循环使用）
        if distribution_type == 'label':
            # 按标签分配数据：每个节点只包含特定标签的数据
            # 首先获取所有唯一标签
            unique_labels = np.unique(y)
            label_count = len(unique_labels)
            
            # 确保标签数量不超过节点数量
            assert label_count <= n, f"标签数量({label_count})超过节点数量({n})"
            
            # 为每个标签分配多个节点，实现更均匀的数据分布
            # 计算每个标签应分配的节点数
            nodes_per_label = []
            remaining_nodes = n
            for i in range(label_count):
                # 最后一个标签分配剩余所有节点
                if i == label_count - 1:
                    nodes_per_label.append(remaining_nodes)
                else:
                    # 平均分配节点
                    nodes = remaining_nodes // (label_count - i)
                    nodes_per_label.append(max(1, nodes))  # 确保每个标签至少有一个节点
                    remaining_nodes -= nodes
            
            # 创建标签到节点的映射
            label_to_agents = {}
            current_node = 0
            
            # 打印节点和标签的映射关系
            print("DC-DOGD：节点与标签映射关系:")
            for i, label in enumerate(unique_labels):
                num_nodes = nodes_per_label[i]
                label_to_agents[label] = list(range(current_node, current_node + num_nodes))
                # 打印当前标签对应的节点
                print(f"DC-DOGD：标签 {label} 分配给节点: {label_to_agents[label]}")
                current_node += num_nodes
            
            # 为每个节点收集其对应的样本索引
            agent_indices = [[] for _ in range(n)]
            for idx, label in enumerate(y):
                if label in label_to_agents:
                    # 获取该标签对应的所有节点
                    agents = label_to_agents[label]
                    # 均匀分配样本到该标签的所有节点
                    agent_idx = agents[idx % len(agents)]
                    agent_indices[agent_idx].append(idx)
                else:
                    # 这种情况理论上不应该发生，因为unique_labels已经包含了所有标签
                    agent_indices[0].append(idx)  # 默认添加到第一个节点
            
            # 打乱每个节点的数据
            # np.random.seed(42)  # 注释掉，由外部脚本控制随机数种子
            for i in range(n):
                np.random.shuffle(agent_indices[i])
            
            # 计算每个节点的数据epoch数（使用最大的数据量来确定）
            max_samples = max(len(indices) for indices in agent_indices)
            data_epoch = max(1, max_samples // bs)
            
            # 为每个节点创建批次索引
            indice_data = np.zeros((n, data_epoch, bs), dtype='int64')
            for i in range(n):
                node_indices = agent_indices[i]
                node_len = len(node_indices)
                for e in range(data_epoch):
                    # 循环使用该节点的数据
                    start = (e * bs) % node_len
                    end = start + bs
                    if end > node_len:
                        # 如果不够一个batch，循环到开头补充
                        batch_indices = node_indices[start:] + node_indices[:end - node_len]
                    else:
                        batch_indices = node_indices[start:end]
                    indice_data[i, e] = batch_indices
        else:  # distribution_type == 'random' 或其他值
            # 随机分配数据（保持原有逻辑）
            data_epoch = max(1, num // (n * bs))
            # np.random.seed(42)  # 注释掉，由外部脚本控制随机数种子
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

        # 4. 初始化模型参数和状态
        self.x = np.zeros((d, n), dtype=np.float64)     # x[:, i]
        self.xh = np.zeros((d, n), dtype=np.float64)    # 累计压缩残差 x̂[:, i]
        self.s = np.zeros((d, n), dtype=np.float64)     # 共识状态 s[:, i]
        self.q = np.zeros((d, n), dtype=np.float64)     # 本地压缩向量 q[:, i]
        self.g = np.zeros((d, n), dtype=np.float64)     # 梯度 g[:, i]

        # 用于记录每轮 t 的 loss (shape=(T,n)) 和累计通信比特 (shape=(T,))
        loss_history = np.zeros((T, n), dtype=np.float64)
        trans_bit_history = np.zeros(T, dtype=np.int64)

        self.offline_order = []  # 记录训练过程中所有被抽取样本的顺序

        # 5. 主循环：t = 0, 1, ..., T-1
        for t in range(T):
            # 对每个 agent i：取一个批次，投影 → 压缩 → 梯度 → 记录 loss → 更新 ptr
            for i in range(n):
                e = ptr[i]
                batch_idx = indice_data[i, e]    # 该 agent 的 bs 个索引
                X_i = A[batch_idx, :]           # shape = (bs, d) 自身批次特征
                y_i = y[batch_idx]              # shape = (bs,) 自身批次标签

                # 5.2.1 投影
                if p.feedback_type == 'full':
                    self.x[:, i] = self.proj(self.x[:, i], p.D / 2)
                else:
                    self.x[:, i] = self.proj(self.x[:, i], (1 - p.eps / p.r) * p.r)

                # 5.2.2 压缩：q_i = Compress(x_i - x̂_i)
                x_i = self.x[:, i]
                xh_i = self.xh[:, i]
                self.q[:, i] = self.compressor.compress(x_i - xh_i, self.degree[i])
                
                # 5.2.3 梯度计算
                # print("x_i: ", x_i)
                x_i_now, grad_i = self.grad_oracle(x_i, X_i, y_i)
                self.g[:, i] = grad_i
                
                # 5.2.4 记录 loss_history[t, i]
                # 优化：使用抽样的测试集进行评估，大幅减少计算量
                # 同时保持评估的正确性，确保gamma接近0时能正确反映节点间无通信的影响
                if p.feedback_type == 'two-bandit':
                    x1 = x_i_now[0]
                    x2 = x_i_now[1]
                    # 在抽样测试集上计算损失
                    loss_history[t, i] = (self.loss(x1, self.A_test, self.y_test) + self.loss(x2, self.A_test, self.y_test)) / 2
                else:
                    # 在抽样测试集上计算损失
                    loss_history[t, i] = self.loss(x_i_now, self.A_test, self.y_test)
                # 归一化：除以测试集样本数
                loss_history[t, i] = loss_history[t, i] / self.test_num
                # 5.2.5 更新 ptr[i]；若到达 data_epoch，则循环回 0
                ptr[i] += 1
                if ptr[i] >= data_epoch:
                    ptr[i] = 0
            
            # 5.3 更新 x̂, s, x
            # 5.3.1 x̂ += q
            self.xh += self.q

            # 5.3.2 s += q @ W
            self.s += self.q @ self.W

            # 5.3.3 计算步长 η_t
            eta_t = self.stepsize(t + 1)

            # 5.3.4 x += γ (s - x̂) - η_t · g
            self.x += (p.gamma * (self.s - self.xh) - eta_t * self.g)

            # 5.4 记录累计通信比特
            trans_bit_history[t] = self.compressor.total_bit

        # 6. 训练结束后，用 loss_history 计算 m_regret
        m_regret = self.max_regret(loss_history)
        return m_regret, trans_bit_history



    def max_regret(self,loss_history):
        p = self.params  # 添加参数对象引用
        # assert self.is_complete
        sum = np.zeros(loss_history.shape)
        sum[0] = loss_history[0]
        for i in range(1, p.T):
            sum[i] = sum[i-1] + loss_history[i]
        max = np.max(sum,axis=1)
        return max

    def proj(self,x,r):
        if  np.linalg.norm(x) > r:
            return x / np.linalg.norm(x) * r
        else:
            return x