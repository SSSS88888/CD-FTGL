import numpy as np
import math

class TopDOGDParameters:
    def __init__(
        self, 
        alpha, 
        compressor_type, 
        w, 
        s, 
        topology, 
        agent_num, 
        edge_num, 
        batch_size,
        label, 
        distribution_type, 
        G, 
        R, 
        r, 
        T, 
        eta_coeff,
        eps_coeff,
        h=None,
        gamma=None,
        eps=None,
        # 新增参数
        manual_params=True,  # 手动设置开关（默认自动计算）
        eta_manual=None,        # 手动设置的h值
        L1_manual=None,        # 手动设置的L1值
        L2_manual=None,        # 手动设置的L2值
        eps_manual=None,      # 手动设置的eps值
        feedback_type=None,  # 新增参数（full/one-bandit）
        kwargs=None,
    ):
        
        # 基本参数验证和设置
        assert alpha >= 0
        self.alpha = alpha                   # 强凸系数
        
        assert compressor_type in ['no', 'rand', 'top', 'gossip', 'quant']
        if compressor_type == 'quant':
            assert s is not None
        else:
            if compressor_type != 'no':
                assert w is not None
        self.compressor_type = compressor_type
        self.s = s
        self.w = w
        
        assert topology in ['ring', 'random', 'full']
        assert agent_num > 0
        if topology == 'random':
            assert edge_num > 0
            assert isinstance(edge_num, int)
        
        self.topology = topology
        self.agent_num = int(agent_num)
        self.edge_num = edge_num
        
        assert batch_size > 0
        assert isinstance(batch_size, int)
        self.batch_size = batch_size
        
        assert distribution_type in ['random', 'label']
        self.distribution_type = distribution_type
        
        # Top-DOGD特定参数
        self.eta_coeff = eta_coeff                  # h系数 
        self.eps_coeff = eps_coeff              # eps系数
        self.eta = None                           # 正则化参数
        self.gamma = gamma                   # gamma参数
        self.eps = None
        self.eps_manual = None                        
        self.L = None                           
        self.L1 = None                           
        self.L2 = None                           
        self.feedback_type = feedback_type      # 反馈类型
        
        # 网络参数
        self.delta = None                    # 网络参数delta
        self.beta = None                     # 网络参数beta
        
        # 其他参数
        self.G = G                           # 梯度界
        self.R = R                           # 正则化参数R（用于计算h）
        self.T = T                           # 总轮数（可选，用于计算h）
        
        # 可视化参数
        if kwargs is None:
            self.kwargs = {'c': '#1772b2', 'ls': '-', 'linewidth': 5}
        else:
            self.kwargs = kwargs
            
        if label is not None:
            self.label = label
            
        # 运行时参数
        self.epoch = None
        self.manual_params = manual_params  # 新增：手动设置开关
        self.eta_manual = eta_manual            # 新增：手动eta值
        self.L1_manual = L1_manual            # 新增：手动L1值
        self.L2_manual = L2_manual            # 新增：手动L2值
        self.eps_manual = eps_manual          # 新增：手动eps值
        self.feedback_type = feedback_type  # 新增属性
    
    def calculate_params(self, W, compressor_w, d):
        """计算Top-DOGD参数"""
        n = self.agent_num
        
        # 计算delta和beta
        if n == 1:
            self.delta = 1
        else:
            self.delta = 1 - np.partition(np.abs(np.linalg.eigvalsh(W)), -2)[-2]
        
        self.beta = np.linalg.norm(np.eye(n) - W, ord=2)
        
        # 确保delta和beta在有效范围内
        assert self.delta > 1e-16 and self.delta <= 1
        assert self.beta >= 0 and self.beta <= 2
        
        # 如果未指定gamma，则根据公式计算
        if self.gamma is None:
            delta = self.delta
            beta = self.beta
            w = compressor_w

            denominator = 2 * delta * beta ** 2 + 4 * beta ** 2 + (2 - w) * (beta ** 2 + 2 * beta) * delta + delta**2
            self.gamma = delta * w / denominator
        
        # 如果未指定L1，则根据公式计算
        if self.L1 is None:
            delta = self.delta
            beta = self.beta
            w = compressor_w
            denominator = 2 * delta * beta ** 2 + 4 * beta ** 2 + (2 - w) * (beta ** 2 + 2 * beta) * delta + delta**2
            gamma = delta * w / denominator
            self.L1 = math.ceil(28 * math.log(n) / (gamma * delta))

        # 如果未指定L2，则根据公式计算
        if self.L2 is None:
            delta = self.delta
            beta = self.beta
            w = compressor_w
            self.L2 = math.ceil(math.log(8 * n)  / w)
        
        self.L = self.L1 + self.L2

        if self.feedback_type == 'full':
            # 如果未指定eta，且提供了G、R和T，则根据公式计算eta
            if self.eta is None and self.G is not None and self.R is not None:
                if self.alpha == 0:
                    self.eta = self.eta_coeff * 2 * self.R  / (self.G * math.sqrt(self.L * self.T))
                elif self.alpha > 0:
                    self.eta = self.eta_coeff * 1 /(self.alpha * (self.L + 8))
            self.eps = 0
        elif self.feedback_type == 'one-bandit':
            if self.eps is None and self.eps_manual is not None:
                self.eps = self.eps_manual
            # if self.eps is None:
            #     if self.alpha == 0:
            #         self.eps = self.eps_coeff * math.sqrt(d) * self.L **(0.25) * self.T ** (-0.25)
            #     elif self.alpha > 0:
            #         self.eps = self.eps_coeff * d ** (2/3) * self.L ** (1/3) * (math.log(self.T+8) / self.T) ** (1/3)
            if self.eta is None and self.G is not None and self.R is not None:
                if self.alpha == 0:
                    self.eta = self.eta_coeff * self.R * self.eps / (d*math.sqrt(self.L * self.T))
                elif self.alpha > 0:
                    self.eta = self.eta_coeff * 1 /self.alpha * (self.L + 8)
            
        else:
            print("Not support feedback_type: {}".format(self.feedback_type))
        return self.delta, self.beta, self.gamma, self.L1, self.L2, self.eta, self.eps
    
    def print_params(self):
        """打印参数信息"""
        print('Top-DOGD optimization parameters:')
        print('Graph topology: {}'.format(self.topology))
        print('Agent number: {}'.format(self.agent_num))
        if self.epoch is not None:
            print('Learning rounds: {}'.format(self.epoch))
        print("------------Parameters-------------")
        print(" |alpha   |{:>20.5f}|".format(self.alpha))  # alpha作为强凸系数
        print(" |eta       |{:>20.5f}|".format(self.eta if self.eta is not None else 0))
        print(" |gamma   |{:>20.5f}|".format(self.gamma if self.gamma is not None else 0))
        print(" |L       |{:>20.5f}|".format(self.L if self.L is not None else 0))
        if self.delta is not None:
            print(" |delta   |{:>20.5f}|".format(self.delta))
        if self.beta is not None:
            print(" |beta    |{:>20.5f}|".format(self.beta))
        if self.G is not None:
            print(" |G       |{:>20.5f}|".format(self.G))
        if self.R is not None:
            print(" |R       |{:>20.5f}|".format(self.R))
        if self.eps is not None:
            print(" |eps       |{:>20.5f}|".format(self.eps))
        print("-----------------------------------")