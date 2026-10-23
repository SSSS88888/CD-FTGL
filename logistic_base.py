import numpy as np
from parameter_CDFTGL import CDFTGLParameters
from scipy.special import expit
class LogisticBase:
    def __init__(self, params) -> None:  # 移除特定类型注解以提高兼容性
        #x列为一个agent的状态
        self.x = None 
        self.xh = None
        self.s = None
        self.g = None
        self.q = None
        self.loss_history = None
        self.params = params
        self.is_complete = False
    
    def loss(self, x, A, y):
        # 完全向量化实现，避免使用np.vectorize
        p = self.params
        num_sample = A.shape[0]
        
        # 使用向量化的logistic损失计算
        # np.logaddexp(0, x)等价于np.log(1+np.exp(x))，但更稳定和高效
        z = -y * (A @ x)
        # 对于大的z值，使用近似以避免数值问题
        z_clipped = np.clip(z, -30, 709)  # 709是np.exp不会溢出的最大输入
        loss = np.sum(np.logaddexp(0, z_clipped)) 
        
        # 添加正则化项
        if p.alpha > 0:
            loss += p.alpha * (num_sample / (p.batch_size)) / 2 * np.linalg.norm(x) ** 2 
        
        assert not np.isnan(loss)
        return loss
    
    def grad(self, x, A, y):
        # 完全向量化实现，避免使用np.vectorize
        p = self.params
        num_sample = A.shape[0]
        
        # 向量化计算sigmoid和梯度
        z = -y * (A @ x)
        # 使用scipy的expit进行向量化的sigmoid计算
        sig_z = expit(z)  # 这已经是向量化的
        
        # 向量化计算梯度
        y_reshaped = y.reshape(-1, 1)  # 为了广播
        sig_z_reshaped = sig_z.reshape(-1, 1)  # 为了广播
        
        # 一步计算所有样本的梯度贡献
        grad_contributions = -y_reshaped * A * sig_z_reshaped
        
        # 沿着样本维度求和
        grad = grad_contributions.sum(axis=0)
        
        # 添加正则化项
        if p.alpha > 0:
            # 注意：这里似乎有重复添加alpha*x的情况，保留原始逻辑但简化计算
            grad += p.alpha * x * (1 + num_sample / (p.batch_size))
        
        return grad

