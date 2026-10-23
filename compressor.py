MBIT = 64

from cmath import sqrt
import numpy as np
class Compressor:
    def __init__(self,w):
        self.w = w
        self.total_bit = 0

class RandSparse(Compressor):
    def compress(self, vec, times):
        d = vec.shape[0]
        k = int(self.w * d)
        
        # 新增校验：禁止零元素压缩
        if k == 0:
            raise ValueError(
                "Compression would discard 100% of elements. "
                "Increase compression rate `self.w` (current value: {})".format(self.w) +
                " Must set w ≥ {min_w:.4f} to preserve at least 1 element.".format(min_w=1/d)
            )
            
        ind = np.random.randint(0, d, k)
        a = np.zeros(d)
        a[ind] = 1
        self.total_bit += times * k * (MBIT + np.ceil(np.log2(d)))        
        return vec * a

class TopSparse(Compressor):
    def compress(self, vec, times):
        d = vec.shape[0]
        k = int(self.w * d)
        
        # 新增校验：禁止零元素压缩
        if k == 0:
            raise ValueError(
                "Compression would discard 100% of elements. "
                "Increase compression rate `self.w` (current value: {})".format(self.w) +
                " Must set w ≥ {min_w:.4f} to preserve at least 1 element.".format(min_w=1/d)
            )
                
        ind = np.argpartition(np.abs(vec), -k)[-k:]
        a = np.zeros(d)
        a[ind] = 1
        self.total_bit += times * k * (MBIT + np.ceil(np.log2(d)))        
        return vec * a

# class RandSparse(Compressor):
#     def compress(self , vec , times):
#         d = vec.shape[0]
#         k = int( self.w * d )
#         ind = np.random.randint(0,d,k)

#         a = np.zeros(d)
#         a[ind] = 1
#         '''
#         这里原有代码是错的：self.total_bit += times * (MBIT * k + np.log2(d))
#         应该log2(d)也应该乘上k
#         '''
#         self.total_bit += times * k * (MBIT + np.ceil(np.log2(d)))        
#         return vec*a

# class TopSparse(Compressor):
#     def compress(self , vec , times):
#         d = vec.shape[0]
#         k = int( self.w * d )
#         ind = np.argpartition(np.abs(vec), -k)[-k:]
#         a = np.zeros(d)
#         a[ind] = 1
#         '''
#         这里原有代码是错的：self.total_bit += times * (MBIT * k + np.log2(d))
#         应该log2(d)也应该乘上k
#         '''
#         self.total_bit += times * k * (MBIT + np.ceil(np.log2(d)))        
#         return vec*a

class RandGossip(Compressor):
    def compress(self,vec, times):
        d = vec.shape[0]
        p = np.random.random()
        if p <= self.w:
            self.total_bit += times * MBIT * d
            return vec
        else:
            a = np.zeros(d)
            return a

class RandQuant():
    # S_W_MAP mapping table
    S_W_MAP = {
        5: 0.05,
        6: 0.1,
        10: 0.5,
        float('inf'): 1.0
    }
    
    def __init__(self,w) -> None:
        self.w = w
        self.total_bit = 0
        # Calculate s based on w
        self.s = self._calculate_s_from_w()

    def _calculate_s_from_w(self):
        # Find the s value corresponding to the current w
        for s, target_w in self.S_W_MAP.items():
            if abs(target_w - self.w) < 1e-15:
                return s
        # If no exact match, raise an error
        raise ValueError(f"No matching s found for w = {self.w}. Available w values: {list(self.S_W_MAP.values())}")

    def update_w(self,d):
        # s is now calculated from self.w, so no need to update it here
        pass

    def compress(self,vec, times):
        d = vec.shape[0]
        s = self.s
        
        # When s is infinity, no compression but record full communication bits
        if s == float('inf'):
            # Calculate bits for sending full vector (each element MBIT bits, d elements)
            self.total_bit += times * MBIT * d
            return vec.copy()  # Return original vector copy
        
        # Original compression logic (only execute when s≠inf)
        sigma = 1 + min(d/s ** 2 , d ** 0.5/s)
        sgn = np.sign(vec)
        norm = np.linalg.norm(vec)
        if norm < 1e-15:
            return np.zeros(d)
        nor_vec = np.abs(vec) / norm
        xi = np.random.random(d)
        quant_vec = np.trunc( (nor_vec * self.s) + xi ) 
        self.total_bit += times * (MBIT + np.ceil(np.log2(self.s * 2 + 1)) * d )
        return norm * sgn * quant_vec /(self.s * sigma)