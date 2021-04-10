import numpy as np

class Moment2D:
    
    def __init__(self, n_max, N=None, m_max=None, M=None, **kwargs):
        """
        N: max order
        M: max repetition
        """
        self.i = kwargs.get('i', 1j)
        self.j = kwargs.get('j', 0)
        self.k = kwargs.get('k', 0)
        self.pi = kwargs.get('pi', np.pi)
        self.abs = kwargs.get('abs', np.abs)
        self.sin = kwargs.get('exp', np.sin)
        self.cos = kwargs.get('cos', np.cos)
        self.exp = kwargs.get('exp', np.exp)
        self.sqrt = kwargs.get('sqrt', np.sqrt)
        self.conj = kwargs.get('conj', np.conjugate)
        self.factorial = kwargs.get('factorial', np.math.factorial)
        self.dtype = kwargs.get('dtype', complex)
        self.pxtype = kwargs.get('dtype', np.real)
        self.momentum_mx = kwargs.get('momenum_mx', lambda size: np.zeros(size, dtype=self.dtype))
        # N:sampling, M:sampling
        self.N = N or n_max
        self.M = M or self.N
        # n_max:order, m_max:repetition, w_space
        self.n_max = n_max
        self.m_max = m_max or n_max
        self.w_space = (self.n_max*self.m_max) // 2
    
    def __call__(self):
        raise NotImplementedError
    
    def polar_uv(self, u, v):
        """
        u: coord X
        v: coord Y
        """
        return ( 1*(u/self.N), 2*self.pi*(v/self.M) )
    