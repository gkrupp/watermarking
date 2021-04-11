import numpy as np

class Moment2D:
    
    def __init__(self, n_max, N=None, m_max=None, M=None, **kwargs):
        """
        n_max: max order
        N: image height
        m_max: max repetition
        M: image width
        """
        self.dtype = kwargs.get('dtype', complex)
        self.pxtype = kwargs.get('dtype', np.real)
        self.momentum_mx = kwargs.get('momenum_mx', lambda size: np.zeros(size, dtype=self.dtype))
        # N:sampling, M:sampling
        self.N = N or n_max
        self.M = M or self.N
        # n_max:order, m_max:repetition
        self.n_max = n_max
        self.m_max = m_max or n_max
        self.w_space = None
    
    def __call__(self):
        raise NotImplementedError
    
    def polar_uv(self, u, v):
        """
        u: coord X
        v: coord Y
        """
        return (
            1 * (u / self.N),
            2 * np.pi * (v / self.M)
        )
    