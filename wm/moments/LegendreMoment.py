import math

from ..Image import Image
from .Moment2D import Moment2D


class LegendreMoment(Moment2D):
    
    def __init__(self, N, n_max=None, **kwargs):
        super().__init__(N, n_max, **kwargs)
        self.dtype = np.float
        self.bound = (-1, 1)
    
    def __call__(self, f_o, n=None, m=None):
        if n is not None and m is not None:
            L = 0
            dx = 2 / self.n_max
            dy = 2 / self.m_max
            dxy = dx * dy
            L_norm = (2*n+1) * (2*m+1) / 4 * dxy
            for x in np.linspace(*self.bound, self.n_max):
                for y in np.linspace(*self.bound, self.m_max):
                    L += self.P(n, x) * self.P(m, y) * f_o.at_circle(x, y)
            return L_norm * L
        else:
            L = self.momentum_mx((self.N,self.M))
            for n in range(self.N):
                for m in range(self.M):
                    L[n][m] = self(f_o, n, m)
            return L
    
    def P(self, n, x):
        P_norm = 1/2**n
        Px_n = sum(
            (math.comb(n,k)**2) * ((x-1)**(n-k)) * ((x+1)**k)
            for k in range(0, n+1)
        )
        return P_norm * Px_n
    
    def reconstruct(self, L, width, height=None):
        height = height or width
        colored = False
        I = Image(np.zeros((width,height), dtype=np.uint8), colored=colored)
        for u in range(width):
            for v in range(height):
                l = 0
                x, y = I.pos_to_unit(u, v)
                for n in range(self.N):
                    for m in range(self.M):
                        l += L[n][m] * self.P(n, x) * self.P(m, y)
                I.im[u][v] = l
        #print(Iim)
        return Image(I.im, colored=colored)
    
    def encode(self, f_o, w=None, pos=None):
        raise NotImplementedError
    
    def decode(self, f_o, pos=None):
        raise NotImplementedError
    