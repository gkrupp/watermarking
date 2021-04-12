import numpy as np

from ..Image import Image
from ._RadialMoment import _RadialMoment

class ZernikeMoment(_RadialMoment):
    
    def __init__(self, n_max, N=None, **kwargs):
        super().__init__(n_max, N or n_max, **kwargs)
        #
        self.RR = {}
    
    def __call__(self, f_o, n=None, m=None, verbose=False):
        if n is not None and m is not None:
            if not self._correct_nm(n, m):
                return 0
            A_nm = 0
            for u in range(self.N):
                for v in range(self.M):
                    r, fi = self.pos_to_polar(u, v)
                    if r > 1: continue
                    A_nm += f_o(r, fi) * self.h(n, m, r, fi)
            return ((n+1)/np.pi) * A_nm
        else:
            A = self.momentum_mx((self.n_max+1, self.m_max+1))
            for n in range(self.n_max+1):
                if verbose: print(''.join([str(n),'/',str(self.n_max)])+' '*32, end='\r')
                for m in range(self.m_max+1):
                    A[n,m] = self(f_o, n, m)
            return A
    
    def h(self, n, m, r, fi):
        du = 2 / self.N
        dv = 2 / self.M
        return np.conjugate(self.V(n, m, r, fi)) * du * dv
    
    def R(self, n, m, r):
        if not self._correct_nm(n, m):
            return 0
        res = 0
        m_abs = np.abs(m)
        for s in range((n-m_abs)//2+1):
            d0 = np.math.factorial(s)
            d1 = np.math.factorial((n+m_abs)//2-s)
            d2 = np.math.factorial((n-m_abs)//2-s)
            denom = d0 * d1 * d2
            res += ((-1.)**s) * np.math.factorial(n-s) * (r**(n-2*s)) / denom
        return res
    
    def R_r(self, r):
        R = np.zeros((self.n_max+1,self.m_max+1))
        R[0,0] = 1
        R[1,1] = r
        for n in range(2,self.n_max+1):
            h = n * (n-1) * (n-2)
            K2 = 2 * h
            R[n,n] = r**n
            R[n,n-2] = n*R[n,n] - (n-1)*R[n-2,n-2]
            for m in range(n-4, -1, -2):
                K1 = (n+m) * (n-m) * (n-2) / 2
                K3 = (-1)*m*m*(n-1) - h
                K4 = (-1) * n * (n+m-2) * (n-m-2) / 2
                r2 = r**2
                R[n,m] = ((K2*r2+K3)*R[n-2,m] + K4*R[n-4,m]) / K1
        return R
    
    def V(self, n, m, r, fi):
        if r not in self.RR:
            self.RR[r] = self.R_r(r)
        return self.RR[r][n, m] * np.exp(1j*m*fi)
    
    def _correct_nm(self, n, m):
        m_abs = np.abs(m)
        return (m_abs <= n) and ((n-m_abs) % 2 == 0)
    
    def moment_from_encode_pos(self, p):
        K = self.encode_K
        if self.encode_dir == 'row':
            s = -1
            for n in range(self.n_max):
                for m in range(n+1):
                    if self._correct_nm(n, m) and m % 4 != 0:
                        s += 1
                        if s == p:
                            return (n, m)
        elif self.encode_dir == 'diagonal':
            s = -1
            for d in range(2,2*min(self.n_max,self.m_max),2):
                for e in range(d//2, d+1):
                    (n, m) = (e, d-e)
                    if self._correct_nm(n, m) and m % 4 != 0:
                        s += 1
                        if s == p:
                            return (n, m)
        else:
            return None
    
    def moments_from_encode_poss(self, pos):
        pos_nm = []
        if isinstance(pos, int):
            pos = list(range(pos))
        for p in pos:
            if isinstance(p, tuple):
                pos_nm.append(p)
            else:
                pos_nm.append(self.moment_from_encode_pos(p))
        return pos_nm
    
    def clearCache(self):
        self.RR = {}
#
