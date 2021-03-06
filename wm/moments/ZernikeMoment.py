import numpy as np

from ..Image import Image
from ._RadialMoment import _RadialMoment

class ZernikeMoment(_RadialMoment):
    
    name = 'ZM'
    
    def __init__(self, n_max, N=None, **kwargs):
        super().__init__(n_max, N or n_max, **kwargs)
        self.qs = kwargs.get('qs', 1.05)
        self.encode_dir = kwargs.get('encode_dir', 'row')
        self.name = 'ZM'
        #
        self.R_table = {}
    
    def __call__(self, f_o, n=None, m=None, pos_nm=None, selective=False, imgrid=None, verbose=False):
        
        if imgrid is None:
            imgrid = self.imgrid(f_o)
        
        # single momentum
        if n is not None and m is not None:
            if not self._correct_nm(n, m):
                return 0
            A_nm = 0
            dxdy = 4 / (self.N * self.M)
            if self.Vmx is not None:
                A_nm += np.sum(imgrid * np.conjugate(self.Vmx[n,m,:,:])) * dxdy
            else:
                for u in range(self.N):
                    for v in range(self.M):
                        if imgrid[u,v] == 0: continue
                        r, fi = self.polar_r_fi[u,v]
                        A_nm += imgrid[u,v] * np.conjugate(self.V(n, m, r, fi)) * dxdy
            return ((n+1)/np.pi) * A_nm
        
        # momentum list
        elif selective:
            A = []
            imgrid = self.imgrid(f_o)
            for k in range(len(pos_nm)):
                n, m = pos_nm[k]
                if verbose: print(''.join([str(k),'/',str(len(pos_nm))])+' '*32, end='\r')
                A.append( (n,m,self(f_o, n, m, imgrid=imgrid)) )
            return A
        
        # momentum matrix
        else:
            A = self.get_momentum_mx()
            for n in range(self.n_max+1):
                if verbose: print(''.join([str(n),'/',str(self.n_max)])+' '*32, end='\r')
                for m in range(self.m_max+1):
                    A[n,m] = self(f_o, n, m, imgrid=imgrid)
            return A
    
    def R(self, n, m, r, kintner=True):
        if not self._correct_nm(n, m):
            return 0
        # Kintner's gets unstable above n,m>34
        if kintner:
            # cache
            if r not in self.R_table:
                self.R_table[r] = self.R_kintner(r)
            return self.R_table[r][n,m]
        # slow computation
        res = 0.0
        m_abs = int(abs(m))
        for s in range((n-m_abs)//2+1):
            d0 = np.math.factorial(s)
            d1 = np.math.factorial((n+m_abs)//2-s)
            d2 = np.math.factorial((n-m_abs)//2-s)
            denom = d0 * d1 * d2
            res += ((-1.)**s) * np.math.factorial(n-s) * (r**(n-2*s)) / denom
        return res
    
    def R_kintner(self, r):
        R = np.zeros((self.n_max+1,self.m_max+1),dtype=np.float32)
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
        return self.R(n, m, r) * np.exp(1j*m*fi)
    
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
#
