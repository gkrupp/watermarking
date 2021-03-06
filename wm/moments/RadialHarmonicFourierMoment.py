import numpy as np

from ..Image import Image
from ._RadialMoment import _RadialMoment


class RadialHarmonicFourierMoment(_RadialMoment):
    
    name = 'RHFM'
    
    def __init__(self, n_max, N=None, **kwargs):
        super().__init__(n_max, N or n_max, **kwargs)
        self.qs = kwargs.get('qs', 0.2)
        self.name = 'RHFM'
        self.encode_K = kwargs.get('encode_K', self.m_max//2)
    
    def __call__(self, f_o, n=None, m=None, F_g=None, pos_nm=None, selective=False, imgrid=None, verbose=False):
        """
        f_o: polar image
        n, m: order, repetition
        F_g
        pos: momentum indexes
        """
        if F_g is None:# and self.Vmx is None:#####
            F_g = 1 / (self.N * self.M) * np.fft.fft2(self.G(f_o))
        
        # single momentum
        if (n is not None) and (m is not None):
            # use precomputed matrix
            if self.Vmx is not None and False:#####
                if imgrid is None:
                    imgrid = self.imgrid(f_o)
                return 1 / (self.N * self.M) * np.sum(imgrid * self.polar_r_fi[:,:,0] * np.conjugate(self.Vmx[n,m,:,:]))
            # use fft
            if n == 0:
                return np.sqrt(2) * F_g[0,m]
            if n > 0 and n % 2 == 0:
                k = n // 2
                return F_g[k,m] + F_g[-k,m]
            else:
                k = (n + 1) // 2
                return 1j * ( F_g[k,m] - F_g[-k,m] )
        
        # momentum list
        elif selective:
            if self.Vmx is not None and False:#####
                imgrid = self.imgrid(f_o)
                return [ (n,m,self(f_o, n, m, imgrid=imgrid)) for (n,m) in pos_nm ]
            else:
                return [ (n,m,self(f_o, n, m, F_g)) for (n,m) in pos_nm ]
        
        # momentum matrix
        else:
            F = self.get_momentum_mx()
            for n in range(self.n_max+1):  # -> n
                for m in range(self.m_max+1):  # -> m
                    F[n,m] = self(f_o, n, m, F_g)
            return F
    
    def T(self, n, r):
        """
        n: radial index
        r: radius
        """
        if r == 0: return 0
        if n == 0: return np.sqrt(1/r)
        elif n % 2 == 1: return np.sqrt(2/r) * np.sin( (n+1)*np.pi*r )
        else: return np.sqrt(2/r) * np.cos( n*np.pi*r )
    
    def H(self, n, r):
        """
        n: radial index
        r: radius
        """
        if n == 0: return np.sqrt(r)
        elif n % 2 == 1: return np.sqrt(2*r) * np.sin( (n+1)*np.pi*r )
        else: return np.sqrt(2*r) * np.cos( n*np.pi*r )
    
    def V(self, n, m, r, fi, useH=False):
        if useH: return self.H(n, r) * np.exp(1j*m*fi)
        else: return self.T(n, r) * np.exp(1j*m*fi)
    
    def G(self, f_o, r_u=None, fi_v=None):
        """
        f_o: polar image
        r_u, fi_v: polar coordinates
        """
        if r_u is not None and fi_v is not None:
            return self.dtype(f_o(r_u, fi_v)) * np.sqrt(r_u/2)
        else:
            mx = np.zeros((self.N,self.M), dtype=self.dtype)
            for u in range(self.N):  # -> F_{u,*}
                for v in range(self.M):  # F_{*,v}
                    r, fi = self.polar_uv(u, v)
                    mx[u,v] = self.dtype(f_o(r, fi)) * np.sqrt(r/2)
            return mx
    
    def moment_from_encode_pos(self, p):
        """
        p: bit index
        """
        K = self.encode_K
        w_space = K * (K-1) / 2
        if self.encode_dir == 'row':
            r = K - 2 - int((-1 + np.sqrt(1 + 8 * (w_space-p-1))) / 2)
            c = p - ((K-1)+(K-r))*r//2 + 1
            return (r, c)
        elif self.encode_dir == 'diagonal':
            d = int((-1 + np.sqrt(1 + 8 * p)) / 2)
            e = p - (1+d)*d//2
            return (e, d-e+1)
        else:
            return None
    
    def moments_from_encode_poss(self, pos):
        """
        pos: bit positions
        """
        pos_nm = []
        if isinstance(pos, int):
            pos = list(range(pos))
        for p in pos:
            if isinstance(p, tuple):
                pos_nm.append(p)
            else:
                pos_nm.append(self.moment_from_encode_pos(p))
        return pos_nm
##
