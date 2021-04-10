import numpy as np

from ..Image import Image
from .Moment2D import Moment2D


class RadialHarmonicFourierMoment(Moment2D):
    
    def __init__(self, n_max, N=None, **kwargs):
        """
        n_max: order
        N: width
        """
        super().__init__(n_max, N or n_max, **kwargs)
        #
        self.qs = kwargs.get('qs', 0.3)  # quantization step
        self.encode_dir = kwargs.get('encode_dir', 'row')
        self.encode_K = kwargs.get('encode_K', self.m_max)
    
    def __call__(self, f_o, n=None, m=None, F_g=None):
        """
        f_o: polar image
        n
        m
        F_g
        """
        if F_g is None:
            F_g = 1/(self.N*self.M) * np.fft.fft2(self.G(f_o))
            
        # single momentum
        if (n is not None) and (m is not None):
            if n == 0:
                return self.sqrt(2) * F_g[0,m]
            if n > 0 and n % 2 == 0:
                k = n//2
                return F_g[k,m] + F_g[-k,m]
            else:
                k = (n+1)//2
                #F = 0
                #for u in range(self.N):  # \sum_{u=0}^{N-1} -> r_u
                #    for v in range(self.M):  # \sum_{v=0}^{M-1} -> fi_v
                #        (r_u, fi_v) = self.polar_uv(u, v)  # polar pos from discrete pos
                #        #
                #        g = self.G(f_o, r_u, fi_v)
                #        h1 = self.exp(-self.i*2*(-k)*self.pi*u/self.N)
                #        ex = self.exp(-self.i*m*2*self.pi*(v/self.M))
                #        rm = self.exp(-self.i*self.pi*u/self.N)
                #        F += g * h1 * ex * rm
                #for u in range(self.N):  # \sum_{u=0}^{N-1} -> r_u
                #    for v in range(self.M):  # \sum_{v=0}^{M-1} -> fi_v
                #        (r_u, fi_v) = self.polar_uv(u, v)  # polar pos from discrete pos
                #        #
                #        g = self.G(f_o, r_u, fi_v)
                #        h2 = self.exp(-self.i*(2*k-1)*self.pi*u/self.N)
                #        ex = self.exp(-self.i*m*2*self.pi*(v/self.M))
                #        F += g * h2 * ex
                return self.i * ( F_g[k,m] - F_g[-k,m] )
        
        # momentum matrix
        else:
            F = self.momentum_mx((self.n_max+1,2*self.m_max+1))
            for n in range(self.n_max+1):  # -> n
                for m in range(-self.m_max, self.m_max+1):  # -> m
                    F[n,m] = self(f_o, n, m, F_g)
            return F
    
    def T(self, n, r):
        """
        n: radius index
        r: radius
        """
        if r == 0: return 0
        if n == 0: return self.sqrt(1/r)
        elif n % 2 == 1: return self.sqrt(2/r) * self.sin((n+1)*self.pi*r)
        else: return self.sqrt(2/r) * self.cos(n*self.pi*r)
    
    def H(self, n, r):
        """
        n: radius index
        r: radius
        """
        if n == 0:
            return self.sqrt(r)
        elif n % 2 == 1:
            return 1/self.i * self.sqrt(r/2) * ( self.exp(self.i*(n+1)*self.pi*r) - self.exp(-self.i*(n+1)*self.pi*r) )
        else:
            return self.sqrt(r/2) * ( self.exp(self.i*n*self.pi*r) + self.exp(-self.i*n*self.pi*r) )
    
    def reconstruct(self, F, width, height=None, K=None, verbose=False):
        """
        F: fourier matrix (n,m: Fourier)
        width: 
        height: 
        """
        height = height or width
        K = K or self.n_max
        colored = False
        I = Image(np.zeros((width,height), dtype=np.uint8), colored=colored)
        for u in range(width):  # -> F_{u,*}
            if verbose: print('px: '+str(u*width)+'/'+str(width*height)+' '*32, end='\r')
            for v in range(height):  # F_{*,v}
                (r, fi) = I.pos_to_polar(u, v)
                if r > 1: continue
                f = 0
                Kn = min(self.n_max,K+1)
                for n in range(Kn):
                    Km = min(K-n,self.m_max)
                    for m in range(-Km, Km+1):
                        f += F[n,m] * self.T(n, r) * self.exp(self.i*m*fi)
                I.im[u,v] = self.pxtype(f)
        return Image(I.im, colored=colored)
    
    def moment_from_encode_pos(self, p):
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
        pos_nm = []
        if isinstance(pos, int):
            pos = list(range(pos))
        for p in pos:
            if isinstance(p, tuple):
                pos_nm.append(p)
            else:
                pos_nm.append(self.moment_from_encode_pos(p))
        return pos_nm
    
    def encode(self, f_o, w=None, pos=None, verbose=False):
        # choose moments to modify
        pos_nm = self.moments_from_encode_poss( pos or len(w) )
        if self.w_space < len(pos_nm):
            print('ERR: not enough moments')
            return None
        # RHFMs
        if verbose: print('1/4: moments'+' '*32, end='\r')
        F = self(f_o)
        # Reconstruction with unmodified RHFMs
        if verbose: print('2/4: reconstruction', end='\r')
        R = self.reconstruct(F, f_o.width, f_o.height, verbose=verbose).im
        D = f_o.im - R
        # modify
        if verbose: print('3/4: encoding'+' '*32, end='\r')
        for k in range(len(w)):
            (n, m) = pos_nm[k]
            F_nm_abs = self.abs(F[n,m])
            l_k = np.round(F_nm_abs / self.qs)
            d = -1/2 if ((l_k + w[k]) % 2 == 1) else 1/2
            F_nm_abs_mod = (l_k + d) * self.qs
            F[n,m] = (F_nm_abs_mod / F_nm_abs) * F[n,m]
            F[n,-m] = (F_nm_abs_mod / F_nm_abs) * F[n,-m]
        # Reconstruction with modified RHFMs
        if verbose: print('4/4: modified reconstruction'+' '*32, end='\r')
        E = self.reconstruct(F, f_o.width, f_o.height, verbose=verbose).im
        # Image combination
        if verbose: print('done'+' '*32, end='\r')
        return Image(E + D, colored=f_o.colored)
    
    def decode(self, f_o, pos=None):
        # choose moments to decode
        pos_nm = self.moments_from_encode_poss( pos or ((self.n_max+self.m_max)//2) )
        if self.w_space < len(pos_nm):
            print('ERR: not enough moments')
            return None
        # Modified RHFMs
        F = self(f_o)
        # decode
        w = []
        for (n, m) in pos_nm:
            F_nm_abs = self.abs(F[n,m])
            l_k = np.floor(F_nm_abs / self.qs)
            w_k = int(l_k % 2)
            #print(n, m, abs(F[n,m]), np.floor(F_nm_abs / self.qs), w_k)
            w.append(w_k)
        return w
    
    def G(self, f_o, r_u=None, fi_v=None):
        if r_u is not None and fi_v is not None:
            return self.dtype(f_o(r_u, fi_v)) * self.sqrt(r_u/2)
        else:
            mx = np.zeros((self.N,self.M), dtype=self.dtype)
            for u in range(self.N):  # -> F_{u,*}
                for v in range(self.M):  # F_{*,v}
                    (r_u, fi_v) = self.polar_uv(u, v)
                    mx[u,v] = self.dtype(f_o(r_u, fi_v)) * self.sqrt(r_u/2)
            return mx
##
