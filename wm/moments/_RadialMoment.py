import numpy as np

from ..Image import Image

class _RadialMoment:
    
    def __init__(self, n_max, N=None, m_max=None, M=None, **kwargs):
        """
        n_max: max order
        N: image height
        m_max: max repetition
        M: image width
        """
        self.dtype = kwargs.get('dtype', complex)
        self.pxtype = kwargs.get('dtype', np.real)
        # N:sampling, M:sampling
        self.N = N or n_max
        self.M = M or self.N
        # n_max:order, m_max:repetition
        self.n_max = n_max
        self.m_max = m_max or n_max
        self.w_space = None
        # coding
        self.qs = kwargs.get('qs', 0.3)
        self.encode_dir = kwargs.get('encode_dir', 'diagonal')
        self.encode_K = kwargs.get('encode_K', self.m_max)
        #
        self.polar_r_fi  = np.zeros((self.N,self.M,2))
        for u in range(self.N):
            for v in range(self.M):
                self.polar_r_fi[u,v] = self.pos_to_polar(u, v)
    
    def __call__(self):
        raise NotImplementedError
    
    def reconstruct_px(self, F, u, v, width=None, height=None, selective=True, verbose=False):
        r, fi = self.polar_r_fi[u,v] #self.pos_to_polar(u, v, width, height)
        if r > 1: return 0
        #
        x = 0
        if selective:
            for (n,m,F_nm) in F:
                x_nm = F_nm * self.V(n, m, r, fi)
                if m: x_nm = 2*np.real(x_nm)
                x += x_nm
        else:
            for n in range(F.shape[0]):
                for m in range(F.shape[1]):
                    x_nm = F[n,m] * self.V(n, m, r, fi)
                    if m: x_nm = 2*np.real(x_nm)
                    x += x_nm
        return self.pxtype(x)
    
    def reconstruct(self, F, width, height=None, selective=True, verbose=False):
        """
        F: momentum matrix (n,m)
        width: image width
        height: image height
        """
        height = height or width
        I = Image(np.zeros((width,height), dtype=np.uint8), colored=False)
        for u in range(width):
            if verbose: print(str((u+1)*width)+'/'+str(width*height)+' '*32, end='\r')
            for v in range(height):
                I.im[u,v] = self.reconstruct_px(F, u, v, width, height, selective=selective)
        return Image(I.im, colored=False)
    
    ## Coding
    
    def encode(self, f_o, w=None, pos=None, selective=True, verbose=False):
        
        # choose moments to modify
        pos_nm = self.moments_from_encode_poss( pos or len(w) )
        
        # compute moments
        F = self(f_o, pos_nm=pos_nm, selective=selective, verbose=verbose)
        
        # Reconstruction with unmodified moments
        R = self.reconstruct(F, f_o.width, f_o.height, selective=selective, verbose=verbose).im
        
        # Reconstruction error (difference)
        D = f_o.im - R
        
        # modify
        for k in range(len(w)):
            (n, m) = pos_nm[k]
            if selective:
                F_nm_abs = np.abs(F[k][-1])
                l_k = np.round(F_nm_abs / self.qs)
                d = -1/2 if ((l_k + w[k]) % 2 == 1) else 1/2
                F_nm_abs_mod = (l_k + d) * self.qs
                F_nm_mod = (F_nm_abs_mod / F_nm_abs) * F[k][-1]
                F[k] = (n, m, F_nm_mod)
            else:
                F_nm_abs = np.abs(F[n,m])
                l_k = np.round(F_nm_abs / self.qs)
                d = -1/2 if ((l_k + w[k]) % 2 == 1) else 1/2
                F_nm_abs_mod = (l_k + d) * self.qs
                F[n,m] = (F_nm_abs_mod / F_nm_abs) * F[n,m]
        
        # Reconstruction with modified moments
        E = self.reconstruct(F, f_o.width, f_o.height, selective=selective, verbose=verbose).im
        
        # Image combination
        if verbose: print('done'+' '*32, end='\r')
        return Image(E + D, colored=f_o.colored)
    
    def decode_moment(self, F_nm):
        F_nm_abs = np.abs(F_nm)
        l_nm = np.floor(F_nm_abs / self.qs)
        w_nm = int(l_nm % 2)
        return w_nm
    
    def decode(self, f_o, pos=None, selective=True, verbose=False):
        
        # choose moments to decode
        pos_nm = self.moments_from_encode_poss( pos ) # or ((self.n_max+self.m_max)//2) )
        
        # modified moments
        F = self(f_o, pos_nm=pos_nm, selective=selective, verbose=verbose)
        
        # decode
        if selective:
            w = [ self.decode_moment(F[k][-1]) for k in range(len(pos_nm)) ]
        else:
            w = [ self.decode_moment(F[n][m]) for (n,m) in pos_nm ]
        
        # return
        if verbose: print('done'+' '*32, end='\r')
        return w
    
    ## Helpers
    
    def get_momentum_mx(self):
        return np.zeros((self.n_max+1, self.m_max+1), dtype=self.dtype)
    
    def polar_uv(self, u, v):
        """
        u: r index
        v: fi index
        """
        return (
            1 * (u / self.N),
            2 * np.pi * (v / self.M)
        )
    
    def pos_to_polar(self, x, y, N=None, M=None):
        """
        x: x coord
        y: y coord
        """
        N = N or self.N
        M = M or self.M
        x_p = (2*x-N+1)/(N-1)
        y_p = (2*y-M+1)/(M-1)
        return (
            np.sqrt((x_p*x_p)+(y_p*y_p)),
            np.arctan2(y_p, x_p)
        )
##
