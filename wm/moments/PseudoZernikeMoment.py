import numpy as np

from .ZernikeMoment import ZernikeMoment

class PseudoZernikeMoment(ZernikeMoment):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'PZM'
    
    def R(self, n, m, r):
        if not self._correct_nm(n, m):
            return 0
        res = 0
        m_abs = np.abs(m)
        for s in range(n-m_abs+1):
            d0 = np.math.factorial(s)
            d1 = np.math.factorial(n+m_abs+1-s)
            d2 = np.math.factorial(n-m_abs-s)
            denom = d0 * d1 * d2
            res += ((-1.)**s) * np.math.factorial(2*n+1-s) * (r**(n-s)) / denom
        return res
#
