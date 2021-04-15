import copy
import numpy as np
from ..Image import Image
from PIL import Image as PILImage

class Benchmark:
    
    def __init__(self, method, data, **kwargs):
        self.method = method
        self.data = data
        self.name = kwargs.get('name', '')
        self.colored = kwargs.get('colored', False)
        self.pos = kwargs.get('pos', None)
    
    def __call__(self, im) -> float:
        tformed = self.transform(im)
        pim = self.to_polar(tformed)
        data_tform = self.decode(pim)
        return self._perf(self.data, data_tform)
    
    def transform(self, im) -> PILImage:
        return copy.deepcopy(im)
    
    def to_PIL(self, pim) -> PILImage:
        return PILImage.fromarray(pim.im)
    
    def to_polar(self, im) -> Image:
        return Image(np.array(im), self.colored)
    
    def decode(self, im) -> list:
        return self.method.decode(im, pos=self.pos)
    
    def _perf(self, data_orig, data_tform) -> float:
        c = 0
        for i in range(min(len(data_orig),len(data_tform))):
            if data_orig[i] == data_tform[i]:
                c += 1
        return c / len(data_orig)
#
