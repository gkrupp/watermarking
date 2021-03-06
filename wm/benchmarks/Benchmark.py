import copy
import numpy as np
from ..Image import Image
from PIL import Image as PILImage

class Benchmark:
    
    def __init__(self, method, data, **kwargs):
        self.method = method
        self.data = data
        self.name = kwargs.get('name', self.__class__.__name__)
        self.repeat = kwargs.get('repeat', 1)
        self.colored = kwargs.get('colored', False)
        self.pos = kwargs.get('pos', None)
    
    def __call__(self, im) -> float:
        perf_sum = 0
        for k in range(self.repeat):
            tformed = self.transform(im)
            pim = self.to_polar(tformed)
            data_tform = self.decode(pim)
            perf_sum += self._perf(self.data, data_tform)
            self.cleanup()
        return perf_sum / self.repeat
    
    def transform(self, im) -> PILImage:
        return copy.deepcopy(im)
    
    def cleanup(self) -> None:
        return None
    
    def to_PIL(self, pim) -> PILImage:
        return PILImage.fromarray(pim.im)
    
    def to_polar(self, im) -> Image:
        return Image(np.array(im), self.colored)
    
    def decode(self, im) -> list:
        return self.method.decode(im, pos=self.pos)
    
    def _perf(self, data_orig, data_tform) -> float:
        if len(data_orig) == 0:
            return 1
        c = 0
        for i in range(min(len(data_orig),len(data_tform))):
            if data_orig[i] == data_tform[i]:
                c += 1
        return c / len(data_orig)
#
