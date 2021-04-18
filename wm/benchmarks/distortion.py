from .Benchmark import Benchmark
from .. import Image
from ..metrics import MSE as MSE_metric, SNR as SNR_metric, PSNR as PSNR_metric
from PIL import Image as PILImage

import numpy as np


class MSE(Benchmark):
    def __init__(self, *args, original=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.original = np.array(original)
    def __call__(self, im) -> float:
        return MSE_metric(self.original, np.array(im))
#

class SNR(Benchmark):
    def __init__(self, *args, original=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.original = np.array(original)
    def __call__(self, im) -> float:
        return SNR_metric(self.original, np.array(im))
#

class PSNR(Benchmark):
    def __init__(self, *args, original=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.original = np.array(original)
    def __call__(self, im) -> float:
        return PSNR_metric(self.original, np.array(im))
#