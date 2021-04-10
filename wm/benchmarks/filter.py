from .Benchmark import Benchmark
from .. import Image
from PIL import Image as PILImage
import numpy as np
from PIL.ImageFilter import GaussianBlur, UnsharpMask


class Blur(Benchmark):
    def __init__(self, *args, radius=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.radius = radius
    def transform(self, im) -> PILImage:
        return im.filter(GaussianBlur(self.radius))
#

class Sharpen(Benchmark):
    def __init__(self, *args, radius=0, percent=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.radius = radius
        self.percent = percent
    def transform(self, im) -> PILImage:
        return im.filter(UnsharpMask(self.radius, self.percent))
#

class SaltPapperNoise(Benchmark):
    def __init__(self, *args, amount=0.001, salt=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.amount = amount
        self.salt = salt
    def transform(self, im) -> PILImage:
        pepper = 1 - self.salt
        w, h = im.size
        probs = np.random.rand(w, h)
        retim = im.copy()
        pa = retim.load()
        for i in range(w):
            for j in range(h):
                if probs[i,j] < self.amount:
                    sb = 1 if np.random.random() < self.salt else 0
                    if isinstance(pa[i,j], int):
                        pa[i,j] = sb*255
                    else:
                        pa[i,j] = (sb*255, sb*255, sb*255)
        return retim
#
