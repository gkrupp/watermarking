from .Benchmark import Benchmark
from .. import Image
from PIL import Image as PILImage
import cv2
import numpy as np
from PIL.ImageFilter import GaussianBlur, UnsharpMask


class Blur(Benchmark):
    def __init__(self, *args, size=(3,3), sigma=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = size
        self.sigma = sigma
    def transform(self, im) -> PILImage:
        return PILImage.fromarray(cv2.GaussianBlur(np.array(im), self.size, self.sigma))
#

class Sharpen(Benchmark):
    def __init__(self, *args, size=0, sigma=0, ratio=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = size
        self.sigma = sigma
        self.ratio = ratio
    def transform(self, im) -> PILImage:
        arr = np.array(im)
        blurred = cv2.GaussianBlur(arr, self.size, self.sigma)
        return PILImage.fromarray(self.ratio*arr - (self.ratio-1)*blurred)
#

class Median(Benchmark):
    def __init__(self, *args, size=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = size
    def transform(self, im) -> PILImage:
        return PILImage.fromarray(cv2.medianBlur(np.array(im), self.size))
#

class SaltPapperNoise(Benchmark):
    def __init__(self, *args, amount=0.001, salt=0.5, repeat=5, **kwargs):
        super().__init__(*args, repeat=repeat, **kwargs)
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

class WhiteNoise(Benchmark):
    def __init__(self, *args, ratio=0.01, limits=(-128,128), **kwargs):
        super().__init__(*args, **kwargs)
        self.ratio = ratio
        self.limits = limits
    def transform(self, im) -> PILImage:
        w, h = im.size
        noise = np.random.randint(*self.limits, size=(w,h))
        retim = np.array(im)
        retim = np.clip(retim.astype('float32') + self.ratio * noise, 0, 255).astype('uint8')
        return PILImage.fromarray(retim)
#