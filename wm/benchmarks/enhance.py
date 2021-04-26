from .Benchmark import Benchmark
from .. import Image
from PIL import Image as PILImage
import cv2
import numpy as np
from PIL.ImageEnhance import Contrast as PILContrast, Brightness as PILBrightness


class Contrast(Benchmark):
    def __init__(self, *args, factor=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.factor = factor
    def transform(self, im) -> PILImage:
        return PILContrast(im).enhance(self.factor)
#

class Brightness(Benchmark):
    def __init__(self, *args, factor=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.factor = factor
    def transform(self, im) -> PILImage:
        return PILBrightness(im).enhance(self.factor)
#
