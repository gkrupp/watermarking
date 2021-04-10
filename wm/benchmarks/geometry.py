from .Benchmark import Benchmark
from .. import Image
from PIL import Image as PILImage


class Resize(Benchmark):
    def __init__(self, *args, scale=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = scale
    def transform(self, im) -> PILImage:
        w, h = im.size
        w, h = int(self.scale*w), int(self.scale*h)
        return im.resize((w,h))
#

class Rotate(Benchmark):
    def __init__(self, *args, angle=0, resample=PILImage.BICUBIC, **kwargs):
        super().__init__(*args, **kwargs)
        self.angle = angle
        self.resample = resample
    def transform(self, im) -> PILImage:
        return im.rotate(self.angle, resample=self.resample)
#

class Flip(Benchmark):
    def __init__(self, *args, direction=PILImage.FLIP_LEFT_RIGHT, **kwargs):
        super().__init__(*args, **kwargs)
        self.direction = direction
    def transform(self, im) -> PILImage:
        return im.transpose(self.direction)
#
