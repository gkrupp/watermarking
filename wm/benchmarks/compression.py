from .Benchmark import Benchmark
from .. import Image
from PIL import Image as PILImage
import numpy as np


class JPEG(Benchmark):
    def __init__(self, *args, quality=0, optimize=True, tmpfile='tmp.jpeg', **kwargs):
        super().__init__(*args, **kwargs)
        self.quality = quality
        self.optimize = optimize
        self.tmpfile = tmpfile
    def transform(self, im) -> PILImage:
        im.save(self.tmpfile, quality=self.quality, optimize=self.optimize)
        return PILImage.open(self.tmpfile)
#

class JPEG2000(Benchmark):
    def __init__(self, *args, tmpfile='tmp.j2k', **kwargs):
        super().__init__(*args, **kwargs)
        self.tmpfile = tmpfile
    def transform(self, im) -> PILImage:
        im.save(self.tmpfile)
        return PILImage.open(self.tmpfile)
#
