from .Benchmark import Benchmark
from .. import Image
from PIL import Image as PILImage

import os
import time
import numpy as np


class JPEG(Benchmark):
    def __init__(self, *args, quality=0, optimize=False, tmpfile=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.quality = quality
        self.optimize = optimize
        self.tmpfile = tmpfile
        if self.tmpfile is None:
            np.random.seed(int(time.time()*10e6)%(2**32))
            self.tmpfile = ''.join(map(chr, np.random.randint(ord('a'), ord('z'), size=16)))+'.jpg'
    def transform(self, im) -> PILImage:
        im.save(self.tmpfile, quality=self.quality, optimize=self.optimize)
        return PILImage.open(self.tmpfile)
    def cleanup(self):
        os.remove(self.tmpfile)
#

class JPEG2000(Benchmark):
    def __init__(self, *args, tmpfile=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tmpfile = tmpfile
        if self.tmpfile is None:
            np.random.seed(int(time.time()*10e6)%(2**32))
            self.tmpfile = ''.join(map(chr, np.random.randint(ord('a'), ord('z'), size=16)))+'.j2k'
    def transform(self, im) -> PILImage:
        im.save(self.tmpfile)
        return PILImage.open(self.tmpfile)
    def cleanup(self):
        os.remove(self.tmpfile)
#
