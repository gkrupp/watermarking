import os, sys
sys.path.append(os.path.abspath('..'))

import numpy as np
import pandas as pd
from wm import Image as PolarImage
from PIL import Image

from wm.moments import RadialHarmonicFourierMoment
from wm.moments import ZernikeMoment, PseudoZernikeMoment

from functools import reduce
from multiprocessing import Pool

from wm.benchmarks.utils import run
from wm.benchmarks.distortion import MSE, SNR, PSNR
from wm.benchmarks.geometry import Resize, Rotate, Flip
from wm.benchmarks.noise import Blur, Sharpen, SaltPapperNoise
from wm.benchmarks.compression import JPEG, JPEG2000

W = 64
images = list(map(lambda name: name+'_'+str(W), [
    'airplane',
    'arctichare',
    'baboon',
    'barbara',
    'boat',
    'fruits',
    'lena',
    'monarch',
    'peppers',
    'zelda'
]))

def gen_bs(method, data, pos, im):
    scales = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.5, 2, 3]
    angles = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45]
    flips = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
    saltpepper = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    # METRICS
    bs_metrics = [
        MSE(method, data, pos=pos, original=im),
        SNR(method, data, pos=pos, original=im),
        PSNR(method, data, pos=pos, original=im)
    ]
    # RESIZE
    scales = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.5, 2, 3]
    bs_resize_nearest = [ Resize(method, data, pos=pos, scale=s, resample=Image.NEAREST, name='Resize/Nearest/'+str(s)) for s in scales ]
    bs_resize_linear  = [ Resize(method, data, pos=pos, scale=s, resample=Image.LINEAR,  name='Resize/Linear/'+str(s))  for s in scales ]
    bs_resize_bicubic = [ Resize(method, data, pos=pos, scale=s, resample=Image.BICUBIC, name='Resize/Bicubic/'+str(s)) for s in scales ]
    bs_resize_lanczos = [ Resize(method, data, pos=pos, scale=s, resample=Image.LANCZOS, name='Resize/Lanczos/'+str(s)) for s in scales ]
    # ROTATE
    angles = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45]
    bs_rotate_nearest = [ Rotate(method, data, pos=pos, angle=a, resample=Image.NEAREST, name='Rotate/Nearest/'+str(a)) for a in angles ]
    bs_rotate_linear  = [ Rotate(method, data, pos=pos, angle=a, resample=Image.LINEAR,  name='Rotate/Linear/'+str(a))  for a in angles ]
    bs_rotate_bicubic = [ Rotate(method, data, pos=pos, angle=a, resample=Image.BICUBIC, name='Rotate/Bicubic/'+str(a)) for a in angles ]
    # FLIP
    flips = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
    bs_flip = [ Flip(method, data, pos=pos, direction=d, name='Flip/'+str('UD' if d else 'LR')) for d in flips ]
    # GAUSSBLUR
    blurs = [1, 2, 3, 4, 5]
    bs_blur = [ Blur(method, data, pos=pos, radius=r, name='Blur/'+str(r)) for r in blurs ]
    # SHARPEN
    sharpens = [5, 10, 20, 30, 40, 50, 100]
    bs_sharpen = [ Sharpen(method, data, pos=pos, radius=3, percent=s, name='Sharpen/'+str(s)) for s in sharpens ]
    # GAUSSBLUR
    sps = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    bs_saltpepper = [ SaltPapperNoise(method, data, pos=pos, amount=a, name='SaltPepper/'+str(a)) for a in sps ]
    # JPEG
    qualities = [100, 95, 90, 85, 80, 75, 70, 65, 60, 50, 10, 5]
    bs_jpeg = [ JPEG(method, data, pos=pos, quality=q, name='JPEG/'+str(q)) for q in qualities ]
    bs_jpeg2000 = [ JPEG2000(method, data, pos=pos, name='JPEG2000') ]
    ###
    ###
    return reduce(lambda a,b: a+b, [
        # RESIZE
        #bs_resize_nearest,
        bs_resize_linear,
        #bs_resize_bicubic,
        #bs_resize_lanczos,
        # ROTATE
        #bs_rotate_nearest,
        bs_rotate_linear,
        #bs_rotate_bicubic,
        # FLIP
        bs_flip,
        # NOISE
        bs_blur,
        bs_sharpen,
        bs_saltpepper,
        # COMPRESSION
        bs_jpeg,
        bs_jpeg2000,
    ], [])

max_order = 50
method = RadialHarmonicFourierMoment(max_order, W)
Ls = [ 8*l for l in range(2+1) ]
repetitions = 10

df = run(method, images, Ls, repetitions, gen_bs, '../images/monochrome/')

df.to_csv(''.join([ method.name, '_', str(Ls[0]), '_', str(Ls[-1]) ]))

