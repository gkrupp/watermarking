import os, sys
sys.path.append(os.path.abspath('../..'))

import numpy as np
import pandas as pd
from wm import Image as PolarImage
from PIL import Image

from wm.moments import RadialHarmonicFourierMoment
from wm.moments import ZernikeMoment, PseudoZernikeMoment

import argparse
from functools import reduce
from multiprocessing import Pool

from wm.benchmarks.utils import run

from wm.benchmarks.geometry import Resize, Rotate, Flip
from wm.benchmarks.noise import Blur, Sharpen, Median, SaltPapperNoise, WhiteNoise
from wm.benchmarks.enhance import Contrast, Brightness
from wm.benchmarks.compression import JPEG, JPEG2000, WebP

W = 256
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

from wm.benchmarks.geometry import Resize, Rotate, Flip
def gen_bs(method, data, pos, im):
    # RESIZE
    scales = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 2, 3]
    bs_resize_linear  = [ Resize(method, data, pos=pos, scale=s, resample=Image.LINEAR,  name='Resize/Linear/'+str(s))  for s in scales ]
    bs_resize_bicubic = [ Resize(method, data, pos=pos, scale=s, resample=Image.BICUBIC, name='Resize/Bicubic/'+str(s)) for s in scales ]
    # ROTATE
    angles = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    bs_rotate_linear  = [ Rotate(method, data, pos=pos, angle=a, resample=Image.LINEAR,  name='Rotate/Linear/'+str(a))  for a in angles ]
    bs_rotate_bicubic = [ Rotate(method, data, pos=pos, angle=a, resample=Image.BICUBIC, name='Rotate/Bicubic/'+str(a)) for a in angles ]
    # FLIP
    flips = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
    bs_flip = [ Flip(method, data, pos=pos, direction=d, name='Flip/'+str('UD' if d else 'LR')) for d in flips ]
    ######
    # GAUSSBLUR
    blurs = [ 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3 ]
    bs_blur_3 = [ Blur(method, data, pos=pos, size=(3,3), sigma=s, name='Blur/3x3/'+str(s)) for s in blurs ]
    bs_blur_5 = [ Blur(method, data, pos=pos, size=(3,3), sigma=s, name='Blur/5x5/'+str(s)) for s in blurs ]
    bs_blur_7 = [ Blur(method, data, pos=pos, size=(3,3), sigma=s, name='Blur/7x7/'+str(s)) for s in blurs ]
    # SHARPEN
    sharpens = [ 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4 ]
    bs_sharpen_3 = [ Sharpen(method, data, pos=pos, size=(3,3), sigma=2, ratio=r, name='Sharpen/3x3/'+str(r)) for r in sharpens ]
    bs_sharpen_5 = [ Sharpen(method, data, pos=pos, size=(3,3), sigma=2, ratio=r, name='Sharpen/5x5/'+str(r)) for r in sharpens ]
    bs_sharpen_7 = [ Sharpen(method, data, pos=pos, size=(3,3), sigma=2, ratio=r, name='Sharpen/7x7/'+str(r)) for r in sharpens ]
    # MEDIAN
    medians = [ 3, 5, 7, 9 ]
    bs_median = [ Median(method, data, pos=pos, size=s, name='Median/'+str(s)) for s in medians ]
    # SALTPEPPER
    sps = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1]
    bs_saltpepper = [ SaltPapperNoise(method, data, pos=pos, amount=a, name='SaltPepper/'+str(a)) for a in sps ]
    # WHITENOISE
    wn = [ 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    bs_whitenoise = [ WhiteNoise(method, data, pos=pos, ratio=r, name='WhiteNoise/'+str(r)) for r in wn ]
    ######
    # CONTRAST
    contrasts = [ 0.8, 0.9, 0.95, 1, 1.05, 1.1, 1.2]
    bs_contrast = [ Contrast(method, data, pos=pos, factor=f, name='Contrast/'+str(f)) for f in contrasts ]
    # BRIGHTNESS
    brightnesses = [ 0.8, 0.9, 0.95, 1, 1.05, 1.1, 1.2]
    bs_brightness = [ Brightness(method, data, pos=pos, factor=f, name='Brightness/'+str(f)) for f in contrasts ]
    ######
    # JPEG
    qualities = [100, 90, 80, 70, 60, 50, 40, 30, 25, 20, 15, 10, 5]
    bs_jpeg =     [ JPEG(method, data, pos=pos, quality=q, name='JPEG/'+str(q)) for q in qualities ]
    bs_jpeg2000 = [ JPEG2000(method, data, pos=pos, name='JPEG2000') ]
    bs_webp =     [ WebP(method, data, pos=pos, quality=q, name='WebP/'+str(q)) for q in qualities ]
    ######
    ###
    ###
    return reduce(lambda a,b: a+b, [
        # GEOMETRY
        bs_resize_linear, bs_resize_bicubic,
        bs_rotate_linear, bs_rotate_bicubic,
        bs_flip,
        # FILTER
        bs_blur_3, bs_blur_5, bs_blur_7,
        bs_sharpen_3, bs_sharpen_5, bs_sharpen_7,
        bs_median,
        # ENHANCE
        bs_contrast, bs_brightness,
        # NIOSE
        bs_saltpepper, bs_whitenoise,
        # COMPRESSION
        bs_jpeg, bs_jpeg2000, bs_webp,
    ], [])

parser = argparse.ArgumentParser()
parser.add_argument('m', metavar='m', type=str)
parser.add_argument('a', metavar='a', type=int)
parser.add_argument('b', metavar='b', type=int)
args = parser.parse_args()

max_L_exp = 8
max_order = 50
Ls = [ 64 ]
repetitions = 5

if args.m == 'RHFM':
    method = RadialHarmonicFourierMoment(max_order, W, Vfile='../../V_RHFM_256.h5')
elif args.m == 'ZM':
    method = ZernikeMoment(max_order, W, Vfile='../../V_ZM_256.h5')
elif args.m == 'PZM':
    method = PseudoZernikeMoment(max_order, W, Vfile='../../V_PZM_256.h5')
else:
    method = None

df = run(method, images[args.a:args.b], Ls, repetitions, gen_bs, '../../images/monochrome/', multiproc=False)

df.to_csv(str(method.name)+'_'+str(args.a)+str(args.b)+'.csv', index=False)
