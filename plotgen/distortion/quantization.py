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
from wm.benchmarks.distortion import MSE, SNR, PSNR

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

def gen_bs(method, data, pos, im):
    # METRICS
    bs_metrics = [
        MSE(method, data, pos=pos, original=im),
        SNR(method, data, pos=pos, original=im),
        PSNR(method, data, pos=pos, original=im)
    ]
    ###
    ###
    return reduce(lambda a,b: a+b, [
        bs_metrics,
    ], [])

parser = argparse.ArgumentParser()
parser.add_argument('m', metavar='m', type=str)
parser.add_argument('d', metavar='d', type=float)
args = parser.parse_args()

max_L_exp = 16
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

df = run(method, images, Ls, repetitions, gen_bs, '../../images/monochrome/', multiproc=False)

df.to_csv(str(method.name)+'_'+str(args.d)+'.csv', index=False)
