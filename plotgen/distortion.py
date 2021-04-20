import os, sys
sys.path.append(os.path.abspath('..'))

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
parser.add_argument('a', metavar='a', type=int)
parser.add_argument('b', metavar='b', type=int)
args = parser.parse_args()

max_L_exp = 32
max_order = 50
method = RadialHarmonicFourierMoment(max_order, W, Vfile='../V_RHFM_256.h5')
Ls = [ 8*l for l in range(max_L_exp+1) ]
repetitions = 10

df = run(method, images[args.a:args.b], Ls, repetitions, gen_bs, '../images/monochrome/', multiproc=False)

df.to_csv('RHFM_'+str(args.a)+str(args.b)+'.csv')