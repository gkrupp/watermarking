import numpy as np
import pandas as pd

from PIL import Image
from multiprocessing import Pool, cpu_count
from .. import Image as PolarImage


class ConsistentRandomBits:
    
    def __init__(self, lengths, repetitions=1, seed=None):
        self.data = {}
        self.lengths = lengths
        self.repetitions = repetitions
        self.generate(seed)
    
    def generate(self, seed=None):
        self.data = {}
        if seed:
            np.random.seed(seed)
        for L in self.lengths:
            self.data[L] = [
                np.random.randint(2, size=L)
                for k in range(self.repetitions)
            ]
        return True
#

def execute_bs(imName, imPath, method, L, data, pos, bs_, k):
    print('\t'.join([imName, str(L), str(k)])+'\n', end='')
    ret = {
        'im': imName,
        'L': L,
        'k': k
    }
    pim = PolarImage(imPath, colored=False)
    im = Image.fromarray(pim.im)
    pime = method.encode(pim, data)
    ime = Image.fromarray(pime.im)
    for b in bs_:
        ret[b.name] = b(ime)
    return ret

def run(method, images, Ls, repetitions, gen_bs, imDir='../images/monochrome/'):
    CRB = ConsistentRandomBits(Ls, repetitions=repetitions)
    calls = []
    for imName in images:
        imPath = imDir + imName + '.png'
        pim = PolarImage(imPath, colored=False)
        im = Image.fromarray(pim.im)
        for L in Ls:
            order = int(np.sqrt(L))
            for k in range(repetitions):
                data = CRB.data[L][k]
                pos = np.arange(L)
                bs_ = gen_bs(method, data, pos, im)
                calls.append(
                    (imName, imPath, method, L, data, pos, bs_, k)
                )
    with Pool(cpu_count()) as p:
        res = p.starmap(execute_bs, calls)
    df = pd.DataFrame(res)
    return df
#
