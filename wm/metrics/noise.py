import numpy as np


def MSE(X, Y):
    w, h = X.shape[:2]
    return 1/(w*h) * np.sum((X.astype('float32') - Y.astype('float32'))**2)

def SNR(X, Y):
    w, h = X.shape[:2]
    return 1/(w*h) * np.sum(X**2) / MSE(X, Y)

def PSNR(X, Y):
    return 20 * np.log10(np.max(X)/np.sqrt(MSE(X, Y)))