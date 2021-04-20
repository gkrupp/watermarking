import numpy as np


def MSE(X, Y):
    w, h = X.shape[:2]
    return 1/(w*h) * np.sum((X.astype('float32') - Y.astype('float32'))**2)

def SNR(X, Y):
    w, h = X.shape[:2]
    return 1/(w*h) * np.sum(X**2) / MSE(X, Y)

def PSNR(X, Y, X_max=255):
    if X_max is None:
        X_max = np.max(X)
    return 20 * np.log10(X_max/np.sqrt(MSE(X, Y)))