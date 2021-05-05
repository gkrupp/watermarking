import numpy as np


def MSE(X, Y):
    w, h = X.shape[:2]
    return 1/(w*h) * np.sum((X - Y)**2)

def SNR(X, Y):
    w, h = X.shape[:2]
    return 10 * np.log10( 1/(w*h) * np.sum(X**2) / MSE(X, Y) )

def PSNR(X, Y, X_max=255):
    if X_max is None:
        X_max = np.max(X)
    return 10 * np.log10(X_max*X_max/MSE(X, Y))
