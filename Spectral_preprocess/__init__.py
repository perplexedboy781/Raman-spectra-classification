import numpy as np
from Spectral_preprocess.dataprocess import airPLS
from scipy import signal

def max_min(nd):
    nd_max = np.max(nd, axis=1).reshape(nd.shape[0], 1)
    nd_min = np.min(nd, axis=1).reshape(nd.shape[0], 1)
    nd_norm = (nd - nd_min) / (nd_max - nd_min)
    return nd_norm

def airpls(data):
    m, n = data.shape
    datadone = []
    for i in range(m):
        datadone.append(data[i] - airPLS(data[i]))
    datadone = np.array(datadone)
    return datadone


def SG(data):
    return signal.savgol_filter(data,20, 2)