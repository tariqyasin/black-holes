import numpy as np
import sys

import cPickle as pickle
from astropy.io import fits

from trim import trim


def bin_model(model_list, bin_array):
    binned_list = list()

    k = 0
    for i in range(0, len(bin_array)):
        value = 0
        
        j = bin_array[i]
        for i in range(0,j):
            value += model_list[k]
            k += 1

        value /= j
        binned_list.append(value)

    return np.asarray(binned_list)

def calculate_chi(data, error, model):
    res = ((data - model) / error)**2
    chi = np.sum(res)
    return chi

