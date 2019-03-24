import numpy as np

def trim(array, a, b):
    '''Cuts a 2d array down to size 'a x b' by removing elements from the edges symmetrically.'''
    i=np.shape(array)[0]
    while i >  a:
        array = np.delete(array, 0, 0)
        array = np.delete(array, -1, 0)
        i = len(array[:, 0])

    i=np.shape(array)[1]
    while i > b:
        array = np.delete(array, 0, 1)
        array = np.delete(array, -1, 1)
        i = len(array[0, :])

    return array