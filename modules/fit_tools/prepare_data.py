import sys
import numpy as np
import scipy as sp

import cPickle as pickle
from astropy.io import fits
import scipy.constants as sc
import os

from trim import trim


def prepare_data(name, noiseless):
    '''A function that prepares the HSIM datacubes for analysis. 
    
    Arguments:
    --------------- 
            name: String. The name of the HARMONI input/output cubes.
                    For example if name='/dir/name' then the following files must exist:
                    -----------------
                    /dir/name.fits 
                    /dir/name_Reduced_cube.fits
                    /dir/name_Transmission_cube.fits
                    /dir/name_Noiseless_Object_cube.fits
                    ----------------

            noiseless: 'True' or 'False' (string). If 'True' then use the noiseless cube.

    Returns:
    ----------------
            data: A list. Each list element corresponds to a single HARMONI spaxel. 
                    Each list element is a dictionary with the following keywords:
                    ------------------
                        'rad' - the radius of the spaxel
                        'spec' - a 1D array. The spectrum of the spaxel.
                        'spec_err' - a 1D array. The measurement error in the spectrum (1 standard deviation)
                        'I_mod' - the intensity predicted by the PSF-convolved theoretical model for the spaxel.
                        'S_mod' - the second moment predicted by the PSF-convolved theoretical model for the spaxel.
                    --------------------

            wavel: An array corresponding to the wavelength axis of the HSIM output cubes.
            in_wavel: An array corresponding to the wavelength axis of the HSIM input cube.
    '''

    # Read in transmission cube.
    hdulist = fits.open(u'{0}_Transmission_cube.fits'.format(name))
    trans_cube = hdulist[0].data

    # Read in data from reduced cube or noiseless object cube.
    if noiseless == 'True':
        hdulist = fits.open(u'{0}_Noiseless_Object_cube.fits'.format(name))
        header = hdulist[0].header

        data_cube = hdulist[0].data
        data_cube = data_cube / trans_cube

    else:
        hdulist = fits.open(u'{0}_Reduced_cube.fits'.format(name))
        header = hdulist[0].header

        data_cube = hdulist[0].data
        data_cube = data_cube / trans_cube

    # Read in measurement error from second extension of reduced cube.
    hdulist = fits.open(u'{0}_Reduced_cube.fits'.format(name))
    var_cube = hdulist[1].data
    err_cube = np.sqrt(var_cube)   # Convert variance to standard deviation
    err_cube = err_cube / trans_cube

    print('Data cubes read')

    # Get header values that define wavelength axes of HSIM output cubes and create wavelength array.
    sample = header[u'CDELT3']
    CRVAL3 = header[u'CRVAL3']
    NAXIS3 = header[u'NAXIS3']

    wavel = np.zeros(NAXIS3)
    for i in range(0, NAXIS3):
        wavel[i] = (CRVAL3 + (i * sample)) * 1e4

    # Get header values that define spatial scale.
    NAXIS1 = header[u'NAXIS1']
    NAXIS2 = header[u'NAXIS2']
    CDELT1 = header[u'CDELT1']
    CDELT2 = header[u'CDELT2']

    # Check if distance keyword exists in header (in Mpc). Needed to recover the correct intensity.
    # If not assign just make it 1.
    if 'D' in header:
        d = header[u'D'] 
    else: 
        d = 1e6

    # Convert the data and errors from units of electrons to solar luminosities.
    electron2ph = sc.h * sc.c / (wavel * 1e-10)
    electron2lum = ph2lum(electron2ph, d, sample, CDELT1, CDELT2)

    data_cube = np.transpose(data_cube)
    data_cube *= electron2lum
    data_cube = np.transpose(data_cube)

    err_cube = np.transpose(err_cube)
    err_cube *= electron2lum
    err_cube = np.transpose(err_cube)

    # Convert the 3D cubes into 2D array, with each entry containing the spectral data for
    # a single spaxel. 

    data_square = np.zeros((NAXIS2, NAXIS1), dtype=object)
    err_square = np.empty((NAXIS2, NAXIS1), dtype=object)

    for j in range(0, NAXIS2):
        for i in range(0, NAXIS1):
            data_square[j, i] = data_cube[:, j, i]
            err_square[j, i] = err_cube[:, j, i]

    # Trim data down to size of HARMONI's FOV. Be careful here as X and Y axis are inverted.

    a = 213 # y-axis HARMONI FOV 
    b = 151 # x-axis HARMONI FOV

    data_square = trim(data_square, a, b)
    err_square = trim(err_square, a, b)

    NAXIS2, NAXIS1 = np.shape(data_square)

    print 'Trimmed data to HARMONI FOV'

    # Convert from 2D array to lists with corresponding radius list R_list.
    RGRID = dist_arr(NAXIS2, NAXIS1, CDELT1)
    squares = RGRID, data_square, err_square
    lists = [x.flatten() for x in squares]

    p = lists[0].argsort()
    R_list, data_list, err_list = [x[p] for x in lists]

    # Load in the theoretical model that matches the input cube (if it exists).
    if os.path.isfile("{0}_model.p".format(name)) == True:
        model = pickle.load( open( "{0}_model.p".format(name), "rb" ))
        I_mod_list = model['I_mod']
        S_mod_list = model['S_mod']
    else:
        I_mod_list = np.zeros(len(R_list))
        S_mod_list = np.zeros(len(R_list))

    # Finally construct a list of dictionaries. Each element of the list corresponds to one spaxel.
    data = []
    for i in range(len(R_list)):
        mdict = {}
        mdict['rad'] = R_list[i]
        mdict['spec'] = data_list[i]
        mdict['spec_err'] = err_list[i]
        mdict['I_mod'] = I_mod_list[i]
        mdict['S_mod'] = S_mod_list[i]

        data.append(mdict)

    # Create a wavelength array corresponding to the input datacube
    inheader = fits.getheader('{0}.fits'.format(name), 0)
    sample = inheader[u'CDELT3']
    CRVAL3 = inheader[u'CRVAL3']
    NAXIS3 = inheader[u'NAXIS3']

    in_wavel = np.zeros(NAXIS3)
    for i in range(0, NAXIS3):
        in_wavel[i] = (CRVAL3 + (i * sample))

    return data, wavel, in_wavel

def ph2lum(photons, d, sample, w, h):
    '''Convert from photon flux to solar luminosities.'''
    b =  sample * 1e4             # bin width in angstrom
    area = 932.46 * 1e4    # collecting area in cm
    t = 900 * 40            # NDIT * DIT - time in seconds

    flux = photons  * 10**7  / ((b * area * t) * 10**-6)
    return flux / (2.8784e-6 * (10/d)**2)

def dist_arr(size_y, size_x, spax):
    '''Function that generates array with distances from centre

    Inputs:

        - size: length of array in pixels
        - spax: spaxel size [mas/spaxel]

    Outputs:

        - distance_array: shape (size_y,size_x)

    '''

    arra = np.zeros((size_y,size_x),dtype=float)
    y, x = arra.shape
    x_inds, y_inds = np.ogrid[:y, :x]
    mid_x, mid_y = (sp.array(arra.shape[:2]) - 1) / float(2)
    n_arra = ((y_inds - mid_y) ** 2 + (x_inds - mid_x) ** 2) ** 0.5
    n_arra *= spax
    #Quick and dirty fix to ensure central value is tiny for Airy function to handle.
    r = np.where(n_arra < 1.E-15, 0.001, n_arra)
    return r