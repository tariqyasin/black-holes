'''
cubefit.py is a program that fits velocity dispersion and intensity to HSIM output datacubes.
For an HSIM input FITS file called name.fits, the following FITS files are required
                --------------------------------------------
                    name.fits 
                    name_Reduced_cube.fits
                    name_Transmission_cube.fits
                    name_Red_SNR_cube.fits
                    name_Noiseless_Object_cube.fits
                -----------------------------------------------
They must all be placed in the same directory.

Optional: If a cPickle file called "name_model.p" exists in the same directory containing the theoretical grids of moments.
- such a file is created when the input datacube is made.

The program does the following:
1. Prepares datacubes for analysis by converting from electrons to photons, trimming datacube to HARMONI's 
   field of view and ordering spaxels radially.
2. Bin annularly to a minimum SNR (or select a small sample of points)
3. Fit either a Gaussian or a Gauss-Hermite function to each spatial bin.
4. Extract dispersion and intensity from the fits

The program creates a cPickle file containing arrays of radius, fitted dispersion, fitted dispersion error, fitted intensity,
fitted intensity error. Search for 'pickle.dump' in the body of the program.

The program also makes plots of the fitted values.
'''

import sys
import os
import matplotlib.pyplot as plt
import math
import numpy as np
import cPickle as pickle

from binning import SNR_bin, reduce
from fitting import fit, func, extract, gauss_moments
from prepare_data import prepare_data


def fit_data(data_dir, file_name, mode, fit_method):

    name = '{}/{}/{}'.format(data_dir, file_name, file_name)

    if fit_method == 'gaussian':
        min_SNR = 40
    else:
        min_SNR = 80
    many_plots = 'True'

    if mode == 'real':
        noiseless = 'False'
    elif mode == 'noiseless':
        noiseless = 'True'
    elif mode == 'select':
        noiseless = 'True'
    else:
        sys.exit('Invalid "mode" given by user')

    # Convert the datacubes to a list of dictionaries. Each dictionary contains information
    # (e.g. radius, spectrum) for a single spaxel.
    data, wavel, in_wavel = prepare_data(name, noiseless)

    # For the real data, bin the spaxels to a minimum SNR. Each element of the data dictionary now represents a single annular bin.
    if mode == 'real':
        data = SNR_bin(data, min_SNR)
    # Bin data as above, but use the noiseless data (with the 'real' errors).
    elif mode == 'noiseless':
        data = SNR_bin(data, min_SNR)
    # From the noiseless data, just select num_noiseless individual spaxels, equally spaced in radius.
    elif mode == 'select':
        data = reduce(data, num_noiseless)

    # Set spectrum error to 1 if using noiseless data (so it is irrelevant in reduced chi fitting).
    if noiseless == True:
        for item in data:
            data['spec_err'] = np.ones(len(wavel))

    # Create dictionary of functions that are used to calculate second moment and partial derivatives of a Gauss-Hermite function.
    ghdict = gauss_moments()

    # THE FITTING
    num = len(data)
    for i, x in enumerate(data):

        out, params, perr = fit(wavel, x['spec'], x['spec_err'], in_wavel, fit_method)
        I, I_err, S, S_err = extract(out, fit_method, ghdict, in_wavel)

        newkeys = {'I': I, 'I_err': I_err, 'S': S, 'S_err': S_err, 'params': params, 'perr': perr}
        x.update(newkeys)

        #print i + 1, "out of", num, "fits complete!"

    # Convert fitted values in keys (and h4) to 1D arrays.
    keys = ['rad', 'S', 'S_err', 'S_mod', 'I', 'I_err', 'I_mod', 'j']
    rad, S, S_err, S_mod, I, I_err, I_mod, j = [np.asarray([x[attribute] for x in data]) for attribute in keys]
    h4 = np.asarray([x['params']['h4'] for x in data])

    # For noiseless data set the error in the fitted velocity equal to '1.0' (so it is irrelevant in Reduced Chi fitting).
    if noiseless == 'True':
        S_err = np.ones(len(S_err))

    # Store the fitted values to the data using pickle - this is the completed data!
    main_data = rad, I, I_err, S, S_err, j
    pickle.dump( main_data, open( "{0}_data.p".format(name), "wb" ) )

    # BELOW IS JUST ANALYTICS

    # Calculate average difference between theoretical model and the data (as a fraction of the model).
    I_diff = (I - I_mod)  / I_mod
    S_diff = (S - S_mod)  / S_mod
    print 'Average absolute difference in dispersion:', np.sum((S - S_mod)) / len(S)
    print 'Average h4 is:', np.sum(h4)/len(h4)

    # Check what fraction of fitted values are within two sigma of the theoretical model (should be about 95%.)
    within_err = 0
    for i in range(0, len(S)):
        if (S_mod[i] > (S[i] - 2*S_err[i])) and (S_mod[i] < (S[i] + 2*S_err[i])):
            within_err += 1

    print 'Fraction within error is:', within_err / float(len(rad))

    # PLOTTING

    # Plot fitted dispersion with errorbars for real data, and without errorbars for noiseless data.
    plt.figure()
    if noiseless == 'False':
        plt.errorbar(rad, S, yerr=S_err, ls='none')
    else: 
        plt.plot(rad, S, 'b-x')
    
    plt.plot(rad, S_mod, 'g-')
    plt.ylabel('Dispersion \n $kms^{-1}$')
    plt.xlim(0, rad[-1])
    plt.title(min_SNR)
    plt.savefig("{}/{}/velocity.pdf".format(data_dir, file_name))

    # Plot fractional difference in dispersion between theoretical model and fitted value.
    plt.figure()
    plt.plot(rad, S_diff, '.', markersize=2)
    plt.ylabel('Dispersion fractional difference')
    plt.xlabel('R / milliarcseconds')
    plt.xlim(0, rad[-1])
    # with Y = 0 line as a visual aid.
    Y = np.zeros(100)   
    X = np.linspace(0,1000, 100)
    plt.plot(X,Y)

    if many_plots == 'True':
        # Plot fractional difference in dispersion between theoretical model and fitted value.
        plt.figure()
        plt.plot(rad, I_diff, '.', markersize=2)
        plt.plot(X,Y)
        plt.ylabel('Intensity fractional difference')
        plt.xlabel('R / milliarcseconds')
        plt.xlim(0, rad[-1])

        # Plot h4 and h6.
        plt.figure()
        plt.plot(rad, h4, 'g.', markersize=3)
        #plt.plot(rad, h6, 'r.', markersize=3)
        plt.plot(X,Y)
        plt.ylabel('h4')
        plt.xlabel('R / milliarcseconds')
        plt.xlim(0, rad[-1])

        # Plot intensity
        plt.figure()
        plt.plot(rad, I)
        plt.ylabel('Intensity \n $L_{\odot} per pixel$')
        plt.xlim(0, rad[-1])

    # Calculate and plot chi.
    # chi = ((S - S_mod) / S_err)
    # plt.plot(R, chi, 'b.')
    # print np.sum(chi)

    # for i in range(len(data)):
    #     spec = data[i]['spec']
    #     spec_err =  data[i]['err']
    #     params = data[i]['params']
    #     plt.figure()
    #     plt.errorbar(wavel, spec, yerr=spec_err, marker='x', ls='none')
    #     fit = func(params, wavel, wavel_params)
    #     #gauss_fit = func(gauss_params, wavel, wavel_params)
    #     plt.plot(wavel, fit, 'r')
    #     #plt.plot(wavel, gauss_fit, 'g')
    #     plt.show(block=False)
    #     input()

if __name__ == "__main__":

    args = sys.argv

    # Define the name of HSIM output FITS files. e.g. for dir/cubename_Reduced_cube.fits, need name = 'dir/cubename'.
    data_dir = str(args[1])
    file_name = str(args[2])

    # Define method of fitting.
    mode = str(args[3])
    fit_method = str(args[4])
    if len(l) == 6:
        num_noiseless = int(args[5])

    fit_data(data_dir, file_name, mode, fit_method)
