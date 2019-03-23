'''Module that contains the loop over wavelength channels for the
simulation pipeline.

Written by Simon Zieleniewski

Last updated 19-01-17

'''

import numpy as n
from scipy.interpolate import interp1d
from modules.frebin import *
from modules.Gaussians import Gauss2D
from modules.misc_utils import update_progress
from modules.psf_utils import create_Gausspsf_channel, create_psf_channel, create_instpsf
from modules.psf_convolution import psf_convolve
import multiprocessing as mp


def make_psf(chann, head, wave, l, AO, psfvars, adrval, newsize, outspax, adr_switch):
    '''Function to take input datacube and process it iteratively through
    each wavelength channel as follows:
    - Generate PSF array for given channel
    - Add effect of ADR to channel
    - Convolve cube channel with PSF
    - Frebin up to chosen output spaxel scale

    Inputs:

        chann: cube channel
        head: Datacube header
        wave: wavelength [um]
        l: iteration no.
        out_cube: Empty output cube
        AO: AO mode [LTAO, SCAO, Gaussian]
        psfvars: list containing [psfparams, psfspax, psfsize,
                                  [D,eps], res_jitter, seeing, user_psf,
                                  user_psflams]
        adrval: ADR value
        newsize: tuple containing (x_newsize, y_newsize) array sizes
        outspax: tuple containing (x, y) output spaxel scales
        adr_switch: On or OFF. Turns ADR effect on or off

    Output:

        cube: Processed cube
        head: Updated header
        inspax: Input spaxel scale (mas, mas)
        outspax: Output spaxel scale (mas, mas)

    '''

    #Create PSF channel
    #If user PSF and 2D image
    upsf = psfvars[-2]
    upsflams = psfvars[-1]
    if upsf != 'None' and upsflams == 'None':
        psf = upsf
    #If User PSF and 3D cube
    elif upsf != 'None' and upsflams != 'None':
        #Interpolate PSF cube
        interp = interp1d(upsflams, upsf, axis=0)
        psf = interp(wave)

    elif AO == 'LTAO' or AO == 'SCAO' or AO == 'GLAO':
        psf = create_psf_channel(psfvars[0], l, psfvars[1], (head['CDELT1'],head['CDELT2']),
                                 psfvars[2], psfvars[3], psfvars[4])

    elif AO == 'Gaussian':
        psf = create_Gausspsf_channel(wave, psfvars[5], psfvars[3], psfvars[4], psfvars[2], False,
                       psfvars[1], (head['CDELT1'],head['CDELT2']))

    else:
        print 'AO = ', AO
        raise ValueError('AO choice or user_PSF error!')

    #Create instrument PSF array
    instpsf = create_instpsf(psfvars[2], psfvars[1], outspax, (head['CDELT1'],head['CDELT2']))

    return psf, instpsf

def my_convolution(chann, head, newsize, outspax, psf, instpsf):

    #Convolve cube channel with PSF channel
    chann = psf_convolve(chann, psf)

    #Convolve cube channel with instrument PSF
    chann = psf_convolve(chann, instpsf)

    #Frebin datacube up to output spaxel scale
    newsize = (int(newsize[0]), int(newsize[1]))
    #chann *= (head['CDELT1']*head['CDELT2']*1.E-6)
    chann = frebin(chann, (newsize[0],newsize[1]), total=True)
    #chann /= (outspax[0]*outspax[1]*1.E-6)

    return chann

