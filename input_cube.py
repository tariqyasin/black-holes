# This program forms the data cube. The height and width of the image must be
# an odd numbers of pixels so it is symmetric, as the black hole is at the
# centre of the central pixel.

import numpy as np
import time
from astropy.io import fits

from grid_model import ModelGrid
from modules.imbh_config import imbh_data

s_o_l = imbh_data['speed_of_light']


def input_cube(name, nsc_params, spec_params, nproc=1, make_flat='False'):
    '''
    This function creates a HARMONI ready input datacube as a FITS  files.

    Arguments:
    --------------------
        name :  The FITS file is saved as "$name.fits". A file containing moment grids is saved as "$name_save.p".
        nsc_params : The parameters of the NSC model. See mom_grid.py for definitions.

        spec_params : A dictionary of parameters that define the template spectrum. It has the following keywords:
                      -  'wavel_0':  Peak wavelength of the spectral line.
                      -  'delta':  Width of the spectrum in angstrom.
                      -  'sample': Sampling rate of the spectrum, in Angstrom.

        nproc : Integer. Numbers of CPU cores to be used to create spectra.
        make_flat : "True" or "False" (string) . If "True" makes a datacube with a flat spectral line.

    Output:
    -----------------------
        - Creates an HSIM ready input datacube called "$name.fits" .
        - Creates a cPickle file "$name_save.fits" containing the grids of zeroth (IGRID) and second moments (AGRID)
    '''

    file("{}.fits".format(name), "wb")    # Check the ouput path exist.

    model = ModelGrid(nsc_params)
    datacube, CRVAL3, sample = make_spectra(model.velocity_grid, spec_params, make_flat, nproc)
    datacube *= model.I_grid    # Scale the spectra by luminosity.

    # Convert from solar luminosities to incident flux.
    flux_ = flux(nsc_params['d'], nsc_params['sqsize'])
    datacube *= flux_

    # These FITS header keywords are required by HSIM. See the HSIM readme for meanings.
    numlam, numy, numx = np.shape(datacube)
    header = fits.Header([('SIMPLE', True), ('NAXIS', 3),
                          ('NAXIS1', numx), ('NAXIS2', numy), ('NAXIS3', numlam),
                          ('CTYPE1', 'x'), ('CTYPE2', 'y'), ('CTYPE3', 'wavelength'),
                          ('CUNIT1', 'mas'), ('CUNIT2', 'mas'), ('CUNIT3', 'angstroms'),
                          ('CDELT1', nsc_params['sqsize']), ('CDELT2', nsc_params['sqsize']), ('CDELT3', sample),
                          ('CRVAL3', CRVAL3), ('CRPIX3', 1),
                          ('FUNITS', 'erg/s/cm2/A/arcsec2'), ('SPECRES', 0.0)])

    # Add our model parameters.
    header.extend(nsc_params),
    header.extend([('wavel_0', spec_params['wavel_0'])])

    # Write the datacube and header to a FITS file.
    hdu = fits.PrimaryHDU(datacube, header)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto('{0}.fits'.format(name), overwrite=True)

    return


def make_spectra(SGRID, spec_params, make_flat, nproc):
    '''
    A function that creates a Gaussian spectrum for each element of a 2D moment grid SGRID. The continuum is 1.

    Arguments:
        SGRID: Grid of velocity dispersions.
        spec_params: Dictionary of parameters than determine spectrum
        nproc: number of processors to use in parallel.
        make_flat: if 'True' then the spectrum is completely flat.
    Returns:
        A datacube with axes [wavelength: y: x].
        CRVAL3: The starting wavelength of the spectrum (required by HSIM).
        sample: The sampling rate of the spectrum (Angstrom).
    '''

    t0 = time.time()

    # Spectrum parameters.
    delta = spec_params['delta']
    sample = spec_params['sample']       
    wavel_0 = spec_params['wavel_0']
    sigma_therm = spec_params['sigma_therm']
    numlam = int(delta / sample)   # Number of data points along the spectral axis.
    CRVAL3 = wavel_0 - delta/2     # CRVAL3 is the wavelength of the first element of the spectrum (in Angstrom).
    gamma = 20.8                   # Gamma is the spectral line strength (in spectral velocity space).

    # Create wavelength array (angstrom)
    wavel = np.zeros(numlam)
    for k in xrange(0, numlam):
        wavel[k] = CRVAL3 + (k * sample)

    numy, numx = np.shape(SGRID)

    def func(numlam, numy, SGRID_col, wavel_0, gamma, i, wavel):
        """Returns an array of spectra."""

        spec = np.zeros((numlam, numy))
        for idx, x in enumerate(SGRID_col):
            sigma = np.sqrt(x**2 + sigma_therm**2)        # Intrinsic dispersion.
            spec[:, idx] = 1 - gaussian(s_o_l * np.log(wavel), sigma, s_o_l * np.log(wavel_0), gamma)

        return spec

    T = [func(numlam, numy, SGRID[:, i], wavel_0, gamma, i, wavel) for i in xrange(0, numx)]

    # Form the 3D datacube from the spectra.
    datacube = np.zeros((numlam, numy, numx))
    for i in xrange(0, numx):
        datacube[:, :, i] = T[i]

    t1 = time.time()
    print 'Time taken to create spectrum:', t1-t0

    return datacube, CRVAL3, sample


def flux(d, sqsize):
    u''' flux at Earth of 1 Solar luminosity at a distance d in units J/s/m^2/arcsec^2 in the K band.'''
    return 2.8784e-6 * (10/d)**2 / (sqsize*sqsize)


def gaussian(u, sigma, u_0, A):
    'Returns a Gaussian spectral line with mean u_0, dispersion sigma, normalization A.'
    w = (u - u_0) / sigma
    alpha = (1 / np.sqrt(2 * np.pi)) * np.exp(- w**2 / 2)
    return (A / sigma) * alpha


if __name__ == "__main__":

    # This code is for testing purposes only.

    name = "../data"

    # Initialise parameters
    nsc_params = {}
    nsc_params['profile'] = 'plummer'
    nsc_params['L'] = 5e6
    nsc_params['ML'] = 1
    nsc_params['mu'] = 0.1
    nsc_params['a'] = 2
    nsc_params['d'] = 5e6
    nsc_params['sqsize'] = 4
    nsc_params['numsq'] = 21

    spec_params = {}
    spec_params['delta'] = 100
    spec_params['sample'] = 1.467
    spec_params['wavel_0'] = 22500
    spec_params['sigma_therm'] = 10.

    input_cube(name, nsc_params, spec_params, nproc=1, make_flat='False')
