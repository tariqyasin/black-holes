import scipy as sp
import numpy as np
import math

from lmfit import minimize, Parameters
from scipy.ndimage.filters import convolve1d
from scipy.interpolate import interp1d
import scipy.integrate as integrate
import matplotlib.pyplot as plt

from frebin import frebin
from Gaussians import Gauss

sol = 299792.458    # Speed of light in km/s.
c = 299792.458

def fit(wavel, spec, spec_err, in_wavel, fit_method):
    '''
    Arguments:
        wavel:      wavelength array
        spec:       the spectrum
        spec_err:   the spectrum error
        in_wavel:   the input cube wavelength array
        fit_method: string. function to fit. 'gaussian' or 'gauss_herm'
        
    Returns:
        out:        the lmfit minimizer object
        params:     a dictionary of the fitted parameters
        perr:       an array of the errors in the fitted parameters.
    '''

    def residual(params, wavel, spec, spec_err, in_wavel):
        model = func(params, wavel, in_wavel)
        return (spec-model) / spec_err

    if fit_method == 'gaussian':
        # Just fit a Gaussian
        params = Parameters()

        params.add('slope', value=0, vary=True)
        params.add('intersect', value=spec[0])

        params.add('mean', value=22500, vary=True)
        params.add('sigma_v', value=30, vary=True, min= 5.0, max=200)
        params.add('gamma', value=1.56106, vary=True, min=1.0)

        params.add('h0', value=1, vary=False)
        params.add('h1', value=0, vary=False)
        params.add('h2', value=0, vary=False)
        params.add('h3', value=0, vary=False)
        params.add('h4', value=0, vary=False, max=0.4)   # See Ryan's thesis.
        params.add('h6', value=0, vary=False)
        params.add('h8', value=0, vary=False)

    elif fit_method == 'gauss_herm':
        # Fit a Gauss-Hermite function.
        params = Parameters()

        params.add('slope', value=0, vary=True)
        params.add('intersect', value=spec[0])

        params.add('mean', value=22500, vary=True, min=0.0, max=22520)
        params.add('sigma_v', value=20, vary=True, min=5.0, max=150)
        params.add('gamma', value=1.56106, vary=True, min=1.0)

        params.add('h0', value=1, vary=False)
        params.add('h1', value=0, vary=False)
        params.add('h2', value=0, vary=False)
        params.add('h3', value=0, vary=False)
        params.add('h4', value=0, vary=False)  
        params.add('h6', value=0, vary=False)
        params.add('h8', value=0, vary=False)

        out = minimize(residual, params, args=(wavel, spec, spec_err, in_wavel))
        gauss_params = out.params.valuesdict()

        # Use Gaussian parameters as the fixed mean, sigma_v and gamma for Gauss-Hermite series.
        params = Parameters()

        params.add('slope', value=gauss_params['slope'], vary=False)
        params.add('intersect', value=gauss_params['intersect'], vary=False)

        params.add('mean', value=gauss_params['mean'], vary=False)
        params.add('sigma_v', value=gauss_params['sigma_v'], vary=False)
        params.add('gamma', value=gauss_params['gamma'], vary=False)

        # Symmetric only, up to h4.
        params.add('h0', value=1, vary=False)
        params.add('h1', value=0, vary=False)
        params.add('h2', value=0, vary=False)
        params.add('h3', value=0, vary=False)
        params.add('h4', value=0, vary=True)
        params.add('h6', value=0, vary=False)
        params.add('h8', value=0, vary=False)

    elif fit_method == 'two_step':
        # Two step fitting - first fit a Gaussian. Then fit h0, h2, h4 (see Ryan's thesis).
        params = Parameters()

        params.add('slope', value=0, vary=True)
        params.add('intersect', value=spec[0])

        params.add('mean', value=22500, vary=True, min=0.0, max=22520)
        params.add('sigma_v', value=20, vary=True, min=5.0, max=150)
        params.add('gamma', value=1.56106, vary=True, min=1.0)

        params.add('h0', value=1, vary=False)
        params.add('h1', value=0, vary=False)
        params.add('h2', value=0, vary=False)
        params.add('h3', value=0, vary=False)
        params.add('h4', value=0, vary=False)  
        params.add('h6', value=0, vary=False)
        params.add('h8', value=0, vary=False)

        out = minimize(residual, params, args=(wavel, spec, spec_err, in_wavel))
        gauss_params = out.params.valuesdict()

        # Use Gaussian parameters as the fixed mean, sigma_v and gamma for Gauss-Hermite series.
        params = Parameters()

        params.add('slope', value=gauss_params['slope'], vary=False)
        params.add('intersect', value=gauss_params['intersect'], vary=False)

        params.add('mean', value=gauss_params['mean'], vary=False)
        params.add('sigma_v', value=gauss_params['sigma_v'], vary=False)
        params.add('gamma', value=gauss_params['gamma'], vary=False)

        # Symmetric only, up to h4.
        params.add('h0', value=1, vary=True)
        params.add('h1', value=0, vary=False)
        params.add('h2', value=0, vary=True)
        params.add('h3', value=0, vary=False)
        params.add('h4', value=0, vary=True)
        params.add('h6', value=0, vary=False)
        params.add('h8', value=0, vary=False)

    else:
        sys.exit('Invalid fit_method given by user - must be "gaussian", "estimate", "fixed, "two_step"')

    out = minimize(residual, params, args=(wavel, spec, spec_err, in_wavel))

    params = out.params.valuesdict()
    pcov = out.covar
    perr = np.sqrt(np.diag(pcov))

    # PLOT FITTED SPECTRUM
    #print 'Gaussian parameters:', gauss_params
    #print 'Parameters:', params
    #plt.figure()
    #plt.errorbar(wavel, spec, yerr=spec_err, marker='x', ls='none')
    #fit = func(params, wavel, in_wavel)
    #gauss_fit = func(gauss_params, wavel, in_wavel)
    #plt.plot(wavel, fit, 'r')
    #plt.plot(wavel, gauss_fit, 'g')
    #plt.show(block=False)
    #input()
    #print params['sigma_v']
    #
    # if params['sigma_v'] > 100:
    #     fit = func(params, wavel, in_wavel)
    #     plt.errorbar(wavel, spec, yerr=spec_err, marker='x', ls='none')
    #     plt.plot(wavel, fit, 'r')
    #     import sys
    #     sys.exit()

    return out, params, perr

def extract(out, fit_method, ghdict, in_wavel):
    '''
    This function extracts the correct second moment and intensity from the fitted Gauss-Hermite velocity profile
    using the dictionary of functions ghdict.
    '''

    params = out.params.valuesdict()
    var_names = out.var_names
    pcov = out.covar
    perr = np.sqrt(np.diag(pcov))

    # Calculate the intensity at the mean.
    I = params['slope'] * params['mean'] + params['intersect']

    calc_sigma_bar = ghdict['sigmabar']
    diff_h0 = ghdict['diff_h0']
    diff_h2 = ghdict['diff_h2']
    diff_h4 = ghdict['diff_h4']
    diff_sigma = ghdict['diff_sigma']

    # Calculate correct second moment.
    if fit_method == 'gaussian':
        sigma_v = params['sigma_v']
        sigma_index = var_names.index("sigma_v")
        sigma_v_err = perr[sigma_index]
    else:
        sigma_v = calculate_dispersion(params, in_wavel, max=True)
        #sigma_v = calc_sigma_bar(params['sigma_v'], params['h0'], params['h2'], params['h4'])

    # Extract errors for Gauss-Hermite fits using covariance matrix.
    if fit_method == 'gauss_herm':
        h4_index = var_names.index("h4")

        diff_h4_val = diff_h4(params['sigma_v'], params['h0'], params['h2'], params['h4'])
        h4_term = perr[h4_index] * diff_h4_val

        sigma_v_err = h4_term
    
    if fit_method == 'two_step':
        sigma_v = calc_sigma_bar(params['sigma_v'], params['h0'], params['h2'], params['h4'])
        diff_h0_val = diff_h0(params['sigma_v'], params['h0'], params['h2'], params['h4'])
        diff_h2_val = diff_h2(params['sigma_v'], params['h0'], params['h2'], params['h4'])
        diff_h4_val = diff_h4(params['sigma_v'], params['h0'], params['h2'], params['h4'])
       
        # Diff is the vector containing the derivatives of the second moment with respect to the Gauss-hermite coefficients.
        diff = np.array([diff_h0_val, diff_h2_val, diff_h4_val])
        prod = diff.dot(pcov)
        diff = prod.dot(diff.transpose())

        sigma_v_err = np.sqrt(diff)
    
    if fit_method == 'gauss_herm2':
        sigma_index = var_names.index("sigma_v")
        h4_index = var_names.index("h4")

        diff_sigma_val = diff_sigma(params['sigma_v'], params['h0'], params['h2'], params['h4'])
        diff_h4_val = diff_h4(params['sigma_v'], params['h0'], params['h2'], params['h4'])
        sig_term = perr[sigma_index] * diff_sigma_val
        h4_term = perr[h4_index] * diff_h4_val
        cov_term = 2 * pcov[sigma_index,h4_index] * diff_sigma_val * diff_h4_val

        sigma_v_err = sp.sqrt(sig_term**2 + h4_term**2 + cov_term)

    # The error in I isn't important.
    I_err = 0
    
    #print('dispersion is', sigma_v, sigma_v_err)

    return I, I_err, sigma_v, sigma_v_err

def func(params, new_wavels, in_wavel):
    ''' 
    The function we are fitting to the actual spectrum. It is the velocity profile convolved with the stellar spectrum and HARMONI's LSF,
    and then rebinned (using the "frebin" function) to the output sampling of HSIM.

    Arguments:
       params: Parameters of the sloped Gauss-Hermite function.
       new_wavels: An array of wavelengths corresponding to the output pixel bins of HSIM.
       in_wavel: An array of wavelengths corresponding to the pixel bins of my HSIM input cube.
    '''

    # Add thermal dispersion in quadrature.
    sigma_v = np.sqrt(params['sigma_v']**2 + 10**2)   
    mean = params['mean']
    gamma = params['gamma']
    hvals = params['h0'], params['h1'], params['h2'], params['h3'], params['h4'], params['h6'], params['h8']

    # Evaluate the gauss_hermite function at wavelengths corresponding to the input cube pixel bins.
    gauss_herm = gauss_hermite(c * np.log(in_wavel), c * np.log(mean), sigma_v, gamma * c / 22500, hvals)
    spec = 1 - gauss_herm

    # Convolve spectrum with HARMONI's line-spread function (R = 7500).
    specres = 2.93333333333         
    sig = specres /(2.*np.sqrt(2.*np.log(2.)))
    gauss_array = Gauss(sig, (in_wavel[1] - in_wavel[0]))
    spec = convolve1d(spec, gauss_array[:, 1], axis=0)

    # Rebin onto pixel bins of the HSIM output cube using HSIM's "frebin" function.
    array_in = np.zeros((len(spec), 1))
    array_in[:, 0] = spec
    array_out = frebin(array_in, (1, len(new_wavels)), True)
    spec_new = array_out[:, 0]

    # Rescale flux to account for new bin width.
    spec_new /= ((new_wavels[1] - new_wavels[0]) / (in_wavel[1] - in_wavel[0]))

    # Allow the spectrum to be sloped (as the spatial PSF is wavelength dependent).
    spec_new *= (params['slope']*new_wavels + params['intersect'])

    return spec_new

def gauss_hermite(wavel, mean, sigma, gamma, hvals):
    'A Gauss hermite function.'
    h0, h1, h2, h3, h4, h6, h8 = hvals[0], hvals[1], hvals[2], hvals[3], hvals[4], hvals[5], hvals[6]

    w = (wavel - mean) / sigma
    alpha = (1 / sp.sqrt(2 * sp.pi)) * sp.exp(- w**2 / 2)

    def h1_func(y):
        return sp.sqrt(2) * y
    def h2_func(y):
        return (1/sp.sqrt(2)) * (2*y**2 - 1)
    def h3_func(y):
        return(1/sp.sqrt(6)) * ((2*sp.sqrt(2)*y**3) - (3*sp.sqrt(2)*y))
    def h4_func(y):
        return (1/sp.sqrt(24)) * (4*y**4 - 12*y**2 + 3)
    def h6_func(y):
        return (1 / sp.sqrt(2**6 * 720)) * (64*y**6 - 480*y**4 + 720*y**2 - 120)
    def h8_func(y):
        return (1/ sp.sqrt(2**8 * 40320)) * (256*y**8 - 3584*y**6 + 13440*y**4 - 13440*y**2 + 1680)

    herm = h0 + (h1 * h1_func(w)) + (h2 * h2_func(w)) + (h3 * h3_func(w)) + (h4 * h4_func(w)) + (h6 * h6_func(w)) + (h8 * h8_func(w))

    gauss_herm = ((herm) * (gamma / sigma) * alpha)

    return gauss_herm


from sympy import sqrt, symbols, init_printing, diff
from sympy.utilities.lambdify import lambdify, implemented_function
from sympy import Function

def gauss_moments():
    '''
    A function that returns a dictionary of functions used to calculate the dispersion of
    Gauss-Hermite functions.

    ghdict['sigmabar'] : calculates the moment.
    ghdict['diff_h0'] : calculates the derivative of the moment with respect to h0.
    ghdict['diff_sigma'] : calculates the derivative of the moment with respect to sigma_v.

    These derivatives are used to calculate the errors from the covariance matrix.
    '''

    sigma, h0, h2, h4 = symbols('sigma h0 h2 h4')

    # Note: 0.707106781187 = 1 / sqrt(2)
    norm = h0 + (0.707106781187 * h2) + (0.25 * sqrt(6) * h4)
    second_mom = h0 + (5 * 0.707106781187 * h2 ) + (9 * 0.25 * sqrt(6) * h4) 
    sigma_bar =  sigma * sqrt(second_mom / norm)

    # Take symbolic partial derivatives
    deriv_h0 = diff(sigma_bar, h0)
    deriv_h2 = diff(sigma_bar, h2)
    deriv_h4 = diff(sigma_bar, h4)
    deriv_sigma = diff(sigma_bar, sigma)

    ghdict = {}

    # Convert to numerical function
    ghdict['sigmabar'] = lambdify((sigma, h0, h2, h4), sigma_bar)
    ghdict['diff_h0'] = lambdify((sigma, h0, h2, h4), deriv_h0)
    ghdict['diff_h2'] = lambdify((sigma, h0, h2, h4), deriv_h2)
    ghdict['diff_h4'] = lambdify((sigma, h0, h2, h4), deriv_h4)
    ghdict['diff_sigma'] = lambdify((sigma, h0, h2, h4), deriv_sigma)

    return ghdict

def calculate_dispersion(params, in_wavel, max=True):
    
    x = np.linspace(in_wavel[0], in_wavel[-2], 10000)

    # Create gauss_hermite line profile (in spectral velocity). The addition to sigma_v.
    mean = params['mean']
    sigma_v = np.sqrt(params['sigma_v']**2)
    gamma = params['gamma']
    hvals = params['h0'], params['h1'], params['h2'], params['h3'], params['h4'], params['h6'], params['h8']

    spec_fit = gauss_hermite(c * np.log(x), c * np.log(mean), sigma_v, gamma * c / 22500, hvals)

    # Ignore negative wings?
    if max==True:
        spec_fit[spec_fit < 0] = 0

    def dispersion(x, y, mean):
        norm = integrate.trapz(y, x)
        print 'norm is', norm

        def second_moment(x, y, norm, mean):
            return (x - mean)**2 * y / norm

        y2 = second_moment(x, y, norm, mean)

        second_moment = integrate.trapz(y2, x)
        return np.sqrt(second_moment)

    disp_fit = dispersion(x, spec_fit, params['mean'])

    return disp_fit * sol/params['mean']
