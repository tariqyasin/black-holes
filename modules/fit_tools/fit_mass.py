import cPickle as pickle
import numpy as np
import sys

from joblib import Parallel, delayed
from astropy.io import fits
from scipy.integrate import trapz
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt

from chi_tools import calculate_chi, bin_model
from modules.con_model.model import create_model


def parallel_wrap(mu_frac, out_head):
    'mu_frac = mu / mu_input  , where mu_input is the mu of the input datacube.'

    ML = out_head['ML']
    mu = mu_frac * out_head['mu']

    import copy
    
    copy_out_head = copy.deepcopy(out_head)
    model = create_model(ML, mu, copy_out_head)
    model['mu_frac'] = mu_frac

    return model


def create_grid(out_head, nproc):
    'Creates a grid of theoretical models in {M_BH / ML , ML} space.'
    
    # Define mu_frac = mu / input_mu 
    # For each value of mu_frac a theoretical model will be created explicity (with ML = 1).
    mu_frac = np.linspace(0.0, 2, num=30)

    # For each of these values, calculate the theoretical model at the input value of ML.
    row = [parallel_wrap(x, out_head) for x in mu_frac]

    # Now shift each of these in the ML direction, to form a grid of chi.
    mu_frac_list = []
    ML_list = []
    S_mod_list = []

    for i in range(len(row)):
        mu_frac = row[i]['mu_frac']
        S_mod_1 = row[i]['S_mod']      

        for y in np.arange(0.30, 1.7, 0.01):
            ML = out_head['ML'] * y                 # Scale ML by a fraction y of input ML
            S_mod = S_mod_1 * np.sqrt(y)            # Scale the kinematic model by root y.

            mu_frac_list.append(mu_frac)
            ML_list.append(ML)
            S_mod_list.append(S_mod)

    print 'Model grid created'

    return mu_frac_list, ML_list, S_mod_list

def compare_data(data, rad_max, out_head, mu_frac_list, ML_list, S_mod_list):
    'Compares the data to the grid of models (defined by mu_frac_list, ML_list, S_mod_list'

    R, I, I_err, S, S_err, bin_array = data

    # Delete data past the radius rad_max.
    while R[-1] > rad_max:
        R = R[:-1]
        S = S[:-1]
        S_err = S_err[:-1]

    # Bin the models into the same annuli as the data, and calculate chi.
    chi_list = []

    for i in range(len(S_mod_list)):
        S_mod = S_mod_list[i]

        # Bin the model in the same way as the data.
        S_mod = bin_model(S_mod, bin_array)

        # Chop model to same size as data.
        S_mod = S_mod[:len(S)]

        # Calculate a value of chi by comparing data to each model in S_mod_list.
        chi = calculate_chi(S, S_err, S_mod)

        chi_list.append(chi)

    mu_frac = np.asarray(mu_frac_list)
    ML = np.asarray(ML_list)
    chi = np.asarray(chi_list)

    print 'Chi grid made.'

    # Convert the x-axis from (mu / mu_input) to (bh_mass / input_bh_mass)
    mbh_frac = mu_frac * (ML / out_head['ML'])

    return mbh_frac, ML, chi

def marginilise(x, chi):
    '''
    Marginilises the 2D array of chi-squared along the y-axis.
    
    Arguments:
            chi: a meshgrid of chi values.
            x: the x values corresponding to the x-axis of the meshgrid.abs
    Returns:
            prob_1d: array. The 1D probability distribution
            mean: mean of the distribution
            sigma: standard deviation of the distribution.
    '''

    likelihood = np.exp(- chi / 2)
    likelihood /= np.sum(likelihood)
    prob_1d = np.sum(likelihood, axis=0)

    # Take moments of the 1D probability distribution.
    norm = trapz(prob_1d, x)
    mean = trapz(prob_1d * x, x) / norm
    sigma_sqr = trapz(prob_1d * (x - mean)**2, x) / norm
    sigma = np.sqrt(sigma_sqr)

    return prob_1d, mean, sigma


def fit_mass(data_dir, name, nproc, user_psf='None'):

    cube_name = "{}/{}/{}".format(data_dir, name, name)

    # The outcube cube header contains all the keywords of the out_head dictionary used to make the input model.. 
    out_head = fits.getheader("{}_Reduced_cube.fits".format(cube_name), 0)
    out_head['USER_PSF'] = user_psf
    if user_psf != 'None':
        print "Used the following PSF for models:", user_psf
    
    # Make a grid of models.
    mu_frac_list, ML_list, S_mod_list = create_grid(out_head, nproc)

    # Load in fitted data
    data = pickle.load(open("{}/{}/{}_data.p".format(data_dir, name, name), "rb"))
    rad_max = 100

    # Compare the grid of models to chi (also transforms coordinates to {mbh_frac, ML}).
    mbh_frac, ML, chi = compare_data(data, rad_max, out_head, mu_frac_list, ML_list, S_mod_list)

    # Interpolate chi onto a fine grid. 
    rbfi = Rbf(mbh_frac, ML, chi, function='cubic', smooth=0)

    xi = np.arange(0.0, 2.5, 0.01)
    yi = np.arange(0.3, 1.7, 0.01)
    X, Y = np.meshgrid(xi, yi)
    Z = rbfi(X, Y)

    # Contour plot of chi with confidence intervals
    plt.figure()
    chi_min = np.amin(chi)
    levels = (chi_min + 2.30, chi_min + 6.17, chi_min + 11.83)
    CS = plt.contour(X, Y, Z, levels, colors='black')
    plt.plot(1, out_head['ML'], 'ro', label='input')
    plt.plot(mbh_frac[np.argmin(chi)], ML[np.argmin(chi)], 'bo', label = 'min. chi^2')
    plt.xlabel('BH_mass / input_BH_mass')
    plt.ylabel('ML')
    plt.legend()
    plt.savefig("{}/{}/contour.pdf".format(data_dir, name))

    # Marginilise ML and calculate mean and deviation.
    prob_1d, mean, sigma = marginilise(xi, Z)

    # Plot 1D probability distribution.
    plt.figure()
    plt.title('1D probability')
    plt.plot(xi, prob_1d)
    plt.xlabel('BH_mass / input_BH_mass')
    plt.ylabel('Probability')
    plt.savefig("{}/{}/1D.pdf".format(data_dir, name))

    # Calculate input black hole mass from starting params.
    input_mbh = out_head['ML'] * out_head['L'] * out_head['mu']
    time = out_head['NDIT'] / 4

    print 'Output folder:', cube_name
    print 'input_BH_mass:', input_mbh, "solar masses"
    print 'BH_mass / input_BH_mass = ', mean, "+-", sigma, '(1 SD)'

    file = open('{}/results.txt'.format(data_dir), 'a+')
    file.write("\n" )
    file.write( "%s" "   " "%.2f" "   " "%.2e" "   " "%.1e" "   " 
                "%.1f" "   "  "%.1f" "   " "%.0F" "   "
                "%.2e" "   " "%.2f" "   " "%.2f" "   "
                    % (name, out_head['mu'], out_head['L'], out_head['d'], 
                    time, out_head['ML'], out_head['CDELT1'],
                    input_mbh, mean, sigma, ))
    file.close()


if __name__ == "__main__":

    l = sys.argv
    print l
    nproc = int(l[2])
    data_dir = str(l[3])
    name = str(l[4])
    user_psf = str(l[5])

    fit_mass(data_dir, name, nproc, user_psf)


