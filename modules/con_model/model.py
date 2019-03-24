import numpy as np
from astropy.io import fits

from modules.con_model.src.TheSimulator_PSFs import psf_setup
from modules.con_model.src.TheSimulator_Loop import make_psf, my_convolution
from modules.utility import dist_arr
from modules.fit_tools.trim import trim
from grid_model import ModelGrid


def create_model(ML, mu, out_head):

    # Replace ML and mu in output cube header with the chosen values.
    out_head['ML'] = ML
    out_head['mu'] = mu
    model = ModelGrid(out_head)
    I_grid = model.I_grid
    moment_grid = model.moment_grid

    # Read convolution parameters from output cube header.
    user_PSF = out_head['USER_PSF']
    spax = (out_head['CDELT1'], out_head['CDELT2'])
    seeing = out_head['SEEING']
    AO = out_head['AOMODE']
    wavel_0 = out_head['WAVEL_0'] / 1e4
    jitter = out_head['RES_JITT']

    D = 37.
    eps = 0.3   

    head = fits.Header([('CDELT1', out_head['sqsize']), ('CDELT2', out_head['sqsize']), 
    						('NAXIS1', out_head['numsq']), ('NAXIS2', out_head['numsq'])])

    head_copy = head.copy()
    
    # Generate PSF parameters and prepare grids for convolution.
    cube, head_copy, psfspax, psfparams, psfsize, upsf, upsflams = psf_setup(I_grid, head_copy, wavel_0, spax, user_PSF, AO, seeing, [D,eps])
    cube, head, psfspax, psfparams, psfsize, upsf, upsflams = psf_setup(moment_grid, head, wavel_0, spax, user_PSF, AO, seeing, [D,eps])    

    # Generate PSFs
    psfvars = [psfparams, psfspax, psfsize, [D, eps], jitter, seeing, upsf, upsflams]
    newsize = (head['NAXIS1']*head['CDELT1']/float(spax[0]),head['NAXIS2']*head['CDELT2']/float(spax[1]))

    psf, instpsf = make_psf(I_grid, head, wavel_0, 0, AO, psfvars, 'None', newsize, spax, 'Off')

    # Convolve grids with PSFs
    I_grid = my_convolution(I_grid, head, newsize, spax, psf, instpsf)
    moment_grid = my_convolution(moment_grid, head, newsize, spax, psf, instpsf)
    
    velocity_grid = np.sqrt(moment_grid/I_grid)

    # Trim to HARMONII size
    I_grid = trim(I_grid, 213, 151)
    velocity_grid = trim(velocity_grid, 213, 151)


    # Convert from 2D array to lists with corresponding radius list R_list.
    naxis2, naxis1 = np.shape(I_grid)
    rad_grid = dist_arr(naxis2, naxis1, spax[0])
    squares = rad_grid, I_grid, velocity_grid

    # Flatten to lists, order by radius and convert to numpy arrays.
    lists = [x.flatten() for x in squares]
    p = lists[0].argsort()
    lists = [x[p] for x in lists]
    radii, I_mod, S_mod = [np.asarray(x) for x in lists]

    model = {'rad': radii, 'I_mod': I_mod, 'S_mod': S_mod, 'mu': mu, 'ML': ML}

    return model
