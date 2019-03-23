# python main.py data example3 0.10  5e6  1e7  10.0  1.0 4 1

import subprocess
import sys
import cPickle as pickle
import os
from astropy.io import fits
from shutil import copyfile

from input_cube import input_cube
from modules.con_model.model import create_model
from modules.fit_tools.cubefit import fit_data
from modules.fit_tools.fit_mass import fit_mass
from modules.imbh_config import spec_params, fixed_nsc_params


def hsim(nproc, name, time, spaxsize):
    ''' Run the input datacube cube "$name.fits" through the HSIM pipeline.
        See HSIM documentation for the meaning of the arguments use by hsim.py '''

    # Calculate number of integrations (NDIT) to give required exposure time.
    ndit = int(time * 4)

    args = nproc, name, ndit, spaxsize, spaxsize
    cmd = "python hsim/hsim.py -c -p {} {}.fits 900 {} K {} {} E-ELT LTAO 0.7 10 None 0 280 True 1 0 True True True True".format(*args)
    subprocess.call(cmd, shell='True')

if __name__ == u"__main__":

    l = sys.argv

    data_dir = str(l[1])                       # Data directory e.g. /home/data 
    model_name = str(l[2])                     # Model name e.g. test
    mu = float(l[3])
    luminosity = float(l[4])
    distance = float(l[5])
    time = float(l[6])
    m2l = float(l[7])
    spaxsize = float(l[8])
    nproc = int(l[9])

    # Create a directory for the new simulation: /home/data/test/
    model_dir = "{}/{}".format(data_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    name = "{}/{}/{}".format(data_dir, model_name, model_name)
    
    # Define NSC parameters. Use header format so it can be appended to datacubes.
    nsc_params = fits.Header()
    nsc_params.extend(fixed_nsc_params)
    nsc_params['L'] = luminosity
    nsc_params['ML'] = m2l
    nsc_params['mu'] = mu
    nsc_params['d'] = distance

    # Create input cube and run through HSIM.
    input_cube(name, nsc_params, spec_params, nproc=nproc)
    hsim(nproc, name, time, spaxsize)

    # Copy data files from HSIM's output directory to the Data directory ("data_dir/model_name")
    copyfile("hsim/Output_cubes/{}_Reduced_cube.fits".format(model_name), "{}_Reduced_cube.fits".format(name))
    copyfile("hsim/Output_cubes/{}_Transmission_cube.fits".format(model_name), "{}_Transmission_cube.fits".format(name))
    copyfile("hsim/Output_cubes/{}_Red_SNR_cube.fits".format(model_name), "{}_Red_SNR_cube.fits".format(name))
    copyfile("hsim/Output_cubes/{}_Noiseless_Object_cube.fits".format(model_name), "{}_Noiseless_Object_cube.fits".format(name))

    # Create the moment grids that perfectly match the input datacube.
    out_head = fits.getheader('{0}_Reduced_cube.fits'.format(name), 0)
    model = create_model(out_head['ML'], out_head['mu'], out_head)

    # Save to a file for comparison with the HSIM output datacubes during fit.
    print('Saving to file')
    pickle.dump(model, open("{0}_model.p".format(name), "wb"))

    # Fit a velocity profile and black hole mass to the data.
    fit_data(data_dir, model_name, 'real', 'gaussian')
    fit_mass(data_dir, model_name, nproc)
    print "Data complete."


# main('/Users/tariq/summer/data/example/example.fits', '/Users/tariq/summer/main/hsim/src/modules/../../Output_cubes/',
# 900, 40, 'K', (4.0, 4.0), 0.7, 10.0, 'E-ELT', 'None', 'LTAO', 0.0, 280.0, 
# True, True, 1.0, 0, 'True', 'True', 'True' ,'True', 114, 1)
