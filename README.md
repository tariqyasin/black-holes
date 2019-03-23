# Intermediate Mass Black Holes with HARMONI

## Purpose

A pipeline created for my Master's research that does the following:
  1. Create an input datacube for HSIM representing a Nuclear Star Cluster (NSC) with a central black hole.
  2. Run the datacube through the HARMONI Simulation Pipeline (HSIM) 
  3. Recover a velocity dispersion profile from the HSIM output datacube.
  4. Use the recovered velocity dispersion profile to fit a black hole mass using Bayesian inference.

By running this pipeline for a large variety of NSC and black hole parameters I could make an early assessment of HARMONI's future performance in its search for intermediate mass black holes. Please note that due to the large amount of computation involved the pipeline takes a long time to run when using datacubes with HARMONI's full field of view. 

## Requirements

The pipeline is written in Python v2.7. Tested on MacOS 10.13 only, but should work on Linux and Windows. 

The required Python modules are:
- numpy 1.10.4
- scipy 1.1.0
- matplotlib 2.2.3
- astropy 2.0.12
- joblib 0.13.2
- lmfit 0.9.12

The code has been tested with the indicated package versions. In particular the pipeline is known not to work with some future versions of numpy.

## Files

The pipeline consists of a large range of files and folders. The main ones are listed below.

__`main.py`__:

Runs the entire pipeline. 

example:
python main.py data example3 0.10  5e6  1e7  10.0  1.0 4 1

This stores the output files of the pipeline into the folder 'data/example' (creating it if it doesnt exist).

Arguments:
  1. Where to store the output data
  2. Assign a name to the model
  3. Black hole mass fraction (mu) i.e. [black hole mass] / [NSC mass]
  4. Luminosity of NSC (solar luminosities)
  5. Distance to NSC (parsecs)
  6. Exposure time (hours)
  7. Mass to light ratio
  8. Spaxel size
  9. Number of processors to use

__`hsim/`__:

Within the `hsim` subfolder is a lightly edited version of the HARMONI Simulation Pipeline (written by Simon Zieleniewski). Please see `hsim/README.md` for details of its arguments.

__`jeans.py`__:

Contains a class that creates a Jeans model of a Nuclear Star Cluster. Contains a second class that projects the Jeans model onto a plane.

__`grid_model.py`__:

Contains a function that carries out pixel integration of the projected Jeans model onto a 2D grid.

__`input_cube.py`__:

Contains the function that produces a HARMONI ready input datacube.

__`modules/fit_tools/cubefit.py`__:

Contains a function to fit a line-of-sight velocity profile to the HARMONI output datacube.

__`modules/fit_tools/fit_mass.py`__:

Contains a function to fit a line-of-sight velocity profile to the HARMONI output datacube.




