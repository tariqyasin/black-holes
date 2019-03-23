import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

from jeans import ProjectedJeansModel
from modules.utility import parsec, y_arr, x_arr, dist_arr, trap2d


class ModelGrid():
    """ A class to hold and plot the unconvolved grid models."""
    
    def __init__(self, nsc_params):
        self.radius_grid, self.I_grid, self.moment_grid, self.velocity_grid = make_grids(nsc_params)

    def make_flat(self):
        grids = self.radius_grid, self.I_grid, self.moment_grid, self.velocity_grid
        self.radius, self.I, self.moment, self.velocity = [x.flatten() for x in grids]

    def plot(self):
        try:
            self.radius
        except AttributeError:
            self.make_flat()

        plt.figure()
        plt.plot(self.radius, self.I)
        plt.xlim(0, self.radius[-1])
        plt.title('new intensity')

        plt.figure()
        plt.plot(self.radius, self.moment)
        plt.title('new moment')
        plt.xlim(0, self.radius[-1])
        plt.show()


def make_grids(nsc_params):
    """
    Creates a grid of size lat*vert squares centred on black hole. The black hole is at the centre of the
    central square. Explicit integration is carried out for a single quadrant, which is then reflected in
    the axis to obtain the whole grid.

   nsc_params (dictionary):
        ML : mass to light ratio
        L : luminosity
        profile  :  String. Either "plummer" or "hernquist". The mass and light profile of the NSC.
        mu  :   Black hole mass as a fraction of galaxy mass.
        a   :   Scale length of galaxy in pc.
        d   :   Distance to galaxy in Mpc.
        sqsize  :   Grid square side length in milliarcseconds.
        lat :   Width of image in grid squares - MUST BE ODD!
        vert:   Height of image in grid squares - MUST BE ODD!

   Returns:
       r_grid : A 2D array containing the radius of each pixel.
       I_grid : A 2D array containing the mass of each pixel in solar masses.
       moment_grid :  A 2D array containing the second moment of each pixel.
       velocity_grid : A data square containing sigma_p for each pixel in km/s.
    """

    numsq = nsc_params['numsq']
    w = parsec(nsc_params['sqsize'], nsc_params['d'])

    jeans = ProjectedJeansModel(nsc_params['a'], nsc_params['L'], nsc_params['profile'],
                                nsc_params['ML'], nsc_params['mu'], w)
    fun_I = jeans.proj_I_func
    fun_A = jeans.proj_mom_func

    # Create grids of radius, x and y.
    r_grid = dist_arr(numsq, numsq, w)
    x_grid = x_arr(numsq, w)
    y_grid = y_arr(numsq, w)

    def pixel_integration(func):
        """ Function that carries out pixel integration over a grid of pixels.

        Integrates func over a grid of pixels defined by x_grid and y_grid, by
        averaging the value of func at the corner of each grid.

        """

        out_grid = ((func(np.sqrt((x_grid - w/2)**2 + (y_grid - w/2)**2)) +
                     func(np.sqrt((x_grid + w/2)**2 + (y_grid - w/2)**2)) +
                     func(np.sqrt((x_grid - w/2)**2 + (y_grid + w/2)**2)) +
                     func(np.sqrt((x_grid + w/2)**2 + (y_grid + w/2)**2))) *
                    w**2 / 4)

        return out_grid

    I_grid = pixel_integration(fun_I)
    moment_grid = pixel_integration(fun_A)

    def central(f):
        """Integrates the function f over the central pixel of the grid.

        The central pixel of the grid contains large variations in velocity and intensity.
        Therefore the integral must be done with a higher accuracy. This function carries out
        the integral using the 2D trapezium rule, and assigns the result to the corresponding
        array element.

        """
        return 4 * (trap2d(cart, 0.01*w, w/2, 0, 0.01*w, 16, f) + trap2d(cart, 0, w/2, 0.01*w, w/2, 16, f))

    I_grid[(numsq-1)/2, (numsq-1)/2] = central(fun_I)
    moment_grid[(numsq-1)/2, (numsq-1)/2] = central(fun_A)

    velocity_grid = np.sqrt(moment_grid / I_grid)
    r_grid *= nsc_params['sqsize'] / w  # Convert from parsecs back to milliarcseconds.
    print "Grids complete!"

    return r_grid, I_grid, moment_grid, velocity_grid


def cart(x, y, f, *args):
    '''Converts a radial function f(r) to 2D cartesion coordinations f(x, y).'''
    r = sp.sqrt(x*x + y*y)
    return f(r, *args)

if __name__ == u"__main__":
    # This code below is only for testing.

    nsc_params = {}
    nsc_params['profile'] = 'plummer'
    nsc_params['L'] = 5e6
    nsc_params['ML'] = 1
    nsc_params['mu'] = 0.1
    nsc_params['a'] = 2
    nsc_params['d'] = 5e6
    nsc_params['sqsize'] = 0.8
    nsc_params['numsq'] = 21

    model = ModelGrid(nsc_params)
    model.plot()
