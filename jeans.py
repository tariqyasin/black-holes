import scipy as sp
import numpy as np
from scipy import interpolate
from scipy import integrate
import matplotlib.pyplot as plt

from modules.utility import trap
from modules.imbh_config import imbh_data
from modules.analytical_model import AnalyticalModel


class JeansModel:
    """Calculates the Jeans model for given NSC parameters.

    Args:
        r_eff (float): Effective radius.
        luminosity (float): Luminosity (solar masses)
        model_name (str): 'Hernquist' or 'Plummer' etc..
        width (float): A distance that dictates the granularity of the model. Should correspond to
               physical distance subtended by a single HARMONI pixel in units of parsecs.
        m2l (float): Mass-to-light ratio.
        mu (float): Black hole mass fraction (m_BH / m_NSC).

    Attributes:
        intensity (function): Function. I(r).
        weighted_moment (function): Function. The I(r) * 2nd_moment(r)
        radius (numpyp.array): The array of radii (parsecs) used to calculate the Jeans model.
    """

    def __init__(self, r_eff, luminosity, model_name, m2l, mu):
        self.model = AnalyticalModel(r_eff, luminosity, model_name)
        self.mom_func = self.calculate_moment(self.model.function.nu, self.model.function.j, m2l*luminosity, mu)

    def mass_enc(self, X, nu):
        """Returns probability of finding 1 unit of mass enclosed within radius X by
            integrating nu(r). The integration is done in a non-dimensionalised space x = r/a."""

        def integrand(x):
            """Special case for x=0 to prevent division by 0 error from nu(x)."""
            if x != 0:
                return 4 * sp.pi * x**2 * nu(x)
            else:
                return 0

        return integrate.quad(integrand, 0, X)[0]

    def calculate_moment(self, nu, j, nsc_mass, mu):

        """Calculate the intensity-weighted second moment.
        The integration is done in a non-dimensionalised space x = r/a

        Args:
            nu: the mass distribution as a function of radius.
            j: the light-intensity as a function of radius.
            nsc_mass: the mass of the NSC in solar masses.
            mu: the black hole mass fraction (m_bh / m_NSC)
        Returns:
            the function j(r) * sigma^2(r)

        """

        def d_phi(x):
            """Returns the derivative of the gravitational field phi."""
            return imbh_data['gravity'] * nsc_mass * (self.mass_enc(x, nu) + mu) / x**2

        def integrand(x):
            """Returns the integrand of Jean's equation."""
            return nu(x) * d_phi(x)

        # Form a table of the integrands of Jean's equation (y) against dimensionless radius (x).
        x_array = np.logspace(3.0, -7, 50000)
        y_array = np.asarray([integrand(x) for x in x_array])
        nu_array = np.asarray([nu(x) for x in x_array])
        j_array = np.asarray([j(x) for x in x_array])

        # Integrate repeatedly from r_lower = R to r_upper = infinite, where r_lower increments slowly downwards
        # from infinite to 0. The integral for each value of r_lower is the value of nu(R) * sigma_p^2(R).
        weighted_moment = integrate.cumtrapz(-y_array, x_array, initial=0)
        weighted_moment *= j_array / nu_array
        weighted_moment[0] = 1e-40    # Need non-zero value for interpolation in log space.

        # Interpolate in log space.
        log_x_array = np.log10(x_array)
        log_weighted_moment = np.log10(weighted_moment)
        linear_interpolation = interpolate.interp1d(log_x_array, log_weighted_moment, kind=u'linear',
                                                    bounds_error=False, fill_value=0)

        def weighted_moment_func(x):
            return 10**(linear_interpolation(np.log10(x)))

        return weighted_moment_func

    def plot(self):
        """Plot j and velocity dispersion against radius for a sample range of radii."""
        self.rad = np.logspace(-8, 2.1, 1000)

        plt.figure()
        plt.title('intensity')
        plt.semilogx(self.rad, self.model.function.j(self.rad))

        plt.figure()
        plt.title('velocity')
        plt.semilogx(self.rad, np.sqrt(self.mom_func(self.rad) / self.model.function.j(self.rad)))
        plt.show()


class ProjectedJeansModel(JeansModel):
    """Projects the JeansModel along the line-of-sight using Abel integrals.

    Args:
        width (float): The minimum value of radius is set to 0.01 * width to prevent divergence.

    Attributes:
        proj_I_func (function): The projected intensity as a function of radius.
        proj_mom_func (function): The projected intensity-weighted second moment as a function of radius.

    """
    def __init__(self, r_eff, luminosity, model_name, m2l, mu, width):
        JeansModel.__init__(self, r_eff, luminosity, model_name, m2l, mu)

        self.rad = np.linspace(0.01*width, 300, num=100000)
        proj_I = self.abel(self.rad, self.model.function.j)
        proj_mom = self.abel(self.rad, self.mom_func)

        def my_interp(func):
            return interpolate.interp1d(self.rad, func, kind=u'linear', bounds_error=False, fill_value=0)

        self.proj_I_func = my_interp(proj_I)
        self.proj_mom_func = my_interp(proj_mom)

    def abel(self, X, f):
        """Carries out abel projection integral of function f for radius R.
        Uses subsitution r = R*cosh(theta)."""

        def integrand(theta, R, f):
            x = X * np.cosh(theta)
            return 2 * X * f(x) * np.cosh(theta)

        lim = np.log(1 + np.sqrt((500/X)**2 - 1))    # Sets theta limit.
        return trap(integrand, 0, lim, 60, X, f)

    def plot(self):
        plt.figure()
        plt.plot(self.rad, self.proj_I_func(self.rad))
        plt.title('proj refactor intensity')
        plt.xlim(0, self.rad[-1])

        plt.figure()
        plt.plot(self.rad, np.sqrt(self.proj_mom_func(self.rad) / self.proj_I_func(self.rad)))
        plt.xlim(0, self.rad[-1])
        plt.title('proj refactor velocity')
        plt.show()


if __name__ == "__main__":
    # The below code is for testing.

    from modules.utility import parsec

    jeans_model = JeansModel(2, 5e6, 'plummer', 1, 0.1)
    jeans_model.plot()

    # print(parsec(0.8, 5e6))
    # jeans_model = ProjectedJeansModel(2, 5e6, 'plummer', 1, 0.1, parsec(0.8, 5e6))