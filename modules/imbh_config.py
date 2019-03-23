"""File that stores hardwired data and physical constants."""

imbh_data = {

    'gravity': 4.30117623e-3,             # Gravitational constant in units of km/s, pc, M_sun,
    'rad_msec': 4.84813681e-9,            # Number of radians in a milliarcsecond.
    'speed_of_light': 299792.458          # Speed of light in km/s.
    }

spec_params = {

    'sample': 0.2933,                  # Sampling rate (Angstrom)
    'wavel_0': 22500.,                 # Peak wavelength / Wavelength of model convolution (Angstrom)
    'delta': 50.,                      # Size of spectral axis (Angstrom)
    'sigma_therm': 10.,                # Thermal dispersion (km/s)
    }

fixed_nsc_params = {

    'profile': 'plummer',
    'a': 2.69,
    'numsq': 160,
    'sqsize': 0.8,
}