import scipy as sp


class AnalyticalModel:
    def __init__(self, r_eff, luminosity, model_name):
        if model_name == 'plummer':
            self.function = Plummer(r_eff, luminosity)
        if model_name == 'hernquist':
            self.function = Hernquist(r_eff, luminosity)


class Plummer:
    def __init__(self, r_eff, luminosity):
        self.r_eff = r_eff
        self.luminosity = luminosity

    def nu(self, x):
        return (3/(4*sp.pi * self.r_eff**3)) * (1 + (x/self.r_eff)**2)**(-5/2)

    def j(self, x):
        return self.luminosity * (3/(4*sp.pi * self.r_eff**3)) * (1 + (x/self.r_eff)**2)**(-5/2)


class Hernquist:
    def __init__(self, r_eff, luminosity):
        self.r_eff = r_eff
        self.luminosity = luminosity

    def nu(self, x):
        return (1 / (2*sp.pi)) * (self.r_eff/x) / (self.r_eff + x)**3

    def j(self, x):
        return (self.luminosity / (2*sp.pi)) * (self.r_eff/x) / (self.r_eff + x)**3
