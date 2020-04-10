class Explorer(object):
    """
  This class is a model for an arbitrary explorer model
  """

    def __init__(self, mass, parameters=None):  # we initialize with mass only
        # There used to be a gravity parameter, but it makes more sense to put it into EnvironmentalModel
        self.type = "N/A"  # type for each explorer
        self.mass = mass
        self.parameters = parameters  # parameters object, used for shadowing
        self.analytics = dict()
        self.objectivefx = [
            ['Distance', self.distance, 'min'],
            ['Time', self.time, 'min'],
            ['Energy', self.energy_expenditure, 'min']
        ]
        self.maxvelocity = 0.01 # a very small number non zero to prevent divide by infinity
        self.minenergy ={}

    def optimizevector(self, arg):
        if isinstance(arg, str):
            id = next((i for i, item in enumerate(self.objectivefx) if arg in item), None)
            if id == None:
                id = 2
                # if the wrong syntax is passed in
            vector = np.zeros(len(self.objectivefx))
            vector[id] = 1
        else:
            vector = np.array(arg)
        return vector

    def distance(self, path_length):
        return path_length

    def velocity(self, slope):
        pass  # should return something that's purely a function of slope

    def time(self, path_lengths, slopes):
        v = self.velocity(slopes)
        if (v == 0).any():
            logger.debug("WARNING, divide by zero velocity")
        return path_lengths / v

    def energyRate(self, path_length, slope, g):
        return 0  # this depends on velocity, time

    def energy_expenditure(self, path_lengths, slopes, g):
        return 0


class Astronaut(Explorer):  # Astronaut extends Explorer
    def __init__(self, mass, parameters=None):
        super(Astronaut, self).__init__(mass, parameters)
        self.type = 'Astronaut'
        self.maxvelocity = 1.6  # the maximum velocity is 1.6 from Marquez 2008
        self.minenergy = {  # Aaron's thesis page 50
            'Earth': lambda m: 1.504 * m + 53.298,
            'Moon': lambda m: 2.295 * m + 52.936
        }

    def velocity(self, slopes):
        if np.logical_or((slopes > 35), (slopes < -35)).any():
            logger.debug("WARNING, there are some slopes steeper than 35 degrees")

        # slope is in degrees, Marquez 2008
        v = np.piecewise(slopes,
                         [slopes <= -20, (slopes > -20) & (slopes <= -10), (slopes > -10) & (slopes <= 0),
                          (slopes > 0) & (slopes <= 6), (slopes > 6) & (slopes <= 15), slopes > 15],
                         [0.05, lambda slope: 0.095 * slope + 1.95, lambda slope: 0.06 * slope + 1.6,
                          lambda slope: -0.2 * slope + 1.6, lambda slope: -0.039 * slope + 0.634, 0.05])
        return v

    def slope_energy_cost(self, path_lengths, slopes, g):
        m = self.mass
        downhill = slopes < 0
        uphill = slopes >= 0
        work_dz = m * g * path_lengths * np.sin(slopes)
        energy_cost = np.empty(slopes.shape)
        energy_cost[downhill] = 2.4 * work_dz[downhill] * 0.3 ** (abs(np.degrees(slopes[downhill])) / 7.65)
        energy_cost[uphill] = 3.5 * work_dz[uphill]

        return energy_cost

    def level_energy_cost(self, path_lengths, slopes, v):
        m = self.mass
        w_level = (3.28 * m + 71.1) * (0.661 * np.cos(slopes) + 0.115 / v) * path_lengths
        return w_level

    def energy_expenditure(self, path_lengths, slopes_radians, g):
        """
        Metabolic Rate Equations for a Suited Astronaut
        From Santee, 2001
        """
        v = self.velocity(np.degrees(slopes_radians))
        slope_cost = self.slope_energy_cost(path_lengths, slopes_radians, g)
        level_cost = self.level_energy_cost(path_lengths, slopes_radians, v)
        total_cost = slope_cost + level_cost
        return total_cost, v

    def path_dl_slopes(self, path):
        x, y, z = path.xyz()
        res = path.em.resolution
        xy = res * np.column_stack((x, y))
        dxy = np.diff(xy, axis=0)
        dl = np.sqrt(np.sum(np.square(dxy), axis=1))
        dz = np.diff(z)
        dr = np.sqrt(dl**2+dz**2)
        slopes = np.arctan2(dz, dl)
        return dl, slopes, dr

    def path_time(self, path):
        dl, slopes, _ = self.path_dl_slopes(path)
        return self.time(dl, slopes)

    def path_energy_expenditure(self, path, g=9.81):
        dl, slopes, _  = self.path_dl_slopes(path)
        return self.energy_expenditure(dl, slopes, g)


import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.size'] = 14
slopes = np.linspace(-25, 25, 100)
a = Astronaut(80)
nrg = a.energyRate(np.ones_like(slopes), slopes, 9.81)/a.velocity(slopes)
plt.plot(slopes, nrg)
plt.xlabel('slope [degrees]')
plt.ylabel('Power [W]')
plt.title('Power output [Santee et al 2001], mass=80kg')
plt.show()
# print(min(nrg))
