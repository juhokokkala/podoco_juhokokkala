###############################################################################
# Copyright (C) 2016 Juho Kokkala
# This is part of Juho Kokkala's PoDoCo project.
#
# This file is licensed under the MIT License.
###############################################################################


import numpy as np
import GPy


class TrafficModel:
    """
    3-component traffic model (incoming,interfloor,outgoing)

    Parameters

    ----------

    N : int, must be positive

        The number of time intervals in a day

    dt : double, must be positive

        The length of the time interval

    params :  dict

        The parameters, contains the following keys

        Ninit_mu : double

            Daily initial population is floor(lognormal(Ninit_mu,Ninit_s))

        Ninit_s : double

            Daily initial population is floor(lognormal(Ninit_mu,Ninit_s))

        ic_params : dict

            Parameters of the incoming intensity process. Contains these keys:

            mu : double

                The mean level of the baseline log-intensity process

            lscale_base : double, must be positive

                The lengthscale of the baseline log-intensity process

            stdev_base : double, must be positive

                The standard deviation of the baseline log-intensity process

            omega : double, must be > 1.0

                The dispersion parameter of the negative-binomial

        if_params : dict

            Parameters of the interfloor intensity process. Same keys as
                ic_params.

        og_params : dict

            Parameters of the outgoing intensity process. Same keys as
                ic_params.

    days : int, default 0, must be nonnegative

        The number of days.

    """

    def __init__(self, N, dt, params, days=0):
        self.dt = dt
        """The number of days"""

        self.N = N
        """The number of intervals in a day"""

        self.params = params
        """Parameters of the model"""

        self.days = days
        """Number of days"""

        self.ic_process = IntensityProcess(N, dt, params['ic_params'], days)
        """The incoming intensity as an IntensityProcess"""

        self.if_process = IntensityProcess(N, dt, params['if_params'], days)
        """The interfloor intensity as an IntensityProcess"""

        self.og_process = IntensityProcess(N, dt, params['og_params'], days)
        """The outgoing intensity as an IntensityProcess"""

        self.y_ic = None
        """Counts of incoming passengers for each interval,day"""

        self.y_if = None
        """Counts of interfloor passengers for each interval,day"""

        self.y_og = None
        """Counts of outgoing passengers for each interval,day"""

        self.popul_init = None
        """Initial population of the building for each day"""

    def sample_baselines(self):
        """Sample realizations of the baseline intensities"""

        for process in [self.ic_process, self.if_process, self.og_process]:
            process.sample_baseline()

    def sample_locals(self):
        """
        Sample realizations of the daily intensities.
        Samples baselines if they do not exist.
        """

        for process in [self.ic_process, self.if_process, self.og_process]:
            process.sample_local()

    def sample_data(self):
        """Sample the count data. Samples intensities if they do not exist."""

        for process in [self.ic_process, self.if_process, self.og_process]:
            if process.x_local is None:
                process.sample_local()

        self.y_ic = np.random.negative_binomial(np.exp(self
                                                       .ic_process.x_local) /
                                                (self.ic_process
                                                 .params['omega'] - 1),
                                                1 / self.ic_process
                                                .params['omega'])

        self.popul_init = np.floor(np.random.lognormal(self.params['Ninit_mu'],
                                                       self.params['Ninit_s'],
                                                       size=self.days))

        self.y_if = np.zeros((self.N, self.days), dtype=int)
        self.y_og = np.zeros((self.N, self.days), dtype=int)

        for d in range(self.days):
            inbuilding = self.popul_init[d]

            for k in range(self.N):
                # Simulate
                rates = [np.exp(self.if_process.x_local[k, d]),
                         np.exp(self.og_process.x_local[k, d])]

                totalrate = np.sum(rates)
                pnotrip = np.exp(-totalrate)
                counts = np.random.multinomial(inbuilding,
                                               [(1-pnotrip)*rates[0]/totalrate,
                                                (1-pnotrip)*rates[1]/totalrate,
                                                pnotrip])

                self.y_if[k, d] = counts[0]
                self.y_og[k, d] = counts[1]

                # Update people in building
                inbuilding += self.y_ic[k, d] - self.y_og[k, d]


class IntensityProcess:
    """
    Intensity process for one traffic component.

    Parameters

    ----------

    N : int, must be positive

        The number of time intervals in a day

    dt : double, must be positive

        The length of the time interval

    params : dict

        Parameters of the process. Contains these keys:

        mu : double

            The mean level of the baseline log-intensity process

        lscale_base : double, must be positive

            The lengthscale of the baseline log-intensity process

        stdev_base : double, must be positive

            The standard deviation of the baseline log-intensity process

        omega : double, must be > 1.0

            The dispersion parameter of the negative-binomial

    days : int, default 0, must be nonnegative

        The number of days.

    x_base : np.ndarray, default None. 1 dimensional with size N.

        The baseline log-intensity.

    x_local : np.ndarray, default None. 2 dimensional with shape N x days.

        The log-intensity

    """

    def __init__(self, N, dt, params, days=0, x_base=None, x_local=None):
        self.dt = dt
        """The number of days"""

        self.N = N
        """The number of intervals in a day"""

        self.params = params
        """Parameters of the model"""

        self.days = days
        """Number of days"""

        self.x_base = x_base
        """The baseline log-intensity"""

        self.x_local = x_local
        """The log-intensity"""

        self.Cov = None
        """The covariance matrix of the baseline log-intensity"""

    def sample_baseline(self, condition=None):
        """Sample the baseline log-intensity"""

        if condition is not None:
            t = self.dt * np.arange(self.N)
            mu = self.params['mu']
            kernel = GPy.kern.Matern32(1,  lengthscale=self
                                       .params['lscale_base'],
                                       variance=self
                                       .params['stdev_base']**2)
            m = GPy.models.GPRegression(condition['t'][:, None],
                                        condition['x'][:, None] - mu,
                                        kernel)
            m.Gaussian_noise.variance = 1e-10
            self.x_base = m.posterior_samples_f(t[:, None], size=1).flatten()
            self.x_base += mu

            return

        if self.Cov is None:
            t = self.dt * np.arange(self.N)
            kernel = GPy.kern.Matern32(1, lengthscale=self
                                       .params['lscale_base'],
                                       variance=self
                                       .params['stdev_base']**2)
            self.Cov = kernel.K(t[:, None])

        self.x_base = np.random.multivariate_normal(np.tile(self.params['mu'],
                                                            self.N), self.Cov)

    def sample_local(self):
        """Sample the log-intensity. Samples baseline if it does not exist."""

        if self.x_base is None:
            self.sample_baseline()

        A = np.exp(-self.dt/self.params['lscale_local'])
        sqrtQ = self.params['stdev_local'] * np.sqrt(1 - A**2)

        self.x_local = np.zeros((self.N, self.days))
        self.x_local[0, :] = np.random.normal(self.x_base[0],
                                              self.params['stdev_local'],
                                              size=self.days)
        for k in range(1, self.N):
            self.x_local[k, :] = np.random.normal(self.x_base[k] +
                                                  A * (self.x_local[k-1, :] -
                                                       self.x_base[k-1]),
                                                  sqrtQ)
