###############################################################################
# Copyright (C) 2016 Juho Kokkala
# This is part of Juho Kokkala's PoDoCo project.
#
# This file is licensed under the MIT License.
###############################################################################
"""
Generating an illustration of Gaussian process regression for the report.
"""

import numpy as np
import GPy
from matplotlib import pyplot as plt
from matplotlib2tikz import save as tikz_save


## Set parameters
mu = 0
lscale = 0.1
stdev = 3

dt = 5 / (60*24)
N = 159
t = dt * np.arange(N)

## Sample
kernel = GPy.kern.Matern32(1, lengthscale=lscale, variance=stdev**2)
Sigma = kernel.K(t[:, None], t[:, None])
mu = np.zeros(N)

np.random.seed(1)
priorsamples = np.random.multivariate_normal(mu, Sigma, 10)

data_t = np.array([0, 2/24, 3/24, (N-1) * dt])
data_x = np.array([np.log(1/12), np.log(10), np.log(5), np.log(1/12)])

model = GPy.models.GPRegression(data_t[:, None], data_x[:, None], kernel)

model.Gaussian_noise.variance = 1e-15

posteriorsamples = model.posterior_samples_f(t[:, None], size=10)

## Plot
plt.cla()
plt.plot(24 * t, priorsamples.T, 'k-')
tikz_save("../report/fig/GPillustration_prior.tikz",
          figureheight='\\figureheight', figurewidth='\\figurewidth')

##
plt.cla()
plt.plot(24 * t, posteriorsamples, 'k-')
plt.ylim([-8, 8])

for t_ in data_t:
    plt.plot(24*np.array([t_, t_]), np.array([-8, 8]), 'k-')

tikz_save("../report/fig/GPillustration_posterior.tikz",
          figureheight='\\figureheight', figurewidth='\\figurewidth')
