###############################################################################
# Copyright (C) 2016 Juho Kokkala
# This is part of Juho Kokkala's PoDoCo project.
#
# This file is licensed under the MIT License.
###############################################################################
"""
Script for testing the one-step-ahead forecast particle filter with the MCMC
results. The NegBin Model for incoming traffic. An additional testday
generated by moving the arrivals 1 hour forward.
"""


import numpy as np
import incoming_filters as icf
# Relative import kludge
from os import chdir
chdir("../utils/")
import stan_utilities as stanu
chdir("../filters/")


## Read MCMC samples
print("Reading samples from Stan output")
stanfile = "../stan_models/output_ic_negbin.csv"
mcmc_header, mcmc_data = stanu.read_output(stanfile)

## Convert MCMC samples to parameters
print("Converting MCMC samples to the parameter format of the filters")

dt = 5/(24*60)
params0 = {}

params0['base'] = mcmc_data[:,
                            mcmc_header.index('x_base.1'):
                            mcmc_header.index('x_base.159')+1].T

lscale_local_mcmc = mcmc_data[:, mcmc_header.index('lscale_local')]
s_local_mcmc = mcmc_data[:, mcmc_header.index('s_local')]

params0['omega'] = 1 / mcmc_data[:, mcmc_header.index('invomega')]

params0['sqrtQ_x'] = (np.sqrt(1 - np.exp(-2 * dt / lscale_local_mcmc)) *
                      s_local_mcmc)

params0['A_x'] = np.exp(-dt / lscale_local_mcmc)

## Read data
print("Reading data")

y_train = np.genfromtxt('../data/incoming_train.csv', delimiter=',')
predicted_raw = np.mean(y_train, axis=1)
y_test = np.genfromtxt('../data/incoming_test.csv', delimiter=',')

## Editing the testday
T = y_test.shape[0]
y_test2 = y_test[:, 0].copy()
y_test2[12:] = y_test[:-12, 0]
y_test2[:12] = 0

## The filtering loop
np.random.seed(3)

predicted = np.zeros(159)

predicted[0] = np.mean(np.exp(params0['base'][0, :] +
                              0.5 * s_local_mcmc**2))

params = params0.copy()
x, params, W = icf.pf_init(Nrep=100, params=params)

predicted[0] = np.mean(np.exp(params0['base'][0, :] +
                              0.5 * s_local_mcmc**2))
params, W = icf.pf_update_negbin(y_test2[0], x, params, W)

for t in range(1, T):
    if t % 10 == 0:
        print("Filtering - step "+str(t))
    predicted[t] = icf.predict_mean(x, params, W)
    x, params, W = icf.pf_step_negbin(y_test2[t], x, params, W)

## Compute errors
rmse = np.sqrt(np.mean(np.square(y_test2 - predicted)))
rmse_raw = np.sqrt(np.mean(np.square(y_test2 - predicted_raw)))

## PLOT
t = (1/12) * np.arange(159)
plt.cla()
plt.plot(t, y_test2, 'rx--', markersize=2)

plt.plot(t, predicted_raw, 'g--.')
plt.plot(t, predicted, 'b.-', markersize=2)
plt.xlabel("Time (h)")
plt.ylabel("Arrivals per 5 min interval")

plt.legend(["Observations", "Raw mean", "Model one-step-ahead forecast"])

plt.show()

tikz_save('../report/fig/ic_shiftedday.tikz',
          figureheight='\\figureheight',
          figurewidth='\\figurewidth')