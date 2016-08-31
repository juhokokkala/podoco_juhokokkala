###############################################################################
# Copyright (C) 2016 Juho Kokkala
# This is part of Juho Kokkala's PoDoCo project.
#
# This file is licensed under the MIT License.
###############################################################################
"""
Script for testing the one-step-ahead forecast particle filter with the MCMC
results. Poisson Model 2 for incoming traffic.
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
stanfile = "../stan_models/output_ic_poisson_2.csv"
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

params0['sqrtQ_x'] = (np.sqrt(1 - np.exp(-2 * dt / lscale_local_mcmc)) *
                      s_local_mcmc)

params0['A_x'] = np.exp(-dt / lscale_local_mcmc)

## Read data
print("Reading data")

y_train = np.genfromtxt('../data/incoming_train.csv', delimiter=',')
predicted_raw = np.mean(y_train, axis=1)
y_test = np.genfromtxt('../data/incoming_test.csv', delimiter=',')

## Initialize arrays for storing results

T, days = y_test.shape
predicted = np.zeros((T, days))
rmse = np.zeros(days)
rmse_raw = np.zeros(days)

## The filtering loop
np.random.seed(3)

predicted[0, :] = np.mean(np.exp(params0['base'][0, :] +
                                 0.5 * s_local_mcmc**2))

for d in range(days):
    params = params0.copy()
    x, params, W = icf.pf_init(Nrep=100, params=params)
    params, W = icf.pf_update_poisson(y_test[0, d], x, params, W)

    for t in range(1, T):
        if t % 10 == 0:
            print("Filtering - day "+str(d)+" step "+str(t))
        predicted[t, d] = icf.predict_mean(x, params, W)
        x, params, W = icf.pf_step_poisson(y_test[t, d], x, params, W)


## Compute errors
rmse = np.sqrt(np.mean(np.square(y_test - predicted), axis=0))
rmse_raw = np.sqrt(np.mean(np.square(y_test - np.tile(predicted_raw[:, None],
                                                      (1, days))), axis=0))

## Save results
np.savez("results_poisson2.npz", rmse=rmse, predicted=predicted)

## Product a LaTeX tabular of the results
print("Day ", end='')
for i in range(days):
    print("& " + str(i + 1), end='')
print(" \\\\ \hline ")
print("Model RMSE ", end='')
for i in range(days):
    print("& " + '%.1f' % rmse[i], end='')
print(" \\\\ ")
print("Raw mean RMSE ", end='')
for i in range(days):
    print("& " + '%.1f' % rmse_raw[i], end='')
print(" \\\\ ")
