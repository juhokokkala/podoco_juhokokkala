###############################################################################
# Copyright (C) 2016 Juho Kokkala
# This is part of Juho Kokkala's PoDoCo project.
#
# This file is licensed under the MIT License.
###############################################################################
"""
Script for testing the one-step-ahead forecast particle filter with the MCMC
results. Interfloor+Outgoing traffic.
"""


import numpy as np
import ifogfilter
# Relative import kludge
from os import chdir
chdir("../utils/")
import stan_utilities as stanu
chdir("../filters/")


## Read MCMC samples

print("Reading samples from Stan output")
stanfile = "../stan_models/output_ifog_negbin.csv"
mcmc_header, mcmc_data = stanu.read_output(stanfile)

## Convert MCMC samples to parameters
print("Converting MCMC samples to the parameter format of the filters")

dt = 5/(24*60)
params0 = {}

params0['base_if'] = mcmc_data[:,
                               mcmc_header.index('x_base_if.1'):
                               mcmc_header.index('x_base_if.159')+1].T

lscale_local_if = mcmc_data[:, mcmc_header.index('lscale_local_if')]
s_local_if = mcmc_data[:, mcmc_header.index('s_local_if')]

params0['omega_if'] = 1 / mcmc_data[:, mcmc_header.index('invomega_if')]

params0['sqrtQ_if'] = np.sqrt(1 - np.exp(-2 * dt / lscale_local_if)) \
                      * s_local_if

params0['A_if'] = np.exp(-dt / lscale_local_if)

params0['base_og'] = mcmc_data[:,
                               mcmc_header.index('x_base_og.1'):
                               mcmc_header.index('x_base_og.159')+1].T

lscale_local_og = mcmc_data[:, mcmc_header.index('lscale_local_og')]
s_local_og = mcmc_data[:, mcmc_header.index('s_local_og')]

params0['omega_og'] = 1 / mcmc_data[:, mcmc_header.index('invomega_og')]

params0['sqrtQ_og'] = np.sqrt(1 - np.exp(-2 * dt / lscale_local_og)) \
                      * s_local_og

params0['A_og'] = np.exp(-dt / lscale_local_og)


## Read data
print("Reading data")

y_train_ic = np.genfromtxt('../data/incoming_train.csv', delimiter=',')

y_train_if = np.genfromtxt('../data/interfloor_train.csv', delimiter=',')
predicted_if_raw = np.mean(y_train_if, axis=1)

y_train_og = np.genfromtxt('../data/outgoing_train.csv', delimiter=',')
predicted_og_raw = np.mean(y_train_og, axis=1)

y_test_ic = np.genfromtxt('../data/incoming_test.csv', delimiter=',')
y_test_if = np.genfromtxt('../data/interfloor_test.csv', delimiter=',')
y_test_og = np.genfromtxt('../data/outgoing_test.csv', delimiter=',')


## Find a distribution for initial people in building

Ninitmin = np.zeros(y_train_if.shape[1])
for d in range(y_train_if.shape[1]):
    net_departures = 0
    for t in range(y_train_if.shape[0]):
        Ninitmin[d] = max(Ninitmin[d], (y_train_if[t, d] + y_train_og[t, d] +
                                        net_departures))
        net_departures += y_train_og[t, d]
        net_departures -= y_train_ic[t, d]

    Ninitmin[d] = max(Ninitmin[d], -net_departures)

meanlogNinit = np.mean(np.log(Ninitmin + 0.5))
stdlogNinit = np.std(np.log(Ninitmin + 0.5))

## Initialize

T, days = y_test_ic.shape
filter_inbuilding = np.zeros((T-1, days))
predicted_if = np.zeros((T, days))
predicted_og = np.zeros((T, days))
rmse_if = np.zeros(days)
rmse_raw_if = np.zeros(days)
rmse_og = np.zeros(days)
rmse_raw_og = np.zeros(days)

## Filtering loop
np.random.seed(3)

print("filtering")
for d in range(days):
    params = params0.copy()
    filter = ifogfilter.IfogFilter(Nrep=1000, params=params, meanlogNinit=3,
                                   stdlogNinit=1)

    predicted_if[0, d], predicted_og[0, d] = filter.predict_mean()

    for t in range(T-1):
        filter_inbuilding[t, d] = np.sum(filter.W * filter.N_inbuilding)
        if t % 10 == 0:
            print("Filtering - day "+str(d)+" step "+str(t))
        filter.update_step(y_test_ic[t, d], y_test_if[t, d], y_test_og[t, d])
        filter.prediction_step()
        predicted_if[t+1, d], predicted_og[t+1, d] = filter.predict_mean()

## Compute errors
rmse_if = np.sqrt(np.mean(np.square(y_test_if - predicted_if), axis=0))
rmse_raw_if = np.sqrt(np.mean(np.square(y_test_if -
                                        np.tile(predicted_if_raw[:, None],
                                                (1, days))), axis=0))

rmse_og = np.sqrt(np.mean(np.square(y_test_og - predicted_og), axis=0))
rmse_raw_og = np.sqrt(np.mean(np.square(y_test_og -
                                        np.tile(predicted_og_raw[:, None],
                                                (1, days))), axis=0))
## Save results

np.savez("results_ifog.npz", rmse_if=rmse_if, predicted_if=predicted_if,
         rmse_og=rmse_og, predicted_og=predicted_og, rmse_raw_if=rmse_raw_if,
         rmse_raw_og=rmse_raw_og, filter_inbuilding=filter_inbuilding)
