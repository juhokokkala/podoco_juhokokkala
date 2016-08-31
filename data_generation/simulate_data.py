###############################################################################
# Copyright (C) 2016 Juho Kokkala
# This is part of Juho Kokkala's PoDoCo project.
#
# This file is licensed under the MIT License.
###############################################################################
"""
Simulating example data from the 3-component traffic model.
"""


import numpy as np
import csv
import trafficmodel

# Relative import kludge
from os import chdir
chdir("../utils/")
import stan_utilities as stanu
chdir("../data_generation/")

## Set parameters
dt = 5 / (60 * 24)
N = 159
days = 18
params = {}
params['Ninit_mu'] = 0
params['Ninit_s'] = 1
params['ic_params'] = {}
params['ic_params']['mu'] = 0
params['ic_params']['lscale_base'] = 0.1
params['ic_params']['stdev_base'] = 3
params['ic_params']['lscale_local'] = 0.06
params['ic_params']['stdev_local'] = 0.15
params['ic_params']['omega'] = 3
params['if_params'] = params['ic_params'].copy()
params['if_params']['mu'] = np.log(1/12)
params['og_params'] = params['if_params'].copy()

tm = trafficmodel.TrafficModel(N, dt, params, days=days)

## Generate data
np.random.seed(1)

condition_ic = {}
condition_ic['t'] = np.array([0, 2/24, 3/24, (N-1)*dt])
condition_ic['x'] = np.array([np.log(1/12), np.log(10), np.log(5),
                              np.log(1/12)])
tm.ic_process.sample_baseline(condition_ic)

condition_og = {}
condition_og['t'] = np.array([0, 2/24, 11/24, (N-1)*dt])
condition_og['x'] = np.array([np.log(1/120), np.log(1/120), 0, 0])
tm.og_process.sample_baseline(condition_og)

tm.sample_data()

# Save baselines
np.savez("../data/groundtruth_baselines", x_base_ic=tm.ic_process.x_base,
         x_base_if=tm.if_process.x_base, x_base_og=tm.og_process.x_base)

## Write into Stan format and into CSVs
stanu.write_incoming_standata("../data/incoming.stan.data", dt,
                              tm.y_ic[:, :12])

stanu.write_3component_standata("../data/3component.stan.data", dt,
                                tm.y_ic[:, :12], tm.y_if[:, :12],
                                tm.y_og[:, :12])

np.savetxt("../data/incoming_train.csv", tm.y_ic[:, :12], fmt='%1u',
           delimiter=',')
np.savetxt("../data/incoming_test.csv", tm.y_ic[:, 12:], fmt='%1u',
           delimiter=',')

np.savetxt("../data/interfloor_train.csv", tm.y_if[:, :12], fmt='%1u',
           delimiter=',')
np.savetxt("../data/interfloor_test.csv", tm.y_if[:, 12:], fmt='%1u',
           delimiter=',')

np.savetxt("../data/outgoing_train.csv", tm.y_og[:, :12], fmt='%1u',
           delimiter=',')
np.savetxt("../data/outgoing_test.csv", tm.y_og[:, 12:], fmt='%1u',
           delimiter=',')
