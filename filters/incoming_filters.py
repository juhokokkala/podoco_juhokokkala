###############################################################################
# Copyright (C) 2016 Juho Kokkala
# This is part of Juho Kokkala's PoDoCo project.
#
# This file is licensed under the MIT License.
###############################################################################
"""
Particle filters for tracking the incoming traffic intensity.

See, the files script_test_poisson_1.py and script_test_negbin.py for
usage.
"""

import numpy as np
import resampling  # resampling (c) Roger R Labbe Jr (MIT License)
from scipy.special import gammaln

def pf_init(Nrep, params):
    """
    Initialize particle filter from MCMC samples.
    """
    for key in params.keys():
        params[key] = np.tile(params[key], Nrep)

    N = params['A_x'].shape[0]
    W = np.repeat(1/N, N)

    x = np.random.normal(params['base'][0, :],
                         params['sqrtQ_x'] / np.sqrt((1 - params['A_x']**2)))
    return x, params, W


def pf_update_poisson(y, x, params, W):
    """Update weights according to measurement"""
    logW = np.log(W) + y * np.log(np.exp(x)) - np.exp(x)
    W = np.exp(logW - np.max(logW))
    W = W / sum(W)

    return params, W


def pf_step_poisson(y, x, params, W, resample=True):
    """One step (measurement) of the particle filter, Poisson obs. model
    
    (Resample)
    Propagate the particles using the prior model,
    Update weights
    Remove the first elements of baselines
    """
    N = W.shape[0]
    if resample:
        ind = resampling.residual_resample(W)
        x = x[ind]
        params['base'] = params['base'][:, ind]
        params['sqrtQ_x'] = params['sqrtQ_x'][ind]
        params['A_x'] = params['A_x'][ind]
        W = np.repeat(1/N, N)

    x = np.random.normal(params['base'][1, :] + params['A_x'] *
                         (x - params['base'][0, :]), params['sqrtQ_x'])

    params = trim_base(params)

    params, W = pf_update_poisson(y, x, params, W)

    return x, params, W


def predict_mean(x, params, W):
    """Expected value of the next observation after the update step"""
    return np.sum(W * (np.exp(params['base'][1, :] + params['A_x'] *
                  (x - params['base'][0, :]) + 0.5 * params['sqrtQ_x']**2)))


def trim_base(params):
    """Cuts the first component of base"""
    params['base'] = params['base'][1:, :]
    return params


def pf_update_negbin(y, x, params, W):
    """Update weights per measurement, NegBin obs. model"""

    phi = np.exp(x) / (params['omega'] - 1)

    logW = (gammaln(y + phi) - gammaln(phi) +
            y * (np.log(params['omega'] - 1) - np.log(params['omega'])) -
            phi * (np.log(params['omega'])))

    W = np.exp(logW - np.max(logW))
    W = W / sum(W)

    return params, W


def pf_step_negbin(y, x, params, W, resample=True):
    """
    One step (measurement) of the particle filter, NegBin obs. model
    (Resample)
    Propagate the particles using the prior model,
    Update weights
    Remove the first elements of baselines
    """
    N = W.shape[0]
    if resample:
        ind = resampling.residual_resample(W)
        x = x[ind]
        params['base'] = params['base'][:, ind]
        params['sqrtQ_x'] = params['sqrtQ_x'][ind]
        params['A_x'] = params['A_x'][ind]
        params['omega'] = params['omega'][ind]
        W = np.repeat(1/N, N)

    x = np.random.normal(params['base'][1, :] + params['A_x'] *
                         (x - params['base'][0, :]), params['sqrtQ_x'])

    params = trim_base(params)

    params, W = pf_update_negbin(y, x, params, W)

    return x, params, W
