###############################################################################
# Copyright (C) 2016 Juho Kokkala
# This is part of Juho Kokkala's PoDoCo project.
#
# This file is licensed under the MIT License.
###############################################################################
"""
Particle filter for tracking the interfloor+outgoing traffic intensities.

See script_test_ifog.py for usage.

Parameters
----------

Nrep - int, number of replications of the initial parameters
params - dict containing the following MCMC samples (np arrays)
         base_if/base_og, TxM baseline log-intensities
         omega_if/omega_og, M - dispersion parameter
         sqrtQ_if/sqrtQ_og, M - error std of the local variation as an AR(1)
         A_if/A_og,         M - transformation parameter of the local variation
                                as an AR(1)
meanlogNinit - float, initial population ~ logNormal(meanlogNinit,stdlogNinit)
stdlogNinit - float, initial population ~ logNormal(meanlogNinit,stdlogNinit)
"""

import numpy as np
import resampling  # resampling (c) Roger R Labbe Jr (MIT License)
from scipy.special import gammaln


class IfogFilter:


    def __init__(self, Nrep, params, meanlogNinit, stdlogNinit):
        for key in params.keys():
            setattr(self, key, np.tile(params[key], Nrep))

        self.N = self.A_if.shape[0]
        """Number of particles"""
        
        self.W = np.repeat(1/self.N, self.N)
        """Weights"""
        
        self.x_if = np.random.normal(self.base_if[0, :],
                                     self.sqrtQ_if / np.sqrt(1 - self.A_if**2))
        """The samples of current interfloor log-intensity"""
                                     
        self.x_og = np.random.normal(self.base_og[0, :],
                                     self.sqrtQ_og / np.sqrt(1 - self.A_og**2))
        """The samples of current outgoing log-intensity"""
        
        self.N_inbuilding = np.floor(np.random.lognormal(meanlogNinit,
                                                         stdlogNinit,
                                                         size=self.N))
        """The samples of current population"""

        """self.p_trip - per-passenger trip probabilities of each particle"""
        self.generate_ptrip()

    def prediction_step(self, resample=True):
        """
        Prediction step.
        
        Resample if asked
        Propagate x_if and x_og
        Generate trip probabilities p_trip
        Remove first timestep of baselines
        """
        
        if resample:
            ind = resampling.residual_resample(self.W)

            self.x_if = self.x_if[ind]
            self.x_og = self.x_og[ind]
            self.N_inbuilding = self.N_inbuilding[ind]
            self.base_if = self.base_if[:, ind]
            self.sqrtQ_if = self.sqrtQ_if[ind]
            self.A_if = self.A_if[ind]
            self.omega_if = self.omega_if[ind]
            self.base_og = self.base_og[:, ind]
            self.sqrtQ_og = self.sqrtQ_og[ind]
            self.A_og = self.A_og[ind]
            self.omega_og = self.omega_og[ind]

            self.W = np.repeat(1/self.N, self.N)

        self.x_if = np.random.normal(self.base_if[1, :] + self.A_if *
                                     (self.x_if - self.base_if[0, :]),
                                     self.sqrtQ_if)

        self.x_og = np.random.normal(self.base_og[1, :] + self.A_og *
                                     (self.x_og - self.base_og[0, :]),
                                     self.sqrtQ_og)

        self.generate_ptrip()

        self.trim_base()

    def predict_mean(self):
        """
        Predicts mean number of trips for new time interval
        
        Assumes that p_trip has been computed (should be run after init or 
        prediction_step)
        """
        predicted_if = np.sum(self.W * self.p_trip[0, :] * self.N_inbuilding)
        predicted_og = np.sum(self.W * self.p_trip[1, :] * self.N_inbuilding)
        return predicted_if, predicted_og

    def generate_ptrip(self):
        """
        Samples the gamma multipliers and computes trip probabilities
        
        Intended to be called from init and from prediction_step
        """
        gamma_if = np.random.gamma(1 / (self.omega_if - 1), self.omega_if - 1)
        gamma_og = np.random.gamma(1 / (self.omega_og - 1), self.omega_og - 1)
        p_trip = np.zeros((3, self.N))

        if_intensity = gamma_if * np.exp(self.x_if)
        og_intensity = gamma_og * np.exp(self.x_og)

        tot_intensity = if_intensity + og_intensity

        p_trip[2, :] = np.exp(-tot_intensity)
        p_trip[0, :] = if_intensity/tot_intensity * (1 - p_trip[2, :])
        p_trip[1, :] = og_intensity/tot_intensity * (1 - p_trip[2, :])

        self.p_trip = p_trip

    def update_step(self, y_ic, y_if, y_og):
        """
        Update weights and population based on measurement
        
        Assumes p_trip is up-to-date.
        """
        logW = np.log(self.W)  \
               + y_if * np.log(self.p_trip[0, :]) \
               + y_og * np.log(self.p_trip[1, :]) \
               + (self.N_inbuilding - y_if - y_og) * np.log(self.p_trip[2, :])

        logW[(y_if + y_og) > self.N_inbuilding] = -np.inf
        logW[(y_ic - y_og + self.N_inbuilding) < 0] = -np.inf

        W = np.exp(logW - np.max(logW))
        self.W = W / sum(W)

        self.N_inbuilding += y_ic - y_og

    def trim_base(self):
        """Removing the first steps of the baselines"""
        self.base_if = self.base_if[1:, :]
        self.base_og = self.base_og[1:, :]
