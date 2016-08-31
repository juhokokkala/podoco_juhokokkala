## PoDoCo project: Bayesian Methods for Forecasting Elevator Traffic

This repository contains open-source Python and Stan implementations related to my Post Docs in Companies project (see http://www.podoco.fi/). A research report (explaining what this is about) will become public later -- see http://www.juhokokkala.fi/podoco/.

## Requires
The Python codes are for Python 3.5.
The Stan codes are for Stan 2.11 (I used CmdStan).
Python packages:
 - NumPy
 - SciPy
 - GPy
 - matplotlib
 - matplotlib2tikz (only for generating tikz files for the article) 

## Instructions
1. Simulate data: run the script simulate_data.py in the data_generation directory
2. Compile the Stan models (CmdStan 2.11) to the stan_models directory
3. Training (MCMC): run_stan_models.bat in stan_models (or similar commands if not using Windows)
4. Testing: Run script_test_poisson_1.py in filters for Poisson Model 1 (Incoming)
            Run script_test_poisson_2.py in filters for Poisson Model 2 (Incoming)
            Run script_test_negbin.py in filters for NegBin Model (Incoming)
            Run script_test_ifog.py in filters for testing Interfloor+Outgoing
            Run script_test_negbin_shiftedday.py in filters for testing NegBin with shifted data (this also plots)
5. Plotting: Run plot_fakedata_illustration.py in plotting to illustrate the fake data
             Run plot_GP_illustration.py in plotting to create general illustration of GP
             Run plot_poisson1_day1.py in plotting to plot Poisson Model 1 performance
                 for one testday
             Run plot_posterior_distributions_poisson1.py in plotting to plot MCMC results
                 of Poisson Model 1 (incoming)
             Run plot_posterior_distributions_poisson2.py in plotting to plot MCMC results
                 of Poisson Model 2 (incoming)
             Run plot_posterior_distributions_negbin.py in plotting to plot MCMC results
                 of NegBin Model (incoming)
             Run plot_posterior_distributions_ifog.py in plotting to plot MCMC results
                 of the Interfloor+Outgoing model.

## Licensing information

Copyright (c) Juho Kokkala 2016 (except the file resampling.py). All files in this repository are licensed under the MIT License. See the file LICENSE or http://opensource.org/licenses/MIT. 

The file resampling.py is Copyright (c) Roger R Labbe Jr. 2015,  taken from the filterpy package, http://github.com/ - also licensed under the MIT License.