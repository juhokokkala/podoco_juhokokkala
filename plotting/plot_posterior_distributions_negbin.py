###############################################################################
# Copyright (C) 2016 Juho Kokkala
# This is part of Juho Kokkala's PoDoCo project.
#
# This file is licensed under the MIT License.
###############################################################################
"""Plot MCMC results, Poisson Model 2 for incoming traffic."""

import numpy as np
# Relative import kludge
from os import chdir
chdir("../utils/")
import stan_utilities as stanu
chdir("../plotting/")
from matplotlib import pyplot as plt
from matplotlib2tikz import save as tikz_save
from scipy import stats


##
print("Reading samples from Stan output")
stanfile = "../stan_models/output_ic_negbin.csv"
mcmc_header, mcmc_data = stanu.read_output(stanfile)

##

base_mcmc = mcmc_data[:, mcmc_header.index('x_base.1'):
                      mcmc_header.index('x_base.159')+1].T

lscale_local_mcmc = mcmc_data[:, mcmc_header.index('lscale_local')]
s_local_mcmc = mcmc_data[:, mcmc_header.index('s_local')]

lscale_base_mcmc = mcmc_data[:, mcmc_header.index('lscale_base')]
s_base_mcmc = mcmc_data[:, mcmc_header.index('s_base')]

omega_mcmc = 1 / mcmc_data[:, mcmc_header.index('invomega')]

del(mcmc_data)   # Saving memory
del(mcmc_header)

##
truebases = np.load("../data/groundtruth_baselines.npz")
true_slocal = 0.15


##
t = np.arange(159)/12

mean_prediction = np.exp(base_mcmc + 0.5 * s_local_mcmc**2)
true_meanpred = np.exp(truebases['x_base_ic'] + true_slocal**2)

plt.cla()

plt.plot(t, np.mean(mean_prediction, axis=1), 'k-')
plt.plot(t, np.percentile(mean_prediction, 95, axis=1), 'k--')
plt.plot(t, true_meanpred, 'g-.')
plt.plot(t, np.percentile(mean_prediction, 5, axis=1), 'k--')

plt.legend(['Posterior mean', '90 \% credible interval', 'Ground truth'])
plt.xlabel('Time')
plt.ylabel('Mean arrivals per interval')

# plt.show()

tikz_save('../report/fig/ic_negbin_mcmc_predmean.tikz',
          figureheight='\\figureheight',
          figurewidth='\\figurewidth')

##
plt.cla()
plt.plot(t, np.mean(base_mcmc, axis=1), 'k-')
plt.plot(t, truebases['x_base_ic'], 'g-.')
plt.plot(t, np.percentile(base_mcmc, 95, axis=1), 'k--')
plt.plot(t, np.percentile(base_mcmc, 5, axis=1), 'k--')

plt.legend(['Posterior mean', '90 \% credible interval', 'Ground truth'])
plt.xlabel('Time')
plt.ylabel('Baseline log-intensity')

# plt.show()

tikz_save('../report/fig/ic_negbin_mcmc_base.tikz',
          figureheight='\\figureheight',
          figurewidth='\\figurewidth')


##
def plot_param_hist(sample, name, prior, bins=30, xlim=None, groundtruth=None,
                    filename=None):
    plt.cla()

    plt.ticklabel_format(axis='x', style='plain')
    plt.axes().get_yaxis().set_ticks([])

    if xlim is None:
        xlim = [np.min(sample), np.max(sample)]
    binwidth = (xlim[1] - xlim[0]) / bins
    x = np.arange(xlim[0], xlim[1], binwidth)
    posterior = np.histogram(sample, x)[0] / (binwidth * sample.shape[0])
    plt.plot(0.5 * x[:-1] + 0.5 * x[1:], posterior, 'k.-')
    plt.plot(x, prior(x), 'k--')
    plt.xlabel(name)

    if groundtruth is not None:
        plt.plot(groundtruth, 0, 'r.')
        plt.legend(['Posterior', 'Prior', 'Ground truth'])
    else:
        plt.legend(['Posterior', 'Prior'])

    if filename is not None:
        tikz_save(filename, figureheight='\\figureheight',
                  figurewidth='\\figurewidth')

##
plot_param_hist(lscale_local_mcmc, '$\\rho_\mathrm{(l)}$',
                lambda x: 2 * stats.t.pdf(x, 2),
                groundtruth=0.06,
                xlim=[0, 7],
                filename='../report/fig/ic_negbin_mcmc_lscale_local.tikz')
##

plot_param_hist(lscale_base_mcmc, '$\\rho_\mathrm{(b)}$',
                lambda x: 2 * (1/0.25) * stats.t.pdf(x/0.25, 2),
                groundtruth=0.1,
                filename='../report/fig/ic_negbin_mcmc_lscale_base.tikz')
##
plot_param_hist(s_local_mcmc, '$\sigma_\mathrm{(l)}$',
                lambda x: 2 * stats.t.pdf(x, 2),
                groundtruth=0.15,
                filename='../report/fig/ic_negbin_mcmc_s_local.tikz')
##
plot_param_hist(s_base_mcmc, '$\sigma_\mathrm{(b)}$',
                lambda x: 2 * stats.t.pdf(x, 2),
                groundtruth=3,
                filename='../report/fig/ic_negbin_mcmc_s_base.tikz')

##

plot_param_hist(omega_mcmc, '$\omega$',
                lambda x: x**(-2),
                groundtruth=3,
                filename='../report/fig/ic_negbin_mcmc_omega.tikz')
