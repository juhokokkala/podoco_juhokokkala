###############################################################################
# Copyright (C) 2016 Juho Kokkala
# This is part of Juho Kokkala's PoDoCo project.
#
# This file is licensed under the MIT License.
###############################################################################
"""Plot filter performance for one testday, Poisson Model 1"""


import numpy as np
from matplotlib import pyplot as plt
from matplotlib2tikz import save as tikz_save

results = np.load("../filters/results_poisson1.npz")
rmse = results['rmse']
predicted = results['predicted']

y_train = np.genfromtxt('../data/incoming_train.csv', delimiter=',')
predicted_raw = np.mean(y_train, axis=1)
y_test = np.genfromtxt('../data/incoming_test.csv', delimiter=',')


t = (1/12) * np.arange(159)

plt.plot(t, predicted_raw, 'g--.')
plt.plot(t, predicted[:, 0], 'b.-')
plt.plot(t, y_test[:, 0], 'rx')
plt.xlabel("Time (h)")
plt.ylabel("Number of arrivals per 5 min interval")

plt.legend(["Raw mean", "Model one-step-ahead forecast", "Observations"])

plt.show()

tikz_save('../report/fig/ic_poisson1_day1.tikz',
          figureheight='\\figureheight',
          figurewidth='\\figurewidth')
