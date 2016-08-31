###############################################################################
# Copyright (C) 2016 Juho Kokkala
# This is part of Juho Kokkala's PoDoCo project.
#
# This file is licensed under the MIT License.
###############################################################################
"""
Plotting the fake data.
data/simulate_data.py needs to be run first.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib2tikz import save as tikz_save


##
t = np.arange(159)/12

plt.cla()

plt.plot(t, np.exp(tm.ic_process.x_local), color='0.5')

plt.plot(t, np.exp(tm.ic_process.x_base +
                   0.5 * tm.ic_process.params['stdev_local']**2), 'k-')

plt.legend(['Daily intensities', 'Baseline'])

plt.xlabel('Time (h)')
plt.ylabel('Mean incomings per interval')

tikz_save('../report/fig/fake_ic_x.tikz',
          figureheight='\\figureheight',
          figurewidth='\\figurewidth')

##
plt.cla()

plt.plot(t, tm.y_ic, 'k.', markersize=1)

plt.xlabel('Time (h)')
plt.ylabel('Incomings per interval')

tikz_save('../report/fig/fake_ic_y.tikz',
          figureheight='\\figureheight',
          figurewidth='\\figurewidth')

##
plt.cla()

plt.plot(t, np.exp(tm.if_process.x_local), color='0.5')

plt.plot(t, np.exp(tm.if_process.x_base +
                   0.5 * tm.if_process.params['stdev_local']**2), 'k-')

plt.legend(['Daily intensities', 'Baseline'])

plt.xlabel('Time (h)')
plt.ylabel('Per-passenger interfloor trip intensity per interval')

tikz_save('../report/fig/fake_if_x.tikz',
          figureheight='\\figureheight',
          figurewidth='\\figurewidth')


##
plt.cla()

plt.plot(t, tm.y_if, 'k.', markersize=1)

plt.xlabel('Time (h)')
plt.ylabel('Interfloor trips per interval')

tikz_save('../report/fig/fake_if_y.tikz',
          figureheight='\\figureheight',
          figurewidth='\\figurewidth')

##
plt.cla()

plt.plot(t, np.exp(tm.og_process.x_local), color='0.5')

plt.plot(t, np.exp(tm.og_process.x_base +
                   0.5 * tm.og_process.params['stdev_local']**2), 'k-')

plt.legend(['Daily intensities', 'Baseline'])

plt.xlabel('Time (h)')
plt.ylabel('Per-passenger outgoing trip intensity per interval')

tikz_save('../report/fig/fake_og_x.tikz',
          figureheight='\\figureheight',
          figurewidth='\\figurewidth')

##
plt.cla()

plt.plot(t, tm.y_og, 'k.', markersize=1)

plt.xlabel('Time (h)')
plt.ylabel('Outgoing trips per interval')

tikz_save('../report/fig/fake_og_y.tikz',
          figureheight='\\figureheight',
          figurewidth='\\figurewidth')
