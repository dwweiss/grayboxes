"""
  Copyright (c) 2016- by Dietmar W Weiss

  This is free software; you can redistribute it and/or modify it
  under the terms of the GNU Lesser General Public License as
  published by the Free Software Foundation; either version 3.0 of
  the License, or (at your option) any later version.

  This software is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this software; if not, write to the Free
  Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
  02110-1301 USA, or see the FSF site: http://www.fsf.org.

  Version:
      2018-05-08 DWW
"""

import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt


def plotFigCostsVsDegreeOfBlackness():
    """
    Plots figure costs = f(degree of blackness) for Wiki
    """

    plt.xkcd()
    plt.rcParams.update({'font.size': 18})
    plt.rc('xtick', labelsize=16)
    # plt.rc('ytick', labelsize=20)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['right'].set_color('none')  # suppress top and right boundary
    ax.spines['top'].set_color('none')

    plt.xticks([0, 33, 66, 100])
    plt.yticks([])
    plt.xlabel('degree of model blackness [%]')
    plt.ylabel('costs')

    # ax.set_xlim([0, 100])
    ax.set_ylim([0, 110])

    x = np.arange(100)
    ySim = interp1d([0, 40, 100], [100, 36, 1], kind=2)(x)
    yObs = interp1d([0, 37, 100], [10, 15, 100], kind=2)(x)
    yTot = ySim + yObs

    plt.plot(x, ySim)
    plt.plot(x, yObs)
    plt.plot(x, yTot)

    plt.annotate('Optimum', xy=(50, 50), arrowprops=dict(arrowstyle='->'),
                 xytext=(55, 85))
    plt.annotate('Total', xy=(30, 62), color='green')
    plt.annotate('Observation', xy=(-2, 17))
    plt.annotate('Simulation', xy=(70, 17))
    plt.show()


if __name__ == '__main__':
    plotFigCostsVsDegreeOfBlackness()
