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
      2018-06-21 DWW
"""

import numpy as np
import matplotlib.pyplot as plt

from grayboxes.white import White
from grayboxes.lightgray import LightGray
from grayboxes.mediumgray import MediumGray
from grayboxes.darkgray import DarkGray
from grayboxes.black import Black


def F_true(x):
    """
    True value of Y(X)
    """
    return np.sin(2 * x[0]) + x[0] + 1


def F_noise(x, noise=0.2):
    """
    Training data are conventional true value plus noise Y(X, Z)
    """
    y = F_true(x)
    return y * (1 + np.random.normal(-noise, noise, size=y.shape))


def f(x, c0=1, c1=1, c2=0.5, c3=0, c4=0, c5=0, c6=0, c7=0):
    """
    Theoretical model, subject to error by the human modeller
    """
    return c0 + c1 * x[0] + c2 * x[0]**2


if __name__ == '__main__':
    """
    Source code for generating the example 'sine curve' in the grayBoxes' wiki
    """

    # training and test data
    nTst, xTstRng = 100, [-1 * np.pi, 1.5 * np.pi]                 # test input
    nTrn, xTrnRng = 20, [-0.5 * np.pi, 1 * np.pi]              # training input
    X = np.atleast_2d(np.linspace(xTrnRng[0], xTrnRng[1], nTrn)).T
    Y = np.atleast_2d([np.atleast_1d(F_noise(_x)) for _x in X])
    x = np.atleast_2d(np.linspace(xTstRng[0], xTstRng[1], nTst)).T
    y = np.atleast_2d([np.atleast_1d(F_true(_x)) for _x in x])

    models = [White(f), LightGray(f),
              # MediumGray(f),
              DarkGray(f), Black()
              ]

    opt = {'neurons': [2], 'regularization': 0.5,  'epochs': 500,
           'goal': 1e-5, 'methods': 'rprop bfgs', 'trials': 5, 
           'c0': np.ones(3)}

    results = {'noise': (X, Y), 'true': (x, y)}   # collection of results (x,y)
    for model in models:
        print('+++ model:', model.identifier)
        y = model(X=X, Y=Y, x=x, **opt)
        results[model.identifier] = (x, y)

    plt.title('Sine curve with wrong white box')
    plt.xlabel('$x\, /\, \pi$')
    plt.ylabel('$y$')
    plt.axvline(x=xTrnRng[0]/np.pi, ls='--', lw=1.5, c='tab:gray')
    plt.axvline(x=xTrnRng[1]/np.pi, ls='--', lw=1.5, c='tab:gray',
                label=r'$\Delta x_{train}$')

    for modelType in list(results.keys()):
        (_x, _y) = results[modelType]
        ls = '-' if modelType not in ('true', 'noise') else ':'
        plt.plot(_x[:, 0]/np.pi, _y[:, 0], label=modelType, ls=ls)
    plt.legend(bbox_to_anchor=(1.1, 1.05), loc='upper left')
    plt.ylim(-5, 10)
    plt.grid()
    plt.show()

    plt.title('Difference to noisy data (train data)')
    plt.xlabel('$x\, /\, \pi$')
    plt.ylabel(r'$y_{true} - y$')
    plt.axvline(x=xTrnRng[0]/np.pi, ls='--', lw=1.5, c='tab:gray')
    plt.axvline(x=xTrnRng[1]/np.pi, ls='--', lw=1.5, c='tab:gray',
                label=r'$\Delta x_{train}$')
    for key, xy in results.items():
        if key != 'noise':
            dy = xy[1] - F_noise(x=xy[0])
            ls = '-' if not key.lower().startswith(('white', 'black')) else ':'
            plt.plot(xy[0][:, 0]/np.pi, dy[:, 0], label=key, ls=ls)
    plt.legend(bbox_to_anchor=(1.1, 1.05), loc='upper left')
    plt.ylim(-2, 6)
    plt.grid()
    plt.show()

    plt.title('Difference to true data')
    plt.xlabel('$x\, /\, \pi$')
    plt.ylabel(r'$y_{true} - y$')
    plt.axvline(x=xTrnRng[0]/np.pi, ls='--', lw=1.5, c='tab:gray')
    plt.axvline(x=xTrnRng[1]/np.pi, ls='--', lw=1.5, c='tab:gray',
                label=r'$\Delta x_{train}$')
    for key, xy in results.items():
        if key != 'true':
            dy = xy[1] - F_true(x=xy[0])
            ls = '-' if not key.lower().startswith(('white', 'black')) else ':'
            plt.plot(xy[0][:, 0]/np.pi, dy[:, 0], label=key, ls=ls)
    plt.legend(bbox_to_anchor=(1.1, 1.05), loc='upper left')
    plt.ylim(-2, 6)
    plt.grid()
    plt.show()