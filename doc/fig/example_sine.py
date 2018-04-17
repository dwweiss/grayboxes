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
      2018-04-17 DWW
"""

import numpy as np
import matplotlib.pyplot as plt
from Forward import Forward


def F_true(x):
    """
    Conventional true value of Y(X)
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
    Theoretical model, subject to modelling error
    """
    return c0 + c1 * x[0] + c2 * x[0]**2


if __name__ == '__main__':
    """
    Source code for generating the example 'sine curve' in the grayBoxes' wiki
    """

    nTst, xTstRng = 100, [-1 * np.pi, 1.5 * np.pi]                 # test input
    nTrn, xTrnRng = 20, [-0.5 * np.pi, 1 * np.pi]              # training input

    X = np.atleast_2d(np.linspace(xTrnRng[0], xTrnRng[1], nTrn)).T
    Y = np.atleast_2d([np.atleast_1d(F_noise(x)) for x in X])
    x = np.atleast_2d(np.linspace(xTstRng[0], xTstRng[1], nTst)).T
    y = np.atleast_2d([np.atleast_1d(F_true(_x)) for _x in x])

    models = ['white', 'light gray',
              # 'medium gray-a',
              'dark gray', 'black']
    results = {'noise': (X, Y), 'true': (x, y)}   # collection of results (x,y)

    opt = {'regularization': 0.5, 'hidden': [2], 'epochs': 2000,
           'goal': 1e-6, 'trainers': 'rprop bfgs', 'trials': 5}  # neural opt.

    for model in models:
        print('++++ model:', model)
        Y = results['noise'][1]
        results[model] = Forward(f=f, model=model)(X=X, Y=Y, x=x, **opt)

    plt.title('Sine curve with wrong white box')
    plt.xlabel('$x$')
    plt.xlabel('$y$')
    for model in models + ['true', 'noise']:
        x, y = results[model][0], results[model][1]
        ls = '-' if model not in ('true', 'noise') else ':'
        plt.plot(x[:, 0], y[:, 0], label=model, ls=ls)
    plt.legend(bbox_to_anchor=(1.1, 1.05), loc='upper left')
    plt.grid()
    plt.show()

    plt.title('Difference to true values')
    plt.xlabel(r'$\Delta y$')
    for model in models:
        x, y = results[model][0], results[model][1] - results['true'][1]
        ls = '-' if model not in ('white', 'black') else ':'
        plt.plot(x[:, 0], y[:, 0], label=r'$\Delta$ '+model, ls=ls)
    plt.legend(bbox_to_anchor=(1.1, 1.05), loc='upper left')
    plt.ylim(-5, 5)
    plt.grid()
    plt.show()
