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
      2018-08-10 DWW
"""

from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt

from grayboxes.boxmodel import BoxModel
from grayboxes.white import White
from grayboxes.lightgray import LightGray
from grayboxes.mediumgray import MediumGray
from grayboxes.darkgray import DarkGray
from grayboxes.black import Black


def F_true(x):
    """
    True value of Y(X)
    """
    
    s = str()
    
    return np.sin(2 * x[0]) + x[0] + 1


def F_noise(x, noise_rel=0.0, noise_abs=0.25):
    """
    Training data is true value plus noise Y(X, Z)
    """
    y = F_true(x)
    return y * (1 + np.random.normal(loc=0, scale=noise_rel, size=y.shape)) + \
        np.random.normal(loc=0, scale=noise_abs, size=y.shape)


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
    X = np.linspace(xTrnRng[0], xTrnRng[1], nTrn).reshape(-1, 1)
    Y = np.asfarray([np.atleast_1d(F_noise(_x)) for _x in X])

    x = np.linspace(xTstRng[0], xTstRng[1], nTst).reshape(-1, 1)
    y_tru = np.asfarray([np.atleast_1d(F_true(_x)) for _x in x])
    Y_tru = np.asfarray([np.atleast_1d(F_true(_x)) for _x in X])

    models = [White(f), 
              LightGray(f), 
              # MediumGray(f), 
              DarkGray(f), 
              Black()]

    opt = {'neurons': [4], 
           'regularization': 0.5,  
           'epochs': 1000,
           'expected': 0.1e-3, 
           'tolerated': 10e-3, 
           'trainer': 'auto', 
           'trials': 3,
           'c0': np.ones(3), 
           'local': 1, 
           'shuffle': True}

    results = OrderedDict()
    for model in models:
        print('+++ model:', model.identifier)
        _y = model(X=X, Y=Y, x=x, **opt)
        results[model] = (x, _y)
    results['train'] = (X, Y)
    results['true'] = (x, y_tru)

    plt.title('Wrong white box')
    plt.xlabel('$x\, /\, \pi$')
    plt.ylabel('$y$')
    for model, xy in results.items():
        key = model.identifier if isinstance(model, BoxModel) else model
        
        if isinstance(model, White) or (isinstance(key, str) and
                                        key in ('true')):
            plt.plot(xy[0][:, 0]/np.pi, xy[1][:, 0], label=key, ls='-')
    plt.axvline(x=xTrnRng[0]/np.pi, ls='--', lw=1.5, c='tab:gray')
    plt.axvline(x=xTrnRng[1]/np.pi, ls='--', lw=1.5, c='tab:gray',
                label=r'$\Delta x_{train}$')
    plt.legend(bbox_to_anchor=(1.1, 1.05), loc='upper left')
    plt.xlim([x/np.pi for x in xTstRng])
    plt.ylim(-2, 5)
    plt.grid()
    plt.show()

    plt.title('Results with wrong white box')
    plt.xlabel('$x\, /\, \pi$')
    plt.ylabel('$y$')
    for model, xy in results.items():
        key = model.identifier if isinstance(model, BoxModel) else model
        if key != 'noise':
            ls = '-'
            if key in ('train'):
                ls = '--'
            if key in ('true'):
                ls = ':'
            plt.plot(xy[0][:, 0]/np.pi, xy[1][:, 0], label=key, ls=ls)
    plt.axvline(x=xTrnRng[0]/np.pi, ls='--', lw=1.5, c='tab:gray')
    plt.axvline(x=xTrnRng[1]/np.pi, ls='--', lw=1.5, c='tab:gray',
                label=r'$\Delta x_{train}$')
    plt.legend(bbox_to_anchor=(1.1, 1.05), loc='upper left')
    plt.xlim([x/np.pi for x in xTstRng])
    plt.ylim(-2, 5)
    plt.grid()
    plt.show()

    plt.title('Difference to train data (noisy)')
    plt.xlabel('$x\, /\, \pi$')
    plt.ylabel(r'$y - y_{nse}$')

    for model, xy in results.items():
        if isinstance(model, BoxModel):
            key = model.identifier
        else:
            key = model
        if key not in ('true', 'train'):
            _y = model(x=X)
            print('117 _y Y', _y.shape, Y.shape)

            dy = _y[:, 0] - Y[:, 0]
            ls = '-' if key not in ('true', 'noise') else ':'
            plt.plot(X/np.pi, dy, label=key, ls=ls)
    plt.axvline(x=xTrnRng[0]/np.pi, ls='--', lw=1.5, c='tab:gray')
    plt.axvline(x=xTrnRng[1]/np.pi, ls='--', lw=1.5, c='tab:gray',
                label=r'$\Delta x_{trn}$')
    plt.legend(bbox_to_anchor=(1.1, 1.05), loc='upper left')
    plt.xlim([x/np.pi for x in xTstRng])
    plt.ylim(-2, 2)
    plt.grid()
    plt.show()

    for ylim in [(-3, 3), (-2, 2)]:
        plt.title('Difference to true data')
        plt.xlabel('$x\, /\, \pi$')
        plt.ylabel(r'$y - y_{tru}$')
        for model, xy in results.items():
            if isinstance(model, BoxModel):
                key = model.identifier
            else:
                key = model
            if key != 'true':
                ls = '-'
                if key in ('train'):
                    ls = '--'
                    dy = xy[1] - Y_tru
                else:
                    dy = xy[1] - y_tru
                if key in ('true'):
                    ls = ':'
                plt.plot(xy[0][:, 0]/np.pi, dy[:, 0], label=key, ls=ls)
        plt.axvline(x=xTrnRng[0]/np.pi, ls='--', lw=1.5, c='tab:gray')
        plt.axvline(x=xTrnRng[1]/np.pi, ls='--', lw=1.5, c='tab:gray',
                    label=r'$\Delta x_{trn}$')
        plt.legend(bbox_to_anchor=(1.1, 1.05), loc='upper left')
        plt.xlim([x/np.pi for x in xTstRng])
        plt.ylim(ylim)
        plt.grid()
        plt.show()
