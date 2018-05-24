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
      2018-05-24 DWW
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import Model as md
from White import White
from LightGray import LightGray
from MediumGray import MediumGray
from DarkGray import DarkGray
from Black import Black
from plotArrays import plotIsoMap, plotIsolines, plotSurface

from pressureDrop import dp_in_red_mid_exp_out


"""
Source code for example 'Pressure drop in pipework' in grayBoxes wiki
"""


def f(x, *args, **kwargs):
    """
    Theoretical submodel
    """
    if x is None:
        return [(0, 2), (0, 2), (0, 2), (0, 2)]       # return could also be: 4
    c0, c1, c2, c3 = args if len(args) > 0 else np.ones(4)

    v1 = x[0]
    if math.isclose(v1, 0):             # zero velocity causes no pressure drop
        return 0
    nu = x[1] * 1e-6                                            # mm2/s to m2/s

    D1, L1 = kwargs.get('D1', 15e-3), kwargs.get('L1', 100e-3)
    D2, L2 = kwargs.get('D2', D1/2),  kwargs.get('L2', L1/2)
    D3, L3 = kwargs.get('D3', D1),    kwargs.get('L3', L1)
    rho = kwargs.get('rho', 1000)
    eps_rough = kwargs.get('eps_rough', 10e-6)
    res = dp_in_red_mid_exp_out(v1, D1, L1, D2, L2, D3, L3, nu, rho, eps_rough,
                                c0, c1, c2, c3)

    return res[0] * 1e-6                                            # Pa to MPa


###############################################################################

if __name__ == '__main__':

    # model selection
    models = [
              White(f),
              LightGray(f),
              # MediumGray(f),
              DarkGray(f),
              Black()
              ]
    for model in models:
        model.silent = not True

    # min and max number of hidden neurons
    medNrnRng, drkNrnRng, blkNrnRng = (1, 1), (10, 20), (15, 30)
    relNoise = 10e-2
    trialsLgr = 3

    # shape of training (X, Y) and test data (x, y), shape: (nPoint, nInp/nOut)
    nX, nY, xTrnRng, yTrnRng = 20, 3, [3, 12], [1, 50]  # [/, /, m/s, mm2/s]
    nx, ny, xTstRng, yTstRng = 40, 3, [0, 15], [1, 50]  # [/, /, m/s, mm2/s]

    X = md.grid((nX, nY), xTrnRng, yTrnRng)  # shape:(nX*nY,2), [m/s,1e-5*m2/s]
    x = md.grid((nx, ny), xTstRng, yTstRng)  # shape:(nx*ny,2), [m/s, mm2/s]
    Y_exa = White(f)(x=X, silent=True)
    y_exa = White(f)(x=x, silent=True)
    Y = md.noise(y=Y_exa, absolute=0, relative=relNoise, uniform=True)

    if 1:
        xRng, yRng = xTstRng, pd.Series(yTstRng)*10
        plotSurface(x[:, 0], x[:, 1]*10, y_exa[:, 0],
                    labels=('$v$', r'$\nu_{kin}$', r'$\Delta\,p_{exa}$'),
                    xrange=xRng, yrange=yRng, units=['m/s', 'mm$^2$/s', 'MPa'])
        plotIsolines(x[:, 0], x[:, 1]*10, y_exa[:, 0],
                     labels=('$v$', r'$\nu_{kin}$', r'$\Delta\,p_{exa}$'),
                     xrange=xRng, yrange=yRng,
                     units=['m/s', 'mm$^2$/s', 'MPa'])
        plotIsoMap(x[:, 0], x[:, 1]*10, y_exa[:, 0],
                   labels=('$v$', r'$\nu_{kin}$', r'$\Delta\,p_{exa}$'),
                   xrange=xRng, yrange=yRng, units=['m/s', 'mm$^2$/s', 'MPa'])
        plotIsolines(X[:, 0], X[:, 1]*10, Y[:, 0],
                     labels=('$v$', r'$\nu_{kin}$', r'$\Delta\,p_{nse}$'),
                     xrange=xRng, yrange=yRng,
                     units=['m/s', 'mm$^2$/s', 'MPa'])
        plotIsoMap(X[:, 0], X[:, 1]*10, Y[:, 0],
                   labels=('$v$', r'$\nu_{kin}$', r'$\Delta\,p_{nse}$'),
                   xrange=xRng, yrange=yRng,
                   units=['m/s', 'mm$^2$/s', 'MPa'])
        plotIsoMap(X[:, 0], X[:, 1]*10, Y[:, 0] - Y_exa[:, 0],
                   labels=('$v$', r'$\nu_{kin}$',
                           r'$\Delta\,p_{nse}-\Delta\,p_{exa}$'),
                   xrange=xRng, yrange=yRng,
                   units=['m/s', 'mm$^2$/s', 'MPa'])
    results = {}
    optNeur = {'neurons': [], 'rr': 0.,  'epochs': 2000, 'show': 0,
               'goal': 1e-6, 'trainers': 'rprop', 'trials': 3, 'plot': 0}

    for model in models:
        print('+++ train and predict:', model.identifier)

        if isinstance(model, (White)):
            y = model(X=X, Y=Y, x=x)
            res = {'x': x, 'y': y, 'X': X, 'dY': model(x=X) - Y, 'neurons': 0}
        elif isinstance(model, (LightGray)):
            model._weights = None
            y = model(X=X, Y=Y, x=x,
                      C0=md.rand(5, [0, 2], [0, 2], [0, 2], [0, 2]),
                      trainers='BFGS', detailed=True)
            res = {'x': x, 'y': y, 'X': X, 'dY': model(x=X) - Y, 'neurons': 0}
        else:
            model.silent = not True
            print('    nrn:', end=' ')
            allNrn = range(min(medNrnRng[0], drkNrnRng[0], blkNrnRng[0]),
                           max(medNrnRng[1], drkNrnRng[1], blkNrnRng[1])+1)

            medNrn = range(medNrnRng[0], medNrnRng[1]+1)
            drkNrn = range(drkNrnRng[0], drkNrnRng[1]+1)
            blkNrn = range(blkNrnRng[0], blkNrnRng[1]+1)

            L2_tst = np.inf
            L2_tst_vs_neurons = np.full(len(allNrn)+1, np.inf)
            for neurons in allNrn:
                if (isinstance(model, MediumGray) and neurons in medNrn) or \
                   (isinstance(model, DarkGray) and neurons in drkNrn) or \
                   (isinstance(model, Black) and neurons in blkNrn):
                    print(neurons, end=' ')
                    optNeur['neurons'] = neurons

                    y = model(X=X, Y=Y, x=x, **optNeur)
                    L2 = np.sqrt(np.mean((y - y_exa)**2))
                    L2_tst_vs_neurons[neurons] = L2
                    if L2_tst > L2:
                        L2_tst = L2
                        res = {'x': x, 'y': y, 'X': X, 'dY': model(x=X) - Y,
                               'neurons': neurons}
            print()
            plt.title('$||y-Y||_2$ vs neurons (' + model.identifier + ')')
            plt.xlabel('neurons [/]')
            plt.ylabel('$L_2$ norm (test) [/]')
            plt.plot(L2_tst_vs_neurons)
            plt.grid()
            plt.show()
        results[model.identifier] = res
    results['noise'] = {'x': X, 'y': Y, 'X': X, 'dY': Y-Y, 'neurons': 0}

    print('best neurons:', [key + ': ' + str(val['neurons'])
          for key, val in results.items()
          if 'neurons' in val and val['neurons'] > 0])

    for pos in ['min', 'med', 'max']:
        plt.title(r"$\varphi(x)$ at '" + pos + "'-position")
        plt.xlabel('$v$ [m/s]')
        plt.ylabel(r'$\Delta p$ [MPa]')
        plt.xlim(xTstRng[0], xTstRng[1])

        for key, res in results.items():
            _nx, _ny = (nx, ny) if key not in ('noise') else (nX, nY)
            if pos == 'min':
                b, e = 0, _nx
            elif pos == 'med':
                b, e = _nx*(_ny//2), _nx*(_ny//2+1)
            elif pos == 'max':
                b, e = _nx*(_ny-1), _nx*_ny
            if key.lower() not in ():
                ls = '-' if key not in ('noise') else '--'
                x, y = res['x'], res['y']
                n = res['neurons']
                s = ' [' + str(n) + ']' if 'neurons' in res and n > 0 else ''
                s = s + 'oise' if key == 'noise' else s
                plt.plot(x[b:e, 0], y[b:e, 0], label=key[0] + s, ls=ls)
        plt.axvline(x=xTrnRng[0], ls='--', lw=1.5, c='tab:gray')
        plt.axvline(x=xTrnRng[1], ls='--', lw=1.5, c='tab:gray',
                    label='train')
        plt.legend(bbox_to_anchor=(1.1, 1.05), loc='upper left')
        plt.grid()
        plt.show()

    for pos in ['min', 'med', 'max']:
        plt.title(r"$\varphi(X) - Y(X)$ at '" + pos + "'-position")
        plt.xlabel('$v$ [m/s]')
        plt.ylabel(r'$\Delta(\Delta p)$ [MPa]')
        plt.xlim(xTstRng[0], xTstRng[1])

        for key, res in results.items():
            if key != 'noise':
                _nx, _ny = (nX, nY)
                if pos == 'min':
                    b, e = 0, _nx
                elif pos == 'med':
                    b, e = _nx*(_ny//2), _nx*(_ny//2+1)
                elif pos == 'max':
                    b, e = _nx*(_ny-1), _nx*_ny
                if key.lower() not in ():
                    ls = '-' if key not in ('noise') else '--'
                    x, y = res['X'], res['dY']
                    n = res['neurons']
                    s = ' ['+str(n)+']' if 'neurons' in res and n > 0 else ''
                    plt.plot(x[b:e, 0], y[b:e, 0], label=key[0]+s, ls=ls)
        plt.axvline(x=xTrnRng[0], ls='--', lw=1.5, c='tab:gray')
        plt.axvline(x=xTrnRng[1], ls='--', lw=1.5, c='tab:gray',
                    label='train')
        plt.legend(bbox_to_anchor=(1.1, 1.05), loc='upper left')
        plt.grid()
        plt.show()
