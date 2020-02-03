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
      2020-01-29 DWW
"""
import initialize
initialize.set_path()

import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from grayboxes.array import grid, noise, rand
from grayboxes.white import White
from grayboxes.lightgray import LightGray
from grayboxes.mediumgray import MediumGray
from grayboxes.darkgray import DarkGray
from grayboxes.black import Black
from grayboxes.plot import plot_isomap, plot_isolines, plot_surface

from coloredlids.flow.pressure_drop import dp_in_red_mid_exp_out


"""
Source code for example 'Pressure drop in pipework' in grayboxes wiki
"""


def f(x, *args, **kwargs):
    """
    Theoretical submodel of pressure drop in a pipework combination
    """
    n = 4
    if x is None:
        return np.ones(n)
    c0, c1, c2, c3 = args if len(args) > 0 else np.ones(n)

    v1 = x[0]
    nu = x[1] * 1e-6                                            # mm2/s to m2/s
    if math.isclose(v1, 0.) or math.isclose(nu, 0.):         # no pressure drop
        return 0.

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
        model.silent = True
    figsize = (4, 3)                                    # excluding plotSurface

    # min and max number of hidden neurons for medium gray, dark gray and black
    med_nrn_rng, drk_nrn_rng, blk_nrn_rng = (1, 1), (2, 8), (15, 25)
    rel_noise = 10e-2
    trials_lgr = 2

    # shape & ranges of train (X,Y) & test data (x,y), shape:(nPoint,nInp/nOut)
    nX, nY, x_trn_rng, y_trn_rng = 16+1, 4+1, [2, 10], [40, 100] # [,,m/s,mm2/s
    nx, ny, x_tst_rng, y_tst_rng = 128+1, 128+1, [0, 12], [40, 100]

    X = grid((nX, nY), x_trn_rng, y_trn_rng)  # shape: (nX*nY, 2), [m/s, mm2/s]
    x = grid((nx, ny), x_tst_rng, y_tst_rng)  # shape: (nx*ny, 2), [m/s, mm2/s]
    Y_tru = White(f)(x=X, silent=True)
    y_tru = White(f)(x=x, silent=True)
    Y = noise(y=Y_tru, absolute=0, relative=rel_noise, uniform=True)

    if 1:
        lx0, lx1 = '$v$', r'$\nu_{kin}$'
        lY_tru, lY = r'$\Delta\,p_{tru}$', r'$\Delta\,p_{trn}$'
        xRng, yRng = x_tst_rng, pd.Series(y_tst_rng)
        plot_surface(x[:, 0], x[:, 1], y_tru[:, 0], labels=(lx0, lx1, lY_tru),
                     xrange=xRng, yrange=yRng, units=['m/s', 'mm$^2$/s','MPa'])
        plot_surface(x[:, 1], x[:, 0], y_tru[:, 0], labels=(lx1, lx0, lY_tru),
                    xrange=xRng, yrange=yRng, units=['m/s', 'mm$^2$/s', 'MPa'])
        plot_isomap(x[:, 0], x[:, 1], y_tru[:, 0],
                    labels=(lx0, lx1, lY_tru), figsize=figsize,
                    xrange=xRng, yrange=yRng, units=['m/s', 'mm$^2$/s', 'MPa'])
        plot_isolines(x[:, 0], x[:, 1], y_tru[:, 0],
                      labels=(lx0, lx1, lY_tru), figsize=figsize,
                      xrange=xRng, yrange=yRng,
                      units=['m/s', 'mm$^2$/s', 'MPa'])
        plot_isomap(X[:, 0], X[:, 1], Y[:, 0],
                    labels=(lx0, lx1, lY), figsize=figsize,
                    xrange=xRng, yrange=yRng, units=['m/s', 'mm$^2$/s', 'MPa'])
        plot_isolines(X[:, 0], X[:, 1], Y[:, 0],
                      labels=(lx0, lx1, lY), figsize=figsize,
                      xrange=xRng, yrange=yRng,
                      units=['m/s', 'mm$^2$/s', 'MPa'])
        plot_isomap(X[:, 0], X[:, 1], Y[:, 0] - Y_tru[:, 0],
                    labels=(lx0, lx1, lY + ' - ' + lY_tru), figsize=figsize,
                    xrange=xRng, yrange=yRng, units=['m/s', 'mm$^2$/s', 'MPa'])
    results = {}
    optNeur = {'neurons': [4,4], 
               'rr': 0.,  
               'epochs': 250, 
               'backend': 'keras',
               'show': 1, 
               'silent': 0,
               'goal': 1e-4, 
               'tolerated': 15e-3,
               'trainer': 'rprop', 
               'trials': 3, 
               'plot': 0}
    plt.figure(figsize=(4, 3))

    for model in models:
        print('+++ train and predict:', model.identifier)

        if isinstance(model, (White)):
            y = model(X=X, Y=Y, x=x)
            res = {'x': x, 'y': y, 'X': X, 'dY': model(x=X) - Y, 'neurons': 0}
        elif isinstance(model, (LightGray)):
            model._weights = None
            y = model(X=X, Y=Y, x=x, C0=rand(trials_lgr, *(4*[[0, 2]])),
                      trainers=['BFGS', 'differential_evolution'],
                      detailed=True)
            res = {'x': x, 'y': y, 'X': X, 'dY': model(x=X) - Y, 'neurons': 0}
        else:
            model.silent = True
            print('    nrn:', end=' ')
            allNrn = range(min(med_nrn_rng[0], drk_nrn_rng[0], blk_nrn_rng[0]),
                           max(med_nrn_rng[1], drk_nrn_rng[1], blk_nrn_rng[1])+1)

            medNrn = range(med_nrn_rng[0], med_nrn_rng[1]+1)
            drkNrn = range(drk_nrn_rng[0], drk_nrn_rng[1]+1)
            blkNrn = range(blk_nrn_rng[0], blk_nrn_rng[1]+1)

            L2_tst = np.inf
            L2_tst_vs_neurons = np.full(len(allNrn)+1, np.inf)
            for neurons in allNrn:
                if (isinstance(model, MediumGray) and neurons in medNrn) or \
                   (isinstance(model, DarkGray) and neurons in drkNrn) or \
                   (isinstance(model, Black) and neurons in blkNrn):
                    print(neurons, end=' ')
                    optNeur['neurons'] = neurons

                    y = model(X=X, Y=Y, x=x, **optNeur)
                    if y is None:
                        print('y is None')
                    else:
                        L2 = np.sqrt(np.mean((y - y_tru)**2))
                        L2_tst_vs_neurons[neurons] = L2
                        if L2_tst > L2:
                            L2_tst = L2
                            res = {'x': x, 'y': y, 'X': X, 'dY': model(x=X) - Y,
                                   'neurons': neurons}
            print()
            plt.figure(figsize=figsize)
            plt.title('$||y-Y||_2$ vs neur. (' + model.identifier + ')')
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

    def curve_indices(nx, ny, pos):
        i = {'min': (0,          nx),
             'med': (nx*(ny//2), nx*(ny//2+1)),
             'max': (nx*(ny-1),  nx*ny)}
        return i[pos]
    nu_pos = {'min': X[:, 1].min(), 'med': (X[:, 1].min() + X[:, 1].max()) / 2,
              'max': X[:, 1].max(), }

    for pos in ['min', 'med', 'max']:
        plt.figure(figsize=figsize)
        plt.title(r'$\Delta p_{prd} (\nu$=' + str(round(nu_pos[pos])) + ')')
        plt.xlabel('$v$ [m/s]')
        plt.ylabel(r'$\Delta p$ [MPa]')
        plt.xlim(x_tst_rng[0], x_tst_rng[1])
        for key, res in results.items():
            _nx, _ny = (nx, ny) if key not in ('noise') else (nX, nY)
            b, e = curve_indices(_nx, _ny, pos)

            if key.lower() not in ():
                ls = '-' if key not in ('noise') else '--'
                x, y = res['x'], res['y']
                n = res['neurons']
                s = ' [' + str(n) + ']' if 'neurons' in res and n > 0 else ''
                s = s + 'se' if key == 'noise' else s
                plt.plot(x[b:e, 0], y[b:e, 0], label=key[:3] + s, ls=ls)
        plt.axvline(x=x_trn_rng[0], ls='--', lw=1.5, c='tab:gray')
        plt.axvline(x=x_trn_rng[1], ls='--', lw=1.5, c='tab:gray',
                    label='train')
        plt.legend(bbox_to_anchor=(1.1, 1.05), loc='upper left')
        plt.grid()
        plt.show()

    for pos in ['min', 'med', 'max']:
        plt.figure(figsize=figsize)
        plt.title(r'$\Delta p_{prd} - \Delta p_{trn} (\nu$=' +
                  str(round(nu_pos[pos])) + ')')
        plt.xlabel('$v$ [m/s]')
        plt.ylabel(r'$\Delta(\Delta p)$ [MPa]')
        plt.xlim(x_tst_rng[0], x_tst_rng[1])

        for key, res in results.items():
            if key != 'noise':
                b, e = curve_indices(nX, nY, pos)

                if key.lower() not in ():
                    ls = '-' if key not in ('noise') else '--'
                    x, y = res['X'], res['dY']
                    n = res['neurons']
                    s = ' ['+str(n)+']' if 'neurons' in res and n > 0 else ''
                    plt.plot(x[b:e, 0], y[b:e, 0], label=key[:3]+s, ls=ls)
        plt.axvline(x=x_trn_rng[0], ls='--', lw=1.5, c='tab:gray')
        plt.axvline(x=x_trn_rng[1], ls='--', lw=1.5, c='tab:gray',
                    label='train')
        plt.legend(bbox_to_anchor=(1.1, 1.05), loc='upper left')
        plt.grid()
        plt.show()

    plt.figure(figsize=figsize)
    plt.title(r'$\Delta p_{tru}(v,\nu_{kin}$=const)')
    plt.xlabel('$v$ [m/s]')
    plt.ylabel(r'$\Delta p$ [MPa]')
    plt.xlim(x_tst_rng[0], x_tst_rng[1])

    x = grid((nx, ny), x_tst_rng, y_tst_rng)  # shape: (nx*ny, 2), [m/s, mm2/s]
    y = White(f)(x=x, silent=True)
    for pos in ['min', 'med', 'max']:
        b, e = curve_indices(nx, ny, pos)
        plt.plot(x[b:e, 0], y[b:e, 0], label=r'$\nu_{kin}$: ' +
                 str(round(nu_pos[pos])))

    plt.axvline(x=x_trn_rng[0], ls='--', lw=1.5, c='tab:gray')
    plt.axvline(x=x_trn_rng[1], ls='--', lw=1.5, c='tab:gray', label='train')
    plt.legend(bbox_to_anchor=(1.1, 1.05), loc='upper left')
    plt.grid()
    plt.show()
