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
      2020-01-28 DWW
"""

import initialize
initialize.set_path()

import numpy as np
from grayboxes.array import grid, noise
from grayboxes.black import Black
from grayboxes.lightgray import LightGray
from grayboxes.plot import plot_isomap
from grayboxes.white import White


def func(x, *c):
    # initial coefficients 
    c0, c1 = 1., 1.
    
    # functions returns initial coefficients if x is None
    if x is None:
        return c0, c1

    # coefficients from positional arguments if len(c) is 2 
    if len(c) == 2:
        c0, c1 = c
        
    # theoretical model returns 1D array 
    return [c0 + x[0] + c1 * np.sin(x[1])]


# White 
X = grid((32, 32), [0., 1.], [-1., 3.])
Y_tru = White(f=func)(x=X, silent=True)
Y_nse = noise(Y_tru, relative=5e-2)


# LightGray
phi = LightGray(f=func)
phi(X=X, Y=Y_nse, trials=5, expected=1e-5, n_it_max=1000, detailed=0,
    trainer=('BFGS', 'L-BFGS-B', 'Nelder-Mead', 'Powell', 'CG'))
y_lgr = phi(x=X, silent=True)
print('Metrics + weights of light gray:', phi.metrics, phi.weights)


# Meta model of LightGray output
M = Black()
M(X=X, Y=y_lgr,
  backend='keras',
  neurons=[8, 4], 
  trials=3, 
  expected=1e-4, 
  trainer='rprop', 
  n_it_max=500, 
  show=100)
y_mta = M(x=X, silent=True)
print('Metrics of meta model training:', M.metrics)


axes = X[:, 0], X[:, 1]
plot_isomap(*axes,         Y_tru[:,0], labels=['x0', 'x1', 'y_wht'])
plot_isomap(*axes,         Y_nse[:,0], labels=['x0', 'x1', 'y_nse'])
plot_isomap(*axes, (Y_nse-Y_tru)[:,0], labels=['x0', 'x1', 'y_nse - y_tru'])

plot_isomap(*axes,         y_lgr[:,0], labels=['x0', 'x1', 'y_lgr \Delta'])
plot_isomap(*axes, (y_lgr-Y_nse)[:,0], labels=['x0', 'x1', 'y_lgr - y_nse'])
plot_isomap(*axes, (y_lgr-Y_tru)[:,0], labels=['x0', 'x1', 'y_lgr - y_tru'])

plot_isomap(*axes,         y_mta[:,0], labels=['x0', 'x1', 'y_mta'])
plot_isomap(*axes, (y_mta-Y_nse)[:,0], labels=['x0', 'x1', 'y_mta - y_nse'])
plot_isomap(*axes, (y_mta-Y_tru)[:,0], labels=['x0', 'x1', 'y_mta - y_tru'])
