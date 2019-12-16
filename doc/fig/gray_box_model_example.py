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
      2019-12-16 DWW
"""

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
x = grid((32, 32), [0., 1.], [-1., 3.])
y_wht = White(f=func)(x=x, silent=True)
y_nse = noise(y_wht, relative=5e-2)


# LightGray
phi = LightGray(f=func)
phi(X=x, Y=y_nse, trials=5, goal=1e-5, n_it_max=1000, detailed=True,
    trainer=('BFGS', 'L-BFGS-B', 'Nelder-Mead', 'Powell', 'CG'))
y_lgr = phi(x=x, silent=True)
print('Metrics + weights of light gray:', phi.metrics, phi.weights)


# Meta model of LightGray
M = Black()
M(X=x, Y=y_lgr, neurons=[8, 4], trials=3, goal=1e-5, trainer='rprop', 
  n_it_max=2000, show=100)
y_mta = M(x=x, silent=True)
print('Metrics of meta model training:', M.metrics)


axes = x[:, 0], x[:,1]
plot_isomap(*axes,         y_wht[:,0], labels=['x0', 'x1', 'y_wht'])
plot_isomap(*axes,         y_nse[:,0], labels=['x0', 'x1', 'y_nse'])
plot_isomap(*axes, (y_nse-y_wht)[:,0], labels=['x0', 'x1', 'y_nse - y_wht'])

plot_isomap(*axes,         y_lgr[:,0], labels=['x0', 'x1', 'y_lgr'])
plot_isomap(*axes, (y_lgr-y_nse)[:,0], labels=['x0', 'x1', 'y_lgr - y_nse'])
plot_isomap(*axes, (y_lgr-y_wht)[:,0], labels=['x0', 'x1', 'y_lgr - y_wht'])

plot_isomap(*axes,         y_mta[:,0], labels=['x0', 'x1', 'y_mta'])
plot_isomap(*axes, (y_mta-y_nse)[:,0], labels=['x0', 'x1', 'y_mta - y_nse'])
plot_isomap(*axes, (y_mta-y_wht)[:,0], labels=['x0', 'x1', 'y_mta - y_wht'])
