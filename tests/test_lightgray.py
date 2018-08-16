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
      2018-08-16 DWW
"""

import unittest	
import os
import sys
import numpy as np

sys.path.append(os.path.abspath('..'))
from grayboxes.lightgray import LightGray

from grayboxes.plotarrays import plot_X_Y_Yref
from grayboxes.model import grid, noise, rand
from grayboxes.white import White


def f(self, x, *args, **kwargs):
    """
    Theoretical submodel for single data point

    Aargs:
        x (1D array_like of float):
            common input

        args (argument list, optional):
            tuning parameters as positional arguments

        kwargs (dict, optional):
            keyword arguments {str: float/int/str}
    """
    if x is None:
        return np.ones(4)
    tun = args if len(args) == 4 else np.ones(4)

    y0 = tun[0] + tun[1] * np.sin(tun[2] * x[0]) + tun[3] * (x[1] - 1.5)**2
    return [y0]


def f2(self, x, *args, **kwargs):
    if x is None:
        return np.ones(4)
    tun = args if len(args) > 0 else np.ones(4)

    y0 = tun[0] + tun[1] * np.sin(tun[2] * x[0]) + tun[3] * (x[1] - 1.5)**2
    return [y0]


methods = [
           # 'all',
           # 'L-BFGS-B',
           'BFGS',
           'Powell',
           # 'Nelder-Mead',
           # 'differential_evolution',
           # 'basinhopping',
           'genetic',
           ]


noise_abs = 0.25
noise_rel = 10e-2
X = grid(8, [-1, 8], [0, 3])
y_exa = White(f)(x=X, silent=True)
Y = noise(y_exa, absolute=noise_abs, relative=noise_rel)
plot_X_Y_Yref(X, Y, y_exa, ['X', 'Y_{nse}', 'y_{exa}'])


class TestUM(unittest.TestCase):
    def setUp(self):
        pass
 
    def tearDown(self):
        pass
 
    def test1(self):
        s = 'Creates exact output y_exa(X), add noise, target is Y(X)'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))
    


        s = 'Tunes model, compare: y(X) vs y_exa(X)'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        # train with 9 random initial tuning parameter sets, each of size 4
        model = LightGray(f2)
        tun0 = rand(9, *(4 * [[0, 2]]))

        for _tun0 in [tun0, None]:
            print('+++ tun0:', _tun0, '*'*40)

            y = model(X=X, Y=Y, tun0=_tun0, x=X, methods=methods,
                      detailed=True, nItMax=5000, bounds=4*[(0, 2)])

            y = LightGray(f2)(X=X, Y=Y, x=X, methods=methods, nItMax=5000,
                              tun0=_tun0, silent=not True, detailed=True)

            plot_X_Y_Yref(X, y, y_exa, ['X', 'y', 'y_{exa}'])
            if 1:
                print('best:', model.best)
                df = model.xy2frame()
                print('=== df:\n', df)

        self.assertTrue(True)

    def test2(self):
        def f2(self, x, *args, **kwargs):
            if x is None:
                return np.ones(4)
            p = args if len(args) > 0 else np.ones(4)
            y0 = p[0] + p[1] * np.sin(p[2] * x[0]) + p[3] * (x[1] - 1.5)**2
            return [y0]

        # train with single initial tuning parameter set, nTun from f2(None)
        if 1:
            y = LightGray(f2)(X=X, Y=Y, x=X, tun0=np.ones(4),
                              silent=not True, methods=methods)

        y = LightGray(f2)(X=X, Y=Y, x=X,
                          silent=not True, methods='all')

        self.assertTrue(y is not None)
 
       
if __name__ == '__main__':
    unittest.main()
