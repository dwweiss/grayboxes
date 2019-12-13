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
      2019-11-22 DWW
"""

import initialize
initialize.set_path()

import unittest
import numpy as np
from typing import Any, List, Optional, Sequence

from grayboxes.lightgray import LightGray
from grayboxes.plot import plot_x_y_y_ref
from grayboxes.array import grid, noise, rand
from grayboxes.white import White


def f(self, x: Optional[Sequence[float]], *c: float, **kwargs: Any) \
        -> List[float]:
    """
    Theoretical submodel for single data point

    Args:
        self (reference):
            reference to instance object

        x:
            common input

        c:
            tuning parameters as positional arguments

        kwargs:
            keyword arguments
    """
    if x is None:
        return np.ones(4)
    c0, c1, c2, c3 = c if len(c) == 4 else np.ones(4)

    y0 = c0 + c1 * np.sin(c2 * x[0]) + c3 * (x[1] - 1.5)**2
    return [y0]


def f2(self, x: Optional[Sequence[float]], *c: float, **kwargs: Any) \
        -> List[float]:
    if x is None:
        return np.ones(4)
    c0, c1, c2, c3 = c if len(c) > 0 else np.ones(4)

    y0 = c0 + c1 * np.sin(c2 * x[0]) + c3 * (x[1] - 1.5)**2
    return [y0]


trainer = [
           # 'all',
           # 'L-BFGS-B',
           'BFGS',
           'Powell',
           # 'Nelder-Mead',
           # 'differential_evolution',
           # 'basinhopping',
           'genetic',
           ]




class TestUM(unittest.TestCase):
    def setUp(self):
        noise_abs = 0.25
        noise_rel = 10e-2
        self.X = grid(8, (-1, 8), (0, 3))
        self.y_exa = White(f)(x=self.X, silent=True)
        self.Y = noise(self.y_exa, absolute=noise_abs, relative=noise_rel)
        plot_x_y_y_ref(self.X, self.Y, self.y_exa, ['X', 'Y_{nse}', 
                                                    'y_{exa}'])

    def tearDown(self):
        pass


    def test1(self):
        s = 'Creates exact output y_exa(X), add noise, target is Y(X)'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        s = 'Tunes model, compare: y(X) vs y_exa(X)'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        # train with 9 random initial tuning parameter sets, each of size 4
        model = LightGray(f2, 'test1')
        c_ini = rand(9, *(4 * [[0, 2]]))

        for _c_ini in [c_ini, None]:
            print('+++ c_ini:\n', _c_ini, '\n', '*'*40)

            y = model(X=self.X, Y=self.Y, c_ini=_c_ini, x=self.X, 
                      trainer=trainer, detailed=True, n_it_max=5000, 
                      bounds=4*[(0, 2)])

            y = LightGray(f2, 'test1b')(X=self.X, Y=self.Y, x=self.X, 
                         trainer=trainer, n_it_max=5000, c_ini=_c_ini,
                         silent=not True, detailed=True)
            plot_x_y_y_ref(self.X, y, self.y_exa, ['X', 'y', 'y_{exa}'])
            if 1:
                print('metrics:', model.metrics)
                df = model.xy_to_frame()
                print('=== df:\n', df)

        self.assertTrue(True)


    def test2(self):
        # train with single ini tun parameter set, n_tun from f2(None)
        variant_a = True

        if variant_a:
            y = LightGray(f2, 'test2')(X=self.X, Y=self.Y, x=self.X, 
                         c_ini=np.ones(4), silent=not True, trainer=trainer,
                         detailed=False)
        else:
            y = LightGray(f2, 'test2b')(X=self.X, Y=self.Y, x=self.X, 
                         silent=False, trainer='all')

        self.assertTrue(y is not None)


if __name__ == '__main__':
    unittest.main()
