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
      2018-11-22 DWW
"""

import __init__
__init__.init_path()

import unittest
import numpy as np
import scipy
from typing import Any, List, Optional, Sequence

from grayboxes.minimum import Minimum
from grayboxes.plot import plot_surface, plot_isomap
from grayboxes.array import rand, grid
from grayboxes.forward import Forward
from grayboxes.white import White


# theoretical submodel
def f(x: Optional[Sequence[float]], *args: float, **kwargs: Any) \
        -> List[float]:
    c0, c1, c2 = args if len(args) > 0 else (1, 1, 1)
    return [+(np.sin(c0 * x[0]) + c1 * (x[1] - 1)**2 + c2)]


# alternative theoretical submodel 
def f1(x: Optional[Sequence[float]], *args: float, **kwargs: Any) \
        -> List[float]:
    c0, c1, c2 = args if len(args) > 0 else (1, 1, 1)
    return [+(np.sin(c0 * x[0]) + c1 * (x[1] - 1)**2 + c2)]

# theoretical submodel without kw-arguments, needed by scipy.optimize.minimum()
def f_return_float(x: Optional[Sequence[float]], *args:float, **kwargs: Any) \
        -> float:
    y = np.sin(x[0]) + (x[1])**2 + 2 
    return y


class TestUM(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test1(self):
        s = 'Use scipy.optimize.minimize()'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))
        res = scipy.optimize.minimize(fun=f_return_float, x0=(4, 2), 
                                      method='nelder-mead',)
        print('res.x:', res.x)

        self.assertTrue(True)

    def test2(self):
        s = 'Minimum, assigns random series of initial x'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        op = Minimum(White('demo'), 'test2')
        x, y = op(x=rand(10, [-5, 5], [-7, 7]), optimizer='nelder-mead',
                  silent=True)
        # op.plot()
        print('x:', x, 'y:', y, '\nop.x:', op.x, 'op.y:', op.y)

        self.assertTrue(True)

    def test3(self):
        s = 'Minimum, assigns random series of initial x'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        op = Minimum(White(f), 'test3')
        x, y = op(x=rand(10, [-5, 5], [-7, 7]), optimizer='nelder-mead',
                  silent=True)
        # op.plot()
        print('x:', x, 'y:', y, '\nop.x:', op.x, 'op.y:', op.y)

        self.assertTrue(True)

    def test4(self):
        s = 'Minimum, generates series of initial x on grid'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        x, y = Forward(White(f), 'test4')(x=grid(3, [-2, 2], [-2, 2]))
        plot_surface(x[:, 0], x[:, 1], y[:, 0])
        plot_isomap(x[:, 0], x[:, 1], y[:, 0])

        op = Minimum(White(f))
        x, y = op(x=rand(3, [-5, 5], [-7, 7]))

        op.plot()
        print('x:', x, 'y:', y)

        self.assertTrue(True)

    def test5(self):
        s = 'Minimum, test all optimizers'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        if True:
            op = Minimum(White('demo'), 'test5')

            optimizers = ['ga', 'BFGS']
            optimizers = op._valid_optimizers

            for optimizer in optimizers:
                print('\n'+'-'*60+'\n' + optimizer + '\n' + '-'*60+'\n')

                x = rand(3, [0, 2], [0, 2])
                if optimizer == 'ga':
                    x, y = op(x=x, optimizer=optimizer, bounds=2*[(0, 2)],
                              generations=5000)
                else:
                    x, y = op(x=x, optimizer=optimizer, silent=True)

                if 0:
                    op.plot('traj')
                print('x:', x, 'y:', y)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
