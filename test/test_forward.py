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
      2019-12-09 DWW
"""

import initialize
initialize.set_path()

import unittest
import numpy as np
from typing import Any, List, Optional, Sequence

from grayboxes.array import grid, cross, rand
from grayboxes.forward import Forward
from grayboxes.lightgray import LightGray
from grayboxes.plot import plot_isomap
from grayboxes.white import White


# function without access to 'self' attributes
def func(x: Optional[Sequence[float]], *c: float, **kwargs: Any) \
        -> List[float]:
#    print('0')
    return 3.3 * np.array(np.sin(x[0]) + (x[1] - 1)**2)


# method with access to 'self' attributes
def method(self, x, *c, **kwargs):
#    print('1')
    return 3.3 * np.array(np.sin(x[0]) + (x[1] - 1)**2)


class TestUM(unittest.TestCase):
    def setUp(self):
        pass
    

    def tearDown(self):
        pass


    def test1(self):
        s = 'Forward() with demo function build-in into BoxModel'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        x, y = Forward(White(func), 'test1')(x=grid(3, [0, 1], [0, 1]))
        plot_isomap(x[:, 0], x[:, 1], y[:, 0])

        self.assertTrue(True)


    def test2(self):
        s = 'Forward() with demo function build-in into BoxModel'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))
        x, y = Forward(White(f='demo'))(x=cross(5, [1, 2], [3, 4]))
        plot_isomap(x[:, 0], x[:, 1], y[:, 0], scatter=True)

        self.assertTrue(True)


    def test3(self):
        s = "Forward, assign external function (without self-argument) to f"
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        operation = Forward(White(func), 'test3')
        _, y = operation(x=rand(12, [2, 3], [3, 4]))
        print('x:', operation.model.x, '\ny1:', operation.model.y)

        self.assertTrue(True)


    def test4(self):
        s = "Forward, assign method (with 'self'-argument) to f"
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        operation = Forward(White(func), 'test4')
        _, y = operation(x=[[2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        print('x:', operation.model.x, '\ny1:', operation.model.y)

        self.assertTrue(True)


    def test5(self):
        s = "Forward, LightGray"
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        X = rand(20, (-1, 1), (-1, 1))
        Y = [[func(x_)] for x_ in X]

        operation = Forward(LightGray(func), 'test5')
        result = operation(X=X, Y=Y, c_ini=[1, 1], trainer='least_squares')
        print('result:', result)
        print('metrics:', operation.model.metrics)
        x, y = operation(x=X)
        print('x:', x, '\n', 'y:', y)

        self.assertTrue(True)


    def test6(self):
        s = "Forward, LightGray 2"
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        X = rand(20, (-1, 1), (-1, 1))
        Y = [[func(x_)] for x_ in X]

        operation = Forward(LightGray(func), 'test5')
        metrics = operation(X=X, Y=Y, c_ini=[1, 1], trainer='least_squares')
        print('metrics:', metrics)
        print('metrics:', operation.model.metrics)
        x, y = operation(x=X)
        print('x:', x, '\n', 'y:', y)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
