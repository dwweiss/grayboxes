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
      2019-04-01 DWW
"""

import __init__
__init__.init_path()

import unittest
import os
import numpy as np

from grayboxes.forward import Forward
from grayboxes.plot import plot_isomap
from grayboxes.array import grid, cross, rand
from grayboxes.white import White


# function without access to 'self' attributes
def func(x, *args):
    print('0')
    return 3.3 * np.array(np.sin(x[0]) + (x[1] - 1)**2)


# method with access to 'self' attributes
def method(self, x, *args, **kwargs):
    print('1')
    return 3.3 * np.array(np.sin(x[0]) + (x[1] - 1)**2)


class TestUM(unittest.TestCase):
    def setUp(self):
        print('///', os.path.basename(__file__))

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
        x, y = Forward(White('demo'), 'test2')(x=cross(5, [1, 2], [3, 4]))
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


if __name__ == '__main__':
    unittest.main()
