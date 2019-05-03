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
      2019-05-03 DWW
"""

import unittest
import sys
import os
import numpy as np
sys.path.append(os.path.abspath('..'))
from grayboxes.boxmodel import grid, noise
from grayboxes.white import White
from grayboxes.lightgray import LightGray
from grayboxes.darkgray import DarkGray
from grayboxes.black import Black
from grayboxes.inverse import Inverse
from grayboxes.neural import RadialBasis

"""
    Demonstrating of use of class Inverse with box models:
        - class White
        - class LightGray 
        - class DarkGray
        - class Black
"""


def f(self, x, *args, **kwargs):
    """
    Theoretical submodel for single data point
    
        y = a + b sin(c x_0) + d (x_1 - 1.5)^2

    Args:
        self (reference):
            reference to instance object

        x (1D array_like of float):
            common input with x.shape: (n_inp,)

        args (float, optional):
            tuning parameters

        kwargs (Union[float, int, str], optional):
            keyword arguments
            
    Returns:
        (1D array of float):
            result of function
        
    """
    if x is None:
        return np.ones(4)
    tun = args if len(args) == 1 else np.ones(4)

    y0 = tun[0] + tun[1] * np.sin(tun[2] * x[0]) + tun[3] * (x[1] - 1.5)**2
    return [y0]


X = grid(5, (0., 1.), (0., 1.),)
Y = np.asfarray([f('dummy', x) for x in X])
Y = noise(Y, relative=5e-2)
n_inp = X.shape[1]
x_ini = np.ones(n_inp)
bounds = [(-10, 10)] * n_inp
y_inv = [2]


class TestUM(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test1(self):
        X_1d = np.atleast_2d(np.linspace(0, 1, 5)).T
        Y_1d = np.sin(X_1d)
        phi = RadialBasis()
        phi.train(X_1d, Y_1d, centers=8, rate=0.5)
        y = phi.predict(X_1d)
        print('X y', X_1d, y)

        self.assertTrue(True)

    def test2(self):
        operation = Inverse(White(f))         
        x, y = operation(x=x_ini, y=y_inv, optimizer='ga', bounds=[(-1, 3)]*2)
        print('x y y_inv', x, y, y_inv)

        self.assertTrue(True)

    def test3(self):
        operation = Inverse(LightGray(f))
        x, y = operation(X=X, Y=Y, neurons=[4], x=x_ini, y=y_inv, 
                         optimizer='cg')
        print('x y y_inv', x, y, y_inv)

        self.assertTrue(True)

    def test4(self):
        operation = Inverse(DarkGray(f))
        x, y = operation(X=X, Y=Y, neurons=[8], x=x_ini, y=y_inv)
        print('x y y_inv', x, y, y_inv)

        self.assertTrue(True)

    def test5(self):
        # Example 5 (expanded form)
        model = Black()
        metrics = model.train(X, Y, neurons=[8] )
        operation = Inverse(model)
        x, y = operation(x=x_ini, y=y_inv)
        print('x y y_inv metrics', x, y, y_inv, metrics)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
    