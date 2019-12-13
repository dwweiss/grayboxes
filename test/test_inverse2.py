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

from grayboxes.array import grid, noise
from grayboxes.white import White
from grayboxes.lightgray import LightGray
from grayboxes.darkgray import DarkGray
from grayboxes.black import Black
from grayboxes.inverse import Inverse
#from grayboxes.neural import RadialBasis

"""
    Demonstrating of use of class Inverse with box models:
        - class White
        - class LightGray 
        - class DarkGray
        - class Black
"""


def f(self, x: Optional[Sequence[float]], *c: float, **kwargs: Any) \
        -> List[float]:
    """
    Theoretical submodel for single data point, x = (x_0, x_1), y = (y_0)
    
        y = a + b sin(c x_0) + d (x_1 - 1.5)^2

    Args:
        self (reference):
            reference to instance object

        x:
            common input with x.shape: (n_inp,)
.
        c:
            tuning parameters

    Kwargs:
        keyword arguments
            
    Returns:
        result of function
        
    """
    if x is None:
        return np.ones(4)
    c0, c1, c2, c3 = c if len(c) == 1 else np.ones(4)

    y0 = c0 + c1 * np.sin(c2 * x[0]) + c3 * (x[1] - 1.5)**2
    return [y0]




class TestUM(unittest.TestCase):
    def setUp(self):
        self.X = grid(5, (0., 1.), (0., 1.))
        self.Y = noise(White(f)(x=self.X), relative=5e-2)
        n_inp = self.X.shape[1]
        self.x_ini = np.ones(n_inp)
        self.bounds = [(-10., 10.)] * n_inp
        self.y_inv = [2]


    def tearDown(self):
        pass


    def test1(self):
        pass
        
    
    def test2(self):
        operation = Inverse(White(f))         
        x, y = operation(x=self.x_ini, y=self.y_inv, optimizer='ga', 
                         bounds=[(-1, 3)]*2)
        print('x y y_inv', x, y, self.y_inv)

        self.assertTrue(True)


    def test3(self):
        operation = Inverse(LightGray(f))
        x, y = operation(X=self.X, Y=self.Y, neurons=[4], x=self.x_ini, 
                         y=self.y_inv, optimizer='cg')
        print('x y y_inv', x, y, self.y_inv)

        self.assertTrue(True)


    def test4(self):
        operation = Inverse(DarkGray(f))
        x, y = operation(X=self.X, Y=self.Y, neurons=[8], x=self.x_ini, 
                         y=self.y_inv)
        print('x y y_inv', x, y, self.y_inv)

        self.assertTrue(True)


    def test5(self):
        # Example 5 (expanded form)
        model = Black()
        metrics = model.train(self.X, self.Y, neurons=[8] )
        operation = Inverse(model)
        x, y = operation(x=self.x_ini, y=self.y_inv)
        print('x y y_inv metrics', x, y, self.y_inv, metrics)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
    