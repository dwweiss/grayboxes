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

import __init__
__init__.init_path()

import unittest
import numpy as np
import matplotlib.pyplot as plt

from grayboxes.black import Black


def L2(y: np.ndarray, Y: np.ndarray) -> float:
    return np.sqrt(np.mean(np.square(y - Y)))


def str_L2(y: np.ndarray, Y: np.ndarray) -> str:
    return str(np.round(L2(y, Y), 4))


class TestUM(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test11(self):

        s = 'Example: Train with sin() + noise, predict outside train range'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        def f(x):
            return np.sin(x) * 10 + 0

        X = np.atleast_2d(np.linspace(-1.75 * np.pi, 1.75 * np.pi, 50)).T
        Y = f(X)
        dx = 0.5 * (X.max() - X.min())
        x = np.atleast_2d(np.linspace(X.min() - dx, X.max() + dx, 100)).T
        
        model = Black()
        
        # keyword 'neurons' selects Neural as empirical model in Black
        y1 = model(X=X, Y=Y, x=x, neurons=[8], plot=1, epochs=500, goal=1e-5,
                   show=None)

        print('+++ shapes of X Y x y:', model.X.shape, model.Y.shape, 
              model.x.shape, model.y.shape, )

        plt.title('Test, L2:' + str(round(model.metrics['L2'], 5)))
        plt.plot(model.x, y1, '-', label='MLP')
        plt.plot(model.X, model.Y, '.', label='target')
        plt.xlabel('x')
        plt.ylabel('y(x)')
        plt.show()

        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
