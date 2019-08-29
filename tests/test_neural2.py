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
      2019-04-30 DWW
"""

import unittest
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../grayboxes'))

from grayboxes.black import Black
#from grayboxes.neural import RadialBasis


def L2(y, Y):
    return np.sqrt(np.mean(np.square(y - Y)))


def str_L2(y, Y):
    return str(np.round(L2(y, Y), 4))


class TestUM(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

#    def _test7(self):
#        s = 'Example 7'
#        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))
#
#        def f(x):
#            return np.sin(x) * 10 + 0
#
#        X = np.atleast_2d(np.linspace(-1.75 * np.pi, 1.75 * np.pi, 50)).T
#        Y = f(X)
#        dx = 0.5 * (X.max() - X.min())
#        x = np.atleast_2d(np.linspace(X.min() - dx, X.max() + dx)).T
#
#        # emprical submodel y = beta(x)
#        beta = RadialBasis()
#        y = beta(X=X, Y=Y, x=x, centers=10, plot=1, epochs=500, goal=1e-5,
#                 show=None, rate=0.8)
#
#        print('beta: X Y x y, y', beta.X.shape, beta.Y.shape, beta.x.shape, 
#              beta.y.shape, y.shape)
#
#        plt.title('Test, L2:' + str(round(beta.metrics['L2'], 5)))
#        plt.plot(beta.x, beta.y, '-')
#        plt.plot(beta.X, beta.Y, '.')
#        plt.legend(['pred', 'targ', ])
#        plt.xlabel('x')
#        plt.ylabel('y(x)')
#        plt.show()
#
#        self.assertTrue(True)

    def test8(self):
        s = 'Example 8'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        def f(x):
            return np.sin(x) * 10 + 0

        X = np.atleast_2d(np.linspace(-1.75 * np.pi, 1.75 * np.pi, 50)).T
        Y = f(X)
        dx = 0.5 * (X.max() - X.min())
        x = np.atleast_2d(np.linspace(X.min() - dx, X.max() + dx, 100)).T
        
        # model can be any box type model (white, gray or black)
        model = Black()
        
        # keyword 'neurons' selects Neural as empirical model in Black
        model = Black()
        y1 = model(X=X, Y=Y, x=x, neurons=[8], plot=1, epochs=500, goal=1e-5,
                   show=None)

#        # keyword 'centers' selects RadialBasis as empirical model in Black
#        y2 = model(X=X, Y=Y, x=x, centers=8, plot=1, epochs=500, goal=1e-5,
#                   show=10, rate=1, basis='gaussian')

#        y3 = model(X=X, Y=Y, x=x, centers=20, plot=1, epochs=500, goal=1e-5,
#                   show=None, rate=1, basis='multiquadratic')

        print('X Y x y', model.X.shape, model.Y.shape, model.x.shape, 
              model.y.shape, )

        plt.title('Test, L2:' + str(round(model.metrics['L2'], 5)))
        plt.plot(model.x, y1, '-')
#        plt.plot(model.x, y2, '-')
#        plt.plot(model.x, y3, '-')
        plt.plot(model.X, model.Y, '.')
        plt.legend(['pred MLP', 'pred RBF-gaus', 
#                    'pred RBF-mult', 
                    'targ', ])
        plt.xlabel('x')
        plt.ylabel('y(x)')
        plt.show()

        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
