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
      2020-01-31 DWW
"""

import initialize
initialize.set_path()

import matplotlib.pyplot as plt
import numpy as np
import unittest

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
        """ 
        target is a sine curve, training input is this curce + noise
        prediction fails outside of input of training data
        """
        s = 'Example: Train with sin() + noise, predict outside train range'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        def f(x):
            return np.sin(x) * 10 + 0

        n_obs = 2000
        n_tst = 400
        X = np.linspace(-2 * np.pi, 2 * np.pi, n_obs).reshape(-1, 1)
        Y = f(X)
        dx = 0.5 * (X.max() - X.min())
        x = np.linspace(X.min() - dx, X.max() + dx, n_tst).reshape(-1, 1)
        
        phi = Black()
        
        # keyword 'neurons' selects Neural as empirical phi in Black
        y1 = phi(X=X, Y=Y, x=x, 
                 activation='auto',
                 backend='keras',
                 epochs=250,
                 expected=0.5e-3,
#                 neurons=[[10]*i for i in range(8)], 
                 neurons=(10,10,10),#'brute_force',
                 output=None,
                 patience=20,
                 plot=1,
                 show=None,
                 tolerated=5e-3,
                 trainer='adam',
                 trials=10,
                 )

        if phi.ready:
            print('+++ shapes of X Y x y:', phi.X.shape, phi.Y.shape, 
                  phi.x.shape, phi.y.shape, )
    
            plt.title('Test, L2:' + str(round(phi.metrics['L2'], 5)))
            plt.plot(phi.x, y1, '-', label='MLP')
            plt.plot(phi.X, phi.Y, '.', label='target')
            plt.xlabel('x')
            plt.ylabel('y(x)')
            plt.show()

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
