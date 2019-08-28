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
      2019-03-19 DWW
"""

import unittest
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('..'))
from grayboxes.boxmodel import BoxModel
from grayboxes.plotarrays import plot_isomap, plot_surface, plot_isolines
from grayboxes.arrays import rand, noise, grid, frame_to_arrays
from grayboxes.white import White
from grayboxes.lightgray import LightGray


def f(self, x, *args, **kwargs):
    """
    Theoretical submodel y=f(x_com, x_tun) for single data point
    """
    n_tun = 3
    if x is None:
        return np.ones(n_tun)          # get number of tuning parameters
    tun = args if len(args) == n_tun else np.ones(n_tun)

    y0 = tun[0] + tun[2] * x[0]**2 + tun[1] * x[1]
    y1 = tun[0] * x[1]
    return [y0, y1]


class TestUM(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test1(self):
        x = grid(100, [0.9, 1.1], [0.9, 1.1])
        y_exa = White('demo', 'test1')(x=x)
        y = noise(y_exa, relative=20e-2)

        plot_isomap(x[:, 0], x[:, 1], y_exa[:, 0], title='$y_{exa,0}$')
        plot_surface(x[:, 0], x[:, 1], y_exa[:, 0], title='$y_{exa,0}$')
        plot_isolines(x[:, 0], x[:, 1], y_exa[:, 0], title='$y_{exa,0}$',
                      levels=[0, 1e-4, 5e-4, .003, .005, .01, .02, .05, .1, 
                              .2])
        plot_isomap(x[:, 0], x[:, 1], y[:, 0], title='$y_0$')
        plot_isomap(x[:, 0], x[:, 1], (y - y_exa)[:, 0], 
                    title='$y_0-y_{exa,0}$')

        self.assertTrue(True)

    def test2(self):
        x = grid(4, [0, 12], [0, 10])
        y_exa = White(f, 'test2')(x=x)
        y = noise(y_exa, relative=20e-2)

        plot_isomap(x[:, 0], x[:, 1], y_exa[:, 0], title='$y_{exa,0}$')
        plot_isomap(x[:, 0], x[:, 1], y_exa[:, 1], title='$y_{exa,1}$')
        plot_surface(x[:, 0], x[:, 1], y_exa[:, 0], title='$y_{exa,0}$')
        plot_surface(x[:, 0], x[:, 1], y_exa[:, 1], title='$y_{exa,1}$')
        plot_isomap(x[:, 0], x[:, 1], y[:, 0], title='$y_0$')
        plot_isomap(x[:, 0], x[:, 1], y[:, 1], title='$y_1$')
        plot_isomap(x[:, 0], x[:, 1], (y - y_exa)[:, 0], 
                    title='$y_0-y_{exa,0}$')
        plot_isomap(x[:, 0], x[:, 1], (y - y_exa)[:, 1], 
                    title='$y_1-y_{exa,1}$')

        self.assertTrue(True)

    def test3(self):
        X = grid(5, [-1, 2], [3, 4])
        print('X:', X)

        Y_exa = White(f, 'test3')(x=X)
        plot_isomap(X[:, 0], X[:, 1], Y_exa[:, 0], title='$Y_{exa,0}$')
        plot_isomap(X[:, 0], X[:, 1], Y_exa[:, 1], title='$Y_{exa,1}$')
        print('Y_exa:', Y_exa)

        Y = noise(Y_exa, absolute=0.1, uniform=True)
        plot_isomap(X[:, 0], X[:, 1], Y[:, 0], title='$Y_{0}$')
        plot_isomap(X[:, 0], X[:, 1], Y[:, 1], title='$Y_{1}$')
        print('Y:', Y)

        dY = Y - Y_exa
        plot_isomap(X[:, 0], X[:, 1], dY[:, 0], title='$Y - Y_{exa,0}$')
        plot_isomap(X[:, 0], X[:, 1], dY[:, 1], title='$Y - Y_{exa,1}$')
        print('dY:', dY)

        self.assertTrue(True)

    def test4(self):
        model = BoxModel(f, 'test4')
        y = model.f([2, 3], 2, 0, 1)
        print('y:', y)

        # sets input
        model.x = [1, 2]
        print('1: model.x:', model.x, 'model.y:', model.y)

        print('test data frame import/export')
        df = model.xy_to_frame()
        print('4: df:', df)

        df = model.XY_to_frame()
        print('5: df:', df)

        model.X = [[2, 3], [4, 5]]
        model.Y = [[22, 33], [44, 55]]
        df = model.XY_to_frame()
        print('6: df:', df)

        y0, y1 = frame_to_arrays(df, ['y0'], ['y1'])
        print('7 y0:', y0, 'y1:', y1)
        y01 = frame_to_arrays(df, ['y0', 'y1'])
        print('8 y01:', y01)
        y12, x0 = frame_to_arrays(df, ['y0', 'y1'], ['x0'])
        print('9 y12:', y12, 'x0', x0)

        self.assertTrue(True)

    def test5(self):
        model = LightGray(f, 'test5')
        n_point = 20
        X = rand(n_point, [0, 10], [0, 10])
        Y = noise(White(f)(x=X), absolute=0.1)

        x = rand(n_point, [0, 10], [0, 10])
        y_exa = White(f)(x=x)
        y = model(X=X, Y=Y, x=x)

        plt.title('Target, exact solution and prediction')
        plt.scatter(X[:, 0], Y[:, 0], marker='o', label='$Y_0(X_0)$ target')
        plt.scatter(x[:, 0], y_exa[:, 0], marker='s', label='$y_{exa,0}(x_0)$')
        plt.scatter(x[:, 0], y[:, 0], marker='v', label='$y_0(x_0)$')
        plt.legend()
        plt.grid()
        plt.show()

        plt.title('Absolute error')
        plt.scatter(x[:, 0], y[:, 0] - y_exa[:, 0], marker='s',
                    label='$y_0 - y_{exa,0}$')
        plt.scatter(x[:, 0], y[:, 1] - y_exa[:, 1], marker='s',
                    label='$y_1 - y_{exa,1}$')
        plt.legend()
        plt.grid()
        plt.show()

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
