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
import matplotlib.pyplot as plt
from typing import List, Optional, Iterable

from grayboxes.boxmodel import BoxModel
from grayboxes.plot import plot_isomap, plot_surface, plot_isolines
from grayboxes.array import rand, noise, grid, frame_to_arrays
from grayboxes.white import White
from grayboxes.lightgray import LightGray


def f(x: Optional[Iterable[float]], *c: float) -> List[float]:
    """
    Theoretical submodel y=f(x_com, x_tun) for single data point
    """
    n = 3
    if x is None:
        return np.ones(n)          # get number of tuning parameters
    c0, c1, c2 = c if len(c) == n else np.ones(n)

    y0 = c0 + c2 * x[0]**2 + c1 * x[1]
    y1 = c0 * x[1]
    return [y0, y1]


def L2_norm(y: Iterable[float], Y: Iterable[float]) -> float:
    return np.sqrt(np.mean(np.square(np.asfarray(y) - np.asfarray(Y))))


class TestUM(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass


    def test1(self):
        x = grid(100, [0.9, 1.1], [0.9, 1.1])
        y_tru = White('demo', 'test1')(x=x)
        y = noise(y_tru, relative=20e-2)

        plot_isomap(x[:, 0], x[:, 1], y_tru[:, 0], title='$y_{tru,0}$')
        plot_surface(x[:, 0], x[:, 1], y_tru[:, 0], title='$y_{tru,0}$')
        plot_isolines(x[:, 0], x[:, 1], y_tru[:, 0], title='$y_{tru,0}$',
                      levels=[0, 1e-4, 5e-4, .003, .005, .01, .02, .05, .1, 
                              .2])
        plot_isomap(x[:, 0], x[:, 1], y[:, 0], title='$y_0$')
        plot_isomap(x[:, 0], x[:, 1], (y - y_tru)[:, 0], 
                    title='$y_0-y_{tru,0}$')

        self.assertTrue(True)


    def test2(self):
        x = grid(4, [0, 12], [0, 10])
        y_tru = White(f, 'test2')(x=x)
        y = noise(y_tru, relative=20e-2)

        plot_isomap(x[:, 0], x[:, 1], y_tru[:, 0], title='$y_{tru,0}$')
        plot_isomap(x[:, 0], x[:, 1], y_tru[:, 1], title='$y_{tru,1}$')
        plot_surface(x[:, 0], x[:, 1], y_tru[:, 0], title='$y_{tru,0}$')
        plot_surface(x[:, 0], x[:, 1], y_tru[:, 1], title='$y_{tru,1}$')
        plot_isomap(x[:, 0], x[:, 1], y[:, 0], title='$y_0$')
        plot_isomap(x[:, 0], x[:, 1], y[:, 1], title='$y_1$')
        plot_isomap(x[:, 0], x[:, 1], (y - y_tru)[:, 0], 
                    title='$y_0-y_{tru,0}$')
        plot_isomap(x[:, 0], x[:, 1], (y - y_tru)[:, 1], 
                    title='$y_1-y_{tru,1}$')

        self.assertTrue(True)


    def test3(self):
        X = grid(5, [-1, 2], [3, 4])
        print('X:', X)

        Y_tru = White(f, 'test3')(x=X)
        plot_isomap(X[:, 0], X[:, 1], Y_tru[:, 0], title='$Y_{tru,0}$')
        plot_isomap(X[:, 0], X[:, 1], Y_tru[:, 1], title='$Y_{tru,1}$')
        print('Y_tru:', Y_tru)

        Y = noise(Y_tru, absolute=0.1, uniform=True)
        plot_isomap(X[:, 0], X[:, 1], Y[:, 0], title='$Y_{0}$')
        plot_isomap(X[:, 0], X[:, 1], Y[:, 1], title='$Y_{1}$')
        print('Y:', Y)

        dY = Y - Y_tru
        plot_isomap(X[:, 0], X[:, 1], dY[:, 0], title='$Y - Y_{tru,0}$')
        plot_isomap(X[:, 0], X[:, 1], dY[:, 1], title='$Y - Y_{tru,1}$')
        print('dY:', dY)

        self.assertTrue(True)


    def test4(self):
        model = BoxModel(f, 'test4')
        model.x = [1, 2]
        model.y = model.f([1,2], 2., 0., 1.)
        print('y:', model.y)

        # sets input
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
        y01 = frame_to_arrays(df, 'y0', 'y1')
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
        y_tru = White(f)(x=x)
        y = model(X=X, Y=Y, x=x)

        plt.title('Target, true solution and prediction')
        plt.scatter(X[:, 0], Y[:, 0], marker='o', label='$Y_0(X_0)$ target')
        plt.scatter(x[:, 0], y_tru[:, 0], marker='s', label='$y_{tru,0}(x_0)$')
        plt.scatter(x[:, 0], y[:, 0], marker='v', label='$y_0(x_0)$')
        plt.legend()
        plt.grid()
        plt.show()

        plt.title('Absolute error')
        plt.scatter(x[:, 0], y[:, 0] - y_tru[:, 0], marker='s',
                    label='$y_0 - y_{tru,0}$')
        plt.scatter(x[:, 0], y[:, 1] - y_tru[:, 1], marker='s',
                    label='$y_1 - y_{tru,1}$')
        plt.legend()
        plt.grid()
        plt.show()

        self.assertTrue(True)
 
    
    def test6(self):
        model = LightGray(f, 'test6')
        n_point = 20
        X = rand(n_point, [0, 10], [0, 10])
        Y = noise(White(f)(x=X), absolute=0.1)

        x = rand(n_point, [0, 10], [0, 10])
        y_tru = White(f)(x=x)

        metrics = model(X=X, Y=Y)
        print('metrics:', metrics)

        y = model(x=x)
        print('L2 (prd):', L2_norm(y, y_tru))

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
