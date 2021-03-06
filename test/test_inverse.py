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
      2020-01-20 DWW
"""

import initialize
initialize.set_path()

import numpy as np
from typing import List, Optional, Iterable
import unittest

from grayboxes.array import grid, rand, noise
from grayboxes.black import Black
from grayboxes.inverse import Inverse
from grayboxes.lightgray import LightGray
from grayboxes.plot import plot_x_y_y_ref
from grayboxes.white import White


def f(self, x: Optional[Iterable[float]], *c: float) -> List[float]:
    c0, c1, c2 = c if len(c) >= 3 else np.ones(3)
    
    return [np.sin(c0 * x[0]) + c1 * (x[1] - 1.)**2 + c2]


class TestUM(unittest.TestCase):
    def setUp(self):
        pass


    def tearDown(self):
        pass


    def test1(self):
        s = 'Inverse, ranges+rand replaced method f()'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        operation = Inverse(White(f), 'test1')
        x, y = operation(x=grid(3, [-1, 1], [4, 8], [3, 5]), y=[0.5])
        operation.plot()

        self.assertTrue(True)


    def test2(self):
        s = 'Inverse, replaced model'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        operation = Inverse(White('demo'), 'test2')
        x_inv, y_inv = operation(x=rand(3, [-5, 5], [-5, 5]), y=[0.5],
                                 optimizer='ga', bounds=2*[(-8, 8)], 
                                 )
        operation.plot()

        self.assertTrue(True)


    def test3(self):
        s = 'Inverse operation on light gray box model'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        noise_abs = 0.9
        n = 5
        X = grid((n, n), [-1, 5], [0, 3])

        #################

        Y_truct = White(f, 'test3')(x=X, silent=True)
        Y_noise = noise(Y_truct, absolute=noise_abs)
        plot_x_y_y_ref(X, Y_noise, Y_truct, ['X', 'Y_{nse}', 'Y_{tru}'])

        #################

        model = LightGray(f, 'test3b')
        Y_fit = model(X=X, Y=Y_noise, c0=np.ones(3), x=X, silent=True)
        plot_x_y_y_ref(X, Y_fit, Y_truct, labels=['X', 'Y_{fit}', 'Y_{tru}'])

        operation = Inverse(model, 'test3c')
        x, y = operation(x=grid((3, 2), [-10, 0], [1, 19]), y=[0.5])
        operation.plot()

        self.assertTrue(True)


    def test4(self):
        s = 'Inverse operation on empirical model'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        noise_abs = 0.2
        n = 10
        X = grid(n, [-1, 5], [0, 3])

        Y_truct = White(f, 'test4')(x=X)
        Y_noise = noise(Y_truct, absolute=noise_abs)

        plot_x_y_y_ref(X, Y_noise, Y_truct, labels=['X', 'Y_{nse}', 'Y_{tru}'])

        model = Black()
        Y_blk = model(X=X, Y=Y_noise, neurons=[8], n=3, epochs=500, x=X)

        plot_x_y_y_ref(X, Y_blk, Y_truct, labels=['X', 'Y_{blk}', 'Y_{tru}'])

        operation = Inverse(model, 'test4b')
        x_inv, y_inv = operation(y=[0.5], x=[(-10, 0), (1, 19)])

        operation.plot()

        self.assertTrue(True)


    def test5(self):
        s = 'Inverse operation on empirical model of tuned theoretical model'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        noise_abs = 0.1
        n = 10
        X = grid(n, [-1, 5], [0, 3])

        # synthetic training data
        Y_true = White(f, 'test5')(x=X)
        Y_noise = noise(Y_true, absolute=noise_abs)
        plot_x_y_y_ref(X, Y_noise, Y_true, labels=['X', 'Y_{nse}', 'Y_{tru}'])

        print('*' * 40)

        # trains and executes theoretical model Y_fit=f(X,w)
        Y_fit = LightGray(f, 'test5b')(X=X, Y=Y_noise, c0=[1, 1, 1], x=X)
        plot_x_y_y_ref(X, Y_fit, Y_true, labels=['X', 'Y_{fit}', 'Y_{tru}'])

        # meta-model of theoretical model Y_emp=g(X,w)
        meta = Black('test5c')
        Y_meta = meta(X=X, Y=Y_fit, neurons=[10], x=X)
        if Y_meta is not None:
            plot_x_y_y_ref(X, Y_fit, Y_meta, 
                           labels=['X', 'Y_{met}', 'Y_{tru}'])
    
            # inverse solution with meta-model (emp.model of tuned theo.model)
            if 1:
                operation = Inverse(meta, 'test6')
                x_inv, y_inv = operation(x=[(-10, 0)], y=[0.5])
                operation.plot()
                print('id:', operation.identifier, 'x_inv', x_inv, 
                      'y_inv', y_inv)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
