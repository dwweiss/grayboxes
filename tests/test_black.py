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
      2019-05-02 DWW
"""

import unittest
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('..'))
from grayboxes.black import Black
from grayboxes.arrays import grid, noise
from grayboxes.white import White


class TestUM(unittest.TestCase):
    def setUp(self):
        self.saveFigures = True
        pass

    def tearDown(self):
        pass

    def test1(self):
        noise_abs = 0.1
        X = grid(-20, [0, 1])
        Y = noise(White(lambda x: [x[0]**2])(x=X), absolute=noise_abs)

        y = Black('black1')(X=X, Y=Y, x=X, neurons=[], silent=True)

        plt.plot(X, Y, '.', X, y, '-')
        plt.show()

        self.assertTrue(True)

    def test2(self):
        # neural network, 1D problem sin(x) with noise
        def f(x, *args):
            a, b = args if len(args) > 0 else 1, 1
            return np.sin(x) + a * x + b

        # training data
        n_point_trn = 20
        noise = 0.01
        X = np.linspace(-1 * np.pi, +1 * np.pi, n_point_trn)
        Y = f(X)
        X, Y = np.atleast_2d(X).T, np.atleast_2d(Y).T
        Y_nse = Y.copy()
        if noise > 0.0:
            Y_nse += np.random.normal(-noise, +noise, Y_nse.shape)

        # test data
        dx = 0.5 * np.pi
        n_point_tst = n_point_trn
        x = np.atleast_2d(np.linspace(X.min()-dx, X.max()+dx, n_point_tst)).T

        blk = Black('black2')
        opt = {'neurons': [10, 10], 'trials': 5, 'goal': 1e-6,
               'epochs': 500, 'trainers': 'bfgs rprop'}

        metrics_trn = blk(X=X, Y=Y, **opt)
        y = blk(x=x)
        metrics_tst = blk.evaluate(x, White(f)(x=x))

        plt.title('$neurons:' + str(opt['neurons']) +
                  ', L_2^{trn}:' + str(round(metrics_trn['L2'], 4)) +
                  ', L_2^{tst}:' + str(round(metrics_tst['L2'], 4)) + '$')
        plt.cla()
        plt.ylim(min(-2, Y.min(), y.min()), max(2, Y.max(), Y.max()))
        plt.yscale('linear')
        plt.xlim(-0.1 + x.min(), 0.1 + x.max())
        plt.scatter(X, Y, marker='x', c='r', label='training data')
        plt.plot(x, y, c='b', label='prediction')
        plt.plot(x, f(x), linestyle=':', label='analytical')
        i_abs_trn = metrics_trn['iAbs']
        plt.scatter([X[i_abs_trn]], [Y[i_abs_trn]], marker='o', color='r',
                    s=66, label='max abs train')
        i_abs_tst = metrics_tst['iAbs']
        plt.scatter([x[i_abs_tst]], [y[i_abs_tst]], marker='o', color='b',
                    s=66, label='max abs test')
        plt.legend(bbox_to_anchor=(1.1, 0), loc='lower left')
        plt.grid()
        plt.show()

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
