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
from typing import List, Optional, Sequence

from grayboxes.array import grid, noise
from grayboxes.black import Black
from grayboxes.white import White


class TestUM(unittest.TestCase):
    def setUp(self):
        self.saveFigures = True


    def tearDown(self):
        pass


    def test1(self):
        noise_abs = 0.1
        X = grid((20, 1), [0, 1])
        Y = noise(White(f=lambda x: [x[0]**2])(x=X), absolute=noise_abs)

        y = Black('black')(X=X, Y=Y, x=X, neurons=[], silent=True)

        plt.plot(X, Y, '.', label='trn')
        plt.plot(X, y, '.', label='tst')
        plt.legend(); plt.grid(); plt.show()

        self.assertTrue(True)


    def test2(self):
        # neural network, 1D problem sin(x) with noise
        def f(x: Optional[Sequence[float]], *c: float) -> List[float]:
            c0, c1 = c if len(c) > 0 else 1., 1.
            return np.sin(x) + c0 * x + c1

        # training data
        n_point_trn = 10
        noise_abs = 0.01
        X = grid((n_point_trn, 1), [-np.pi, np.pi])
        Y_exa = f(X)
        Y_nse = noise(Y_exa, absolute=noise_abs)
        
#        print('X:', X)
#        print('Y_exa:', Y_exa)
#        print('Y_nse:', Y_nse)

        # test data
        dx = 0.5 * np.pi
        n_point_tst = n_point_trn
        x = grid((n_point_tst, 1), [X.min() - dx, X.max() + dx])

        blk = Black('black')
        kwargs_ = {'neurons': [10, 10], 'trials': 5, 'goal': 1e-6,
                   'epochs': 500, 'trainers': 'bfgs rprop',
                   'silent': True,
                   }

        metrics_trn = blk(X=X, Y=Y_nse, **kwargs_)
        print('tst blk 88 bk.ready', blk.ready) 
        y = blk(x=x, silent=True)
        y_exa = White(f)(x=x,silent=True)
        print('tst blk 91 bk.ready', blk.ready, 'y', y)
        metrics_tst = blk.evaluate(x, Y_exa)

        plt.title('$neurons:' + str(kwargs_['neurons']) +
                  ', L_2^{trn}:' + str(round(metrics_trn['L2'], 4)) +
                  ', L_2^{tst}:' + str(round(metrics_tst['L2'], 4)) + '$')
        plt.cla()
        plt.ylim(min(-2, Y_nse.min(), y.min()), 
                 max(2, Y_nse.max(), Y_nse.max()))
        plt.yscale('linear')
        plt.xlim(-0.1 + x.min(), 0.1 + x.max())
        
        plt.scatter(X, Y_nse, marker='x', c='r', label='trn')
        plt.plot(x, y, c='b', label='tst')
        plt.plot(x, f(x), linestyle=':', label='exa')

        i_abs_trn = metrics_trn['i_abs']
        plt.scatter([X[i_abs_trn]], [Y_nse[i_abs_trn]], marker='o', color='r',
                    s=66, label='max abs train')

        i_abs_tst = metrics_tst['i_abs']
        plt.scatter([x[i_abs_tst]], [y[i_abs_tst]], marker='o', color='b',
                    s=66, label='max abs test')

        plt.legend(bbox_to_anchor=(1.1, 0), loc='lower left')
        plt.grid()
        plt.show()

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
