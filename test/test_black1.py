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
      2020-02-18 DWW
"""

import initialize
initialize.set_path()

import matplotlib.pyplot as plt
import numpy as np
import sys
from typing import List
import unittest

from grayboxes.array import grid, noise 
from grayboxes.black import Black
from grayboxes.datatype import Float1D
from grayboxes.white import White


class TestUM(unittest.TestCase):
    def setUp(self):
        self.save_figures = True

    def tearDown(self):
        pass

    def test2(self):
        # neural network, 1D problem sin(x) with noise
        def f(x: Float1D, *c: float) -> List[float]:
            c0, c1 = c if len(c) > 0 else 1., 1.
            return np.sin(x) + c0 * x + c1

        # training data
        N = 300
        absolute_ = 0.1
        X = grid((N, 1), [-np.pi, np.pi])
        Y_tru = White(f)(x=X, silent=True)
        Y_nse = noise(Y_tru, absolute=absolute_)

        # test data
        dx = 1. * np.pi
        n = 100
        x = grid((n, 1), [X.min() - dx, X.max() + dx])

        if 'grayboxes.neuraltf' in sys.modules:
            neurons = [[8]*i for i in range(2, 4+1)]
        else:
            neurons = [10, 10]

        phi = Black('black')

        metrics_trn = phi(X=X, Y=Y_nse, 
                          activation='leakyrelu',
                          backend='tensorflow',
                          batch_size=[None, 8, 16, 32, 64],
                          epochs=500, 
                          expected=0.1e-3,
                          neurons=neurons,
                          output='linear',
                          plot=1,
                          silent=0,
                          tolerated=5e-3,
                          trainer='adam',
                          trials=3, 
                          )
        
        y = phi(x=x, silent=True)

        if phi.ready:
            y_tru = White(f)(x=x, silent=True)
            metrics_tst = phi.evaluate(x, y_tru)
    
            plt.title('MSE$^{trn}$:' + str(round(metrics_trn['mse'], 4))+',' +
                      ' MSE$^{tst}$:' + str(round(metrics_tst['mse'], 4))+'$')
#            plt.ylim(min(-2, Y_nse.min(), y.min()), 
#                     max(2, Y_nse.max(), Y_nse.max()))
            plt.yscale('linear')
            plt.xlim(-0.1 + x.min(), 0.1 + x.max())
            
            plt.scatter(X, Y_nse, marker='x', c='r', label='trn')
            plt.plot(x, y, c='b', label='tst')
            plt.plot(x, f(x), linestyle=':', label='tru')
    
            i_abs_trn = metrics_trn['i_abs']
            plt.scatter(X[i_abs_trn], Y_nse[i_abs_trn], marker='o', 
                        color='r', s=66, label='max abs train')
    
            i_abs_tst = metrics_tst['i_abs']
            plt.scatter(x[i_abs_tst], y[i_abs_tst], marker='o', 
                        color='b', s=66, label='max abs test')
    
            plt.legend(bbox_to_anchor=(1.1, 0), loc='lower left')
            plt.grid()
            plt.show()

        self.assertTrue(phi.ready)


if __name__ == '__main__':
    unittest.main()
