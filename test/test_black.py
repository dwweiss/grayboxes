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
      2020-02-03 DWW
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
from grayboxes.datatypes import Float1D 
from grayboxes.white import White


class TestUM(unittest.TestCase):
    def setUp(self):
        self.saveFigures = True


    def tearDown(self):
        pass


    def test1(self):
        n_point = 500
        X = np.linspace(-1., 1., n_point).reshape(-1, 1)
        Y_tru = White(f=lambda x: [np.cos(x[0] * 2 * np.pi)])(x=X)
        Y = noise(Y_tru, absolute=0.1)

        ok = True
        for backend in ['keras',
#                               'neurolab'
                               ]:
            phi = Black()
            y = phi(X=X, Y=Y, x=X.copy(), 
                    activation='sigmoid', 
                    backend=backend,
                    epochs=500 if backend == 'neurolab' else 250,
                    expected=1e-3,
                    learning_rate=0.5, 
                    neurons=[10, 10] if backend == 'neurolab' else 'auto',
                    output='sigmoid',
                    patience=30,
                    plot=True, 
    #                regularization=None,
                    show=100,
                    silent=False,
                    tolerated=10e-3,
                    trainer='auto' if backend == 'neurolab' else 'adam',
                    trials=10,
                    validation_split=0.2,
                    verbose=0,
                    )
    
            if phi.ready:            
                plt.plot(X.ravel(), Y.ravel(), '.', label='trn')
                if y is not None:
                    plt.plot(X.ravel(), y.ravel(), '.', label='tst')
                plt.legend() 
                plt.grid()
                plt.show()
            else:
                print('??? backend:', backend, 
                      ' ==> phi.ready is False')
                ok = False

        self.assertTrue(ok)


    def _test2(self):
        # neural network, 1D problem sin(x) with noise
        def f(x: Float1D, *c: float) -> List[float]:
            c0, c1 = c if len(c) > 0 else 1., 1.
            return np.sin(x) + c0 * x + c1

        # training data
        n_trn = 300
        noise_abs = 0.1
        X = grid((n_trn, 1), [-np.pi, np.pi])
        Y_tru = White(f)(x=X, silent=True)
        Y_nse = noise(Y_tru, absolute=noise_abs)

        # test data
        dx = 1. * np.pi
        n_tst = 100
        x = grid((n_tst, 1), [X.min() - dx, X.max() + dx])

        if 'grayboxes.neuralk' in sys.modules:
            neurons = [[4]*i for i in range(1, 6+1)]
        else:
            neurons = [10, 10]

        phi = Black('black')

        metrics_trn = phi(X=X, Y=Y_nse, 
                          activation='sigmoid',
                          epochs=500, 
                          expected=0.1e-3,
                          goal=1e-6,
                          neurons=neurons,
                          output='lin',
                          plot=1,
                          silent=0,
                          tolerated=5e-3,
                          trainer='auto',
                          trials=3, 
                          )
        
        y = phi(x=x, silent=True)

        if phi.ready:
            y_tru = White(f)(x=x, silent=True)
            metrics_tst = phi.evaluate(x, y_tru)
    
            plt.title('$L_2^{trn}:' + str(round(metrics_trn['L2'], 4)) + ',' +
                      ' L_2^{tst}:' + str(round(metrics_tst['L2'], 4)) + '$')
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
