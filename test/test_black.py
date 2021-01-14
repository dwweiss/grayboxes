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
      2020-12-18 DWW
"""

import initialize
initialize.set_path()

import matplotlib.pyplot as plt
import numpy as np
import unittest

try:
    from grayboxes.black import Black
except:
    from black import Black
    

class TestUM(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test1(self):
        N = 2000         # number training samples
        n = 2 * N        # number of test samples
        noise = 0.10     # absolute noise added to true values
        
        def f_true(x):   # calculates true values
            return np.sin(x * 2. * np.pi)
        
        # training data, input and target: (X, Y)
        X = np.linspace(-1., 1., N).reshape(-1, 1)
        Y_true = f_true(X)
        Y = Y_true + np.random.uniform(-noise, +noise, Y_true.shape)

        # test data, input: x
        x = np.linspace(-2., 2., n).reshape(-1, 1)
        y_true = f_true(x)

        plt.title('training data and true values')
        plt.plot(X.ravel(), Y_true.ravel(), '-', label='true')
        plt.plot(X.ravel(), Y.ravel(),      '-', label='train')
        plt.plot(x.ravel(), y_true.ravel(), ':', label='true cont.')
        plt.yscale('linear'); plt.ylim(-2., 2.); plt.legend(); plt.grid()
        plt.show()
        
        ok = True
        for backend in [
             'tensorflow',
#            'neurolab'         
            ]:
            phi = Black()
            y = phi(X=X, Y=Y, x=x,
#                activation=('leakyrelu', 'elu',)  # 'tanh', 'sigmoid', 'relu')
#                           if backend == 'tensorflow' else 'sigmoid', #'auto',
#                activation='leakyrelu' if backend == 'tensorflow' else 'sigmoid',
                activation=('leakyrelu', 'elu'),
                backend=backend,
                batch_size=[None],            # + [N // i for i in (2,10,20,)],
                epochs=250 if backend == 'tensorflow' else 150,
                expected=0.5e-3, 
                learning_rate=0.1,                             
                # neurons='auto',
#                neurons=[[nrn]*hid for hid in range(2, 20+1)          # 2, 5+1
#                                   for nrn in range(5, 5+1)],         # 2, 8+1
                neurons=[4, 6, 8, 10, 12, 10, 8, 6, 4],
                output='linear',
                patience=10,
                plot=1,    # plot=0:none, 1:plot final only, 2:plot all
                rr=0.1,    # no weight decay (regularization) if neurolab:rprop
                show=100,
                silent=False,
                tolerated=5e-3,
                trainer='adam' if backend=='tensorflow' else ('bfgs',),#'rprop'
                trials=10,
                validation_split=0.2,
                verbose=0,
                )
    
            if phi.ready:
                plt.title('train and pred, mse (trn/val): ' + 
                    str(np.round(phi.metrics['mse_trn']*1e3, 3)) + 'e-3 / ' + 
                    str(np.round(phi.metrics['mse_val']*1e3, 3)) + 'e-3')
                plt.plot(X.ravel(), Y.ravel(),      '-', label='train')
                plt.plot(x.ravel(), y.ravel(),      '-', label='test')
                plt.plot(x.ravel(), y_true.ravel(), ':', label='true')
                plt.yscale('linear')
                plt.ylim(-2., 2.)
                plt.legend() 
                plt.grid()
                plt.show()

                plt.title('training/prediction data minus true values')
                plt.plot(X.ravel(), (Y - Y_true).ravel(), '-', label='train')
                plt.plot(x.ravel(), (y - y_true).ravel(), ':', label='test')
                plt.yscale('linear')
                plt.ylim(-0.2, 0.2)
                plt.legend() 
                plt.grid()
                plt.show()                
            else:
                ok = False
                print('??? backend:', backend, '==> phi.ready is False')

        self.assertTrue(ok)


if __name__ == '__main__':
    unittest.main()
