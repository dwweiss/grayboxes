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

import unittest
import numpy as np
import matplotlib.pyplot as plt

from grayboxes.array import grid
from grayboxes.neuralnl import Neural as NeuralNl
from grayboxes.neuraltf import Neural as NeuralTf
from grayboxes.plot import (plot_surface, plot_isolines, plot_isomap, \
                            plot_wireframe)
try:
    dummy = tf
except:
    print("!!! Module 'tensorflow' not imported")
    print('    Set in Spyder: tools->Preferences->IPython console->')
    print("        Startup->'import tensorflow'")
    print('    or restart kernel and enter in IPython console:')
    print('        $ import tensorflow as tf') 


class TestUM(unittest.TestCase):
    """
    Comparison of implementations of feed forward neural networks:
        - Neurolab: NeuralN()
        - Keras (Tensorflow):: NeuralK()
    """
    
    def setUp(self):
        pass


    def tearDown(self):
        pass


    def _test1(self):
        s = 'c0 + c1 + sin(x) + noise'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        def f(x):
            return np.cos(x) * 5 + 2

        n_point = 2000
        X = np.linspace(-2*np.pi, 2*np.pi, n_point).reshape(-1, 1)

        plt.title('Y vs Y + noise')
        Y = f(X)
        Y += 5e-2 * np.random.uniform(-1, 1, size=Y.shape) * (Y.max()-Y.min())        
        
        plt.plot(X, f(X), ':', label='f(X)')
        plt.plot(X, Y, label='Y+noise')
        plt.legend()
        plt.show()
        
        # test data range is training data range +/- dx
        dx = 0.0 * (X.max() - X.min())
        x = np.linspace(X.min() - dx, X.max() + dx, 50).reshape(-1, 1)
        

        for net in (
#                    NeuralNl(),
                    NeuralTf(),
                    ):
            if isinstance(net, NeuralTf):
                trainer = (
                           'adam', 
#                           'nadam'
#                           'rmsprop', 
#                           'sgd', 
                           )
                activation=(
#                            'elu',
#                            'relu',
#                            'sigmoid',
                            'tanh',
                             )                
            else:
                trainer = (
#                           'bfgs', 
#                           'rprop',
                           )
                activation='tanh'
                
            print('x shape', x.shape)

            y = net(X=X, Y=Y, x=x, 
                    activation=activation, 
                    epochs=500, 
                    expected=1e-4,
                    learning_rate=0.1, 
                    neurons=[[10]*i for i in range(3, 3+1)], 
                    output='linear',
                    patience=30,
                    plot=True, 
                    tolerated=5e-3,
                    trainer=trainer,
                    trials=3,
                    regularization=None,
                    show=None,
                    validation_split=0.25,
                    verbose=0,
                    )

            if not net.ready:
                print('??? network not ready')
                print('+++ y:', y)
            else:                
                plt.title('Test, L2:' + str(round(net.metrics['L2'], 5)))
                plt.plot(x.reshape(-1), y.reshape(-1), '-')
                plt.plot(X.reshape(-1), Y.reshape(-1), '.')
                plt.legend(['pred', 'targ', ])
                plt.xlabel('x')
                plt.ylabel('y(x)')
                plt.show()

        self.assertTrue(True)


    def test2(self):
        s = 'Test 2, 2D input and 2D target'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        X = grid((256,)*2, [0, 1]*2)
        Y = np.sin(X * 2 * np.pi)
        print('+++ shapes X Y:', X.shape, Y.shape)
        x = X.copy()

        net = NeuralTf()
        y = net(X, Y, x, 
                neurons='auto', 
                plot=1, epochs=250, expected=1e-3,
                trainer='adam', trials=3, activation='tanh', output='tanh',
                validation_split=0.2)
        dy = y - Y
        X, Y, x = net.X, net.Y, net.x
        if X.shape[1] == 2:
            plot_wireframe(X[:, 0], X[:, 1], y[:, 0], title='$y_{prd}$',
                           labels=['$x_0$', '$x_1$', r'$Y_{trg}$'])
            plot_wireframe(X[:, 0], X[:, 1], Y[:, 0], title='$Y_{trg}$',
                           labels=['$x_0$', '$x_1$', r'$Y_{trg}$'])
            plot_wireframe(X[:, 0], X[:, 1], dy[:, 0], title=r'$\Delta y$',
                           labels=['$x_0$', '$x_1$', r'$\Delta y$'])
            plot_isolines(X[:, 0], X[:, 1], y[:, 0], labels=['$x_0$', '$x_1$'], 
                          title='$y_{prd}$')
            plot_isomap(X[:, 0], X[:, 1], y[:, 0], labels=['$x_0$', '$x_1$'], 
                        title='$y_{prd}$')
            plot_isomap(X[:, 0], X[:, 1], Y[:, 0], labels=['$x_0$', '$x_1$'], 
                        title='$Y_{trg}$')
            plot_isolines(X[:, 0], X[:, 1], Y[:, 0], labels=['$x_0$', '$x_1$'], 
                          title='$Y_{trg}$')
            plot_isomap(X[:, 0], X[:, 1], dy[:, 0], labels=['$x_0$', '$x_1$'], 
                        title=r'$\Delta y$')
            plot_surface(X[:, 0], X[:, 1], dy[:, 0], labels=['$x_0$', '$x_1$'], 
                         title=r'$\Delta y$')
            plot_surface(X[:, 0], X[:, 1], y[:, 0], labels=['$x_0$', '$x_1$'], 
                         title='$y_{prd}$')

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
