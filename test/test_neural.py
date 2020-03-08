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
      2020-02-13 DWW
"""

import initialize
initialize.set_path()

import unittest
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import neurolab as nl

from grayboxes.neuraltf import Neural as NeuralTf
from grayboxes.neuralnl import Neural as NeuralNl

from grayboxes.plot import (plot_surface, plot_isolines, 
                            plot_isomap, plot_wireframe)


def L2(y: np.ndarray, Y: np.ndarray) -> float:
    return np.sqrt(np.mean(np.square(y - Y)))


def str_L2(y: np.ndarray, Y: np.ndarray) -> str:
    return str(np.round(L2(y, Y), 4))


class TestUM(unittest.TestCase):
    def setUp(self):
        pass


    def tearDown(self):
        pass


    def _test1(self):
        s = 'Example 1: newff and train from Neurolab'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        X = np.linspace(-7, 7, 20).reshape(-1, 1)
        Y = np.sin(X) * 10

        norm_y = nl.tool.Norm(Y)
        YY = norm_y(Y)
        net = nl.net.newff(nl.tool.minmax(X), [5, YY.shape[1]])
        # net.trainf = nl.train.train_rprop  # or:
        net.trainf = nl.train.train_bfgs

        mse_seq = net.train(X, YY, epochs=10000, show=100, goal=1e-6)
        mse = mse_seq[-1]
        y_trn = norm_y.renorm(net.sim(X))

        print(mse)
        plt.subplot(211)
        plt.plot(mse_seq, label='mse')
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('metrics')

        x_tst = np.linspace(-12, 8, 150).reshape(-1, 1)
        y_tst = norm_y.renorm(net.sim(x_tst)).ravel()

        plt.subplot(212)
        plt.plot(x_tst, y_tst, '-')
        plt.plot(X, Y, '.')
        plt.legend(['pred', 'targ'])
        plt.xlabel('$x$')
        plt.ylabel('$y(x)$')
        plt.show()

        self.assertTrue(True)


    def _test2(self):
        s = 'Example 2 __call__()'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        def f(x):
            return np.sin(x) * 1 + 0

        X = np.linspace(-1.75 * np.pi, 1.75 * np.pi, 50).reshape(-1, 1)
        dx = 0.25 * (X.max() - X.min())
        x = np.linspace(X.min() - dx, X.max() + dx).reshape(-1, 1)
        X[0, 0] = x[0, 0]
        X[-1, 0] = x[-1, 0]
        Y = f(X)

        class Train2(nl.core.Train):
            def error(self, net, inp, target, out=None):
                """Only for train with teacher"""
                if out is None:
                    out = net.sim(inp)
                return net.errorf(target, out)

        class TanSig2:
            out_minmax = [-1, 1]
            inp_active = [-2, 2]

            def __call__(self, x):
                return np.sin(x)
                # return np.tanh(x)

            def deriv(self, x, y):
                delta, reciprocal = 1e-6, 1e6
                return (self(y + delta) - self(y)) * reciprocal

        net = NeuralNl()
        for outputf in (nl.trans.PureLin, ):  # TanSig2, ):
            net(X=X, Y=Y, neurons=[6], epochs=2000, goal=1e-6, show=0,
                trials=3, trainer='rprop', regularization=0.0, plot=0,
                # outputf=nl.trans.PureLin,    TODO fails with 'invalid output'
                # errorf=nl.error.MSE,
                silent=True)
            y = net(x=x)
            if y is not None:
                plt.plot(x, y, '-',
                         label='tst:'+str_L2(net(x=x), f(x)) + ' ' +
                               'trn:'+str_L2(net(x=X), Y))
        plt.title('Test (' + net.metrics['trainer'] + ') ' +
                  'L2_trn: ' + str(round(net.metrics['L2'], 2)))
        plt.plot(x, f(x), '--', label='tst')
        plt.plot(X, Y, 'o', label='trn')
        # plt.legend(['pred', 'targ', 'true'])
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y(x)')
        plt.show()

        self.assertTrue(True)


    def _test3(self):
        s = 'Example 3 compact form'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        def f(x):
            return np.sin(x) * 10 + 0

        X = np.linspace(-1.75 * np.pi, 1.75 * np.pi, 50).reshape(-1, 1)
        Y = f(X)
        dx = 0.5 * (X.max() - X.min())
        x = np.linspace(X.min() - dx, X.max() + dx).reshape(-1, 1)
        
        net = NeuralTf()
        y = net(X=X, Y=Y, x=x, neurons=[6], plot=1, epochs=500, goal=1e-5,
                trials=5, trainer='cg gdx rprop bfgs',
                regularization=0.0, show=None,
                )

        if net.ready:
            plt.title('Test, L2:' + str(round(net.metrics['L2'], 5)))
            plt.plot(x.ravel(), y.ravel(), '-')
            plt.plot(X, Y, '.')
            plt.legend(['pred', 'targ', ])
            plt.xlabel('x')
            plt.ylabel('y(x)')
            plt.show()

        self.assertTrue(True)


    def _test4(self):
        s = 'Example 4, get X and Y from dataframe'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        df = DataFrame({'p0': [10, 20, 30, 40], 'p1': [11, 21, 31, 41],
                        'p2': [12, 22, 32, 42], 'r0': [31, 41, 51, 52],
                        'r1': [32, 42, 52, 55]})
        xkeys = ['p0', 'p2']
        ykeys = ['r0', 'r1']
        net = NeuralTf()
        net.set_XY(df[xkeys], df[ykeys], xkeys, ykeys)
        metrics = net.train(X=None, Y=None, goal=1e-6, neurons=[10, 3], 
                            plot=1, epochs=2000,
                            trainer='cg gdx rprop bfgs', trials=10,
                            regularization=0.01, smartTrials=False)

        print('Metrics:', metrics)

        self.assertTrue(True)

    def _test5(self):
        s = 'Example 5, X and Y are 2D arrays'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        X = [[10, 11], [11, 33], [33, 14], [37, 39], [20, 20]]
        Y = [[10, 11], [12, 13], [35, 40], [58, 68], [22, 28]]
        x = X.copy()

        net = NeuralTf()
        y = net(X, Y, x, neurons=[10, 10], 
                activation='sigmoid', 
                epochs=1000, 
                expected=0.5e-4,
                output='sigmoid',
                plot=1, 
                tolerated=5e-3,
                trainer='auto', 
                trials=5, 
                validation_split=0.0,
                )
        plt.title('Y - net.Y')
        plt.plot((np.asfarray(Y) - net.Y).ravel())
        plt.show()
        
        if net.ready:
            dy = y - Y

            X = net.X
            Y = net.Y
            x = net.x
#            X, Y, x = net.X, net.Y, net.x
#            X, Y, x = net.X, net.Y, net.x
            
#            X = np.asfarray(X)
#            Y = np.asfarray(Y)
#            x = np.asfarray(x)
            
            X0, X1, Y0, y0, dy0 = X[:, 0], X[:, 1], Y[:, 0], y[:, 0], dy[:, 0]
            
            if X.shape[1] == 2:
                plot_wireframe(X0, X1, y0, title='$y_{prd}$',
                               labels=['x', 'y', r'$Y_{trg}$'])
                plot_wireframe(X0, X1, Y0, title='$Y_{trg}$',
                               labels=['x', 'y', r'$Y_{trg}$'])
                plot_wireframe(X0, X1, dy0, title=r'$\Delta y$',
                               labels=['x', 'y', r'$\Delta y$'])
                plot_isolines(X0, X1, y0, title='$y_{prd}$')
                plot_isomap(X0, X1, y0, title='$y_{prd}$')
                plot_isomap(X0, X1, Y0, title='$Y_{trg}$')
                plot_isolines(X0, X1, Y0, title='$Y_{trg}$')
                plot_isomap(X0, X1, dy0, title=r'$\Delta y$')
                plot_surface(X0, X1, dy0, title=r'$\Delta y$')
                plot_surface(X0, X1, y0, title='$y_{prd}$')

        self.assertTrue(net.ready)


    def test6(self):
        s = 'Example 6'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        N = 2000
        n = 300
        m = 1   
        nse = 0e-2
        X = np.random.uniform(-2*np.pi, 2*np.pi, size=(N, m))
        Y_tru = np.sin(X)
        Y = Y_tru + np.random.uniform(-nse, +nse, size=X.shape)
        x = np.random.uniform(-2*np.pi, 2*np.pi, size=(n, m))
        y_tru = np.sin(x)
    
        for phi in (
#                    Neural, 
#                    NeuralNl, 
                    NeuralTf,
                    ):
            phi = phi()
            y = phi(X=X, Y=Y, x=x,
                    activation='tanh',
                    epochs=150,
                    expected=1e-3, 
                    learning_rate=0.1,
                    neurons=[[i]*j for i in range(3, 6) for j in range(1, 5)],
                    output=None,
                    patience=25,
                    plot=1, 
                    tolerated=5e-3,
                    trainer='adam', 
                    trials=5,
                    )
            
            if phi.ready:
                dy = y - y_tru
                
                X0, Y0, x0, y0, dy0 = X[:,0], Y[:,0], x[:,0], y[:,0], dy[:,0]
                if X.shape[1] > 1:
                    X1 = X[:, 1]
                    if X.shape[1] > 2:
                        X2 = X[:, 2]
                if x.shape[1] > 1:
                    x1 = x[:, 1]
                    if X.shape[1] > 2:
                        x2 = x[:, 2]
                if Y.shape[1] > 1:
                    Y1 = Y[:, 1]
                    if Y.shape[1] > 2:
                        Y2 = Y[:, 2]
        
                if X.shape[1] == 1:
                    plt.plot(x0, y0, label='pred')
                    plt.plot(X0, Y0, label='targ')
                    plt.legend()
                    plt.show()
                elif X.shape[1] == 2:
                    plot_surface(x0, x1, y0, title='$y_{prd}$')
                    plot_isolines(X0, X1, y0, title='$y_{prd}$')
                    plot_isolines(X0, X1, Y0, title='$Y_{trg}$')
                if X.shape[1] == 2:
                    plot_isomap(x0, x1, dy0, title='$y_{prd} - Y_{trg}$')

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
