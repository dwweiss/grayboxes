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
      2019-11-22 DWW
"""

import initialize
initialize.set_path()

import unittest
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import neurolab as nl

from grayboxes.neural import Neural
from grayboxes.plot import (plot_surface, plot_isolines, plot_isomap, \
                            plot_wireframe)


def L2(y: np.ndarray, Y: np.ndarray) -> float:
    return np.sqrt(np.mean(np.square(y - Y)))


def str_L2(y: np.ndarray, Y: np.ndarray) -> str:
    return str(np.round(L2(y, Y), 4))


class TestUM(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test1(self):
        s = 'Example 1 __call__()'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        def f(x):
            return np.sin(x) * 1 + 0

        X = np.atleast_2d(np.linspace(-1.75 * np.pi, 1.75 * np.pi, 50)).T
        dx = 0.25 * (X.max() - X.min())
        x = np.atleast_2d(np.linspace(X.min() - dx, X.max() + dx)).T
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

        net = Neural()
        for outputf in (nl.trans.PureLin, ):  # TanSig2, ):
            net(X=X, Y=Y, neurons=[6], epochs=2000, goal=1e-6, show=0,
                trials=3, trainer='rprop', regularization=0.0, plot=0,
                # outputf=nl.trans.PureLin,    TODO fails with 'invalid output'
                # errorf=nl.error.MSE,
                silent=True)
            y = net(x=x)
            if 1:
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

    def test2(self):
        s = 'Example 2 compact form'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        def f(x):
            return np.sin(x) * 10 + 0

        X = np.atleast_2d(np.linspace(-1.75 * np.pi, 1.75 * np.pi, 50)).T
        Y = f(X)
        dx = 0.5 * (X.max() - X.min())
        x = np.atleast_2d(np.linspace(X.min() - dx, X.max() + dx)).T

        net = Neural()
        y = net(X=X, Y=Y, x=x, neurons=[6], plot=1, epochs=500, goal=1e-5,
                trials=5, trainer='cg gdx rprop bfgs',
                regularization=0.0, show=None)

        plt.title('Test, L2:' + str(round(net.metrics['L2'], 5)))
        plt.plot(x, y, '-')
        plt.plot(X, Y, '.')
        plt.legend(['pred', 'targ', ])
        plt.xlabel('x')
        plt.ylabel('y(x)')
        plt.show()

        self.assertTrue(True)

    def test3(self):
        s = 'Example 3'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        df = DataFrame({'p0': [10, 20, 30, 40], 'p1': [11, 21, 31, 41],
                        'p2': [12, 22, 32, 42], 'r0': [31, 41, 51, 52],
                        'r1': [32, 42, 52, 55]})
        xkeys = ['p0', 'p2']
        ykeys = ['r0', 'r1']
        net = Neural()
        net.import_dataframe(df, xkeys, ykeys)
        metrics = net.train(goal=1e-6, neurons=[10, 3], plot=1, epochs=2000,
                            trainer='cg gdx rprop bfgs', trials=10,
                            regularization=0.01, smartTrials=False)

        self.assertTrue(True)

    def test4(self):
        s = 'Example 4'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        X = [[10, 11], [11, 33], [33, 14], [37, 39], [20, 20]]
        Y = [[10, 11], [12, 13], [35, 40], [58, 68], [22, 28]]
        x = X.copy()

        net = Neural()
        y = net(X, Y, x, neurons=6, plot=1, epochs=1000, goal=1e-6,
                trainer='cg gdx rprop bfgs', trials=5)
        dy = y - Y
        X, Y, x = net.X, net.Y, net.x
        if X.shape[1] == 2:
            plot_wireframe(X[:, 0], X[:, 1], y[:, 0], title='$y_{prd}$',
                           labels=['x', 'y', r'$Y_{trg}$'])
            plot_wireframe(X[:, 0], X[:, 1], Y[:, 0], title='$Y_{trg}$',
                           labels=['x', 'y', r'$Y_{trg}$'])
            plot_wireframe(X[:, 0], X[:, 1], dy[:, 0], title=r'$\Delta y$',
                           labels=['x', 'y', r'$\Delta y$'])
            plot_isolines(X[:, 0], X[:, 1], y[:, 0], title='$y_{prd}$')
            plot_isomap(X[:, 0], X[:, 1], y[:, 0], title='$y_{prd}$')
            plot_isomap(X[:, 0], X[:, 1], Y[:, 0], title='$Y_{trg}$')
            plot_isolines(X[:, 0], X[:, 1], Y[:, 0], title='$Y_{trg}$')
            plot_isomap(X[:, 0], X[:, 1], dy[:, 0], title=r'$\Delta y$')
            plot_surface(X[:, 0], X[:, 1], dy[:, 0], title=r'$\Delta y$')
            plot_surface(X[:, 0], X[:, 1], y[:, 0], title='$y_{prd}$')

        self.assertTrue(True)

    def test5(self):
        s = 'Example 5: newff and train without class Neural'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        X = np.atleast_2d(np.linspace(-7, 7, 20)).T
        Y = np.sin(X) * 10

        norm_y = nl.tool.Norm(Y)
        YY = norm_y(Y)
        net = nl.net.newff(nl.tool.minmax(X), [5, YY.shape[1]])
        # net.trainf = nl.train.train_rprop  # or:
        net.trainf = nl.train.train_bfgs

        err = net.train(X, YY, epochs=10000, show=100, goal=1e-6)
        y_trn = norm_y.renorm(net.sim(X))

        print(err[-1])
        plt.subplot(211)
        plt.plot(err)
        plt.legend(['L2 error'])
        plt.xlabel('Epoch number')
        plt.ylabel('error (default SSE)')

        x_tst = np.atleast_2d(np.linspace(-5, 8, 150)).T
        y_tst = norm_y.renorm(net.sim(x_tst)).ravel()

        plt.subplot(212)
        plt.plot(x_tst, y_tst, '-', X, Y, '.')
        plt.legend(['pred', 'targ'])
        plt.xlabel('x')
        plt.ylabel('y(x)')
        plt.show()

        self.assertTrue(True)

    def test6(self):
        s = 'Example 6'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        X = np.atleast_2d(np.linspace(-2 * np.pi, 2 * np.pi, 50)).T
        Y = np.sin(X) * 5
        x = X
        y = Neural()(X=X, Y=Y, x=x, neurons=[8, 2], plot=1, epochs=2000,
                     goal=1e-5, trainer='rprop bfgs', trials=8)

        if X.shape[1] == 1:
            plt.plot(X, y, label='pred')
            plt.plot(X, Y, label='targ')
            plt.legend()
            plt.show()
        elif X.shape[1] == 2:
            plot_surface(X[:, 0], X[:, 1], y[:, 0], title='$y_{prd}$')
            plot_isolines(X[:, 0], X[:, 1], y[:, 0], title='$y_{prd}$')
            plot_isolines(X[:, 0], X[:, 1], Y[:, 0], title='$y_{trg}$')
        dy = y - Y
        if X.shape[1] == 2:
            plot_isomap(X[:, 0], X[:, 1], dy[:, 0],
                        title='$y_{prd} - y_{trg}$')

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
