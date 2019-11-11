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
      2019-08-29 DWW
"""

import __init__
__init__.init_path()

import unittest
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from grayboxes.black import Black
from grayboxes.plot import plot_isomap


class TestUM(unittest.TestCase):
    def setUp(self):
        print('///', os.path.basename(__file__))
        self.saveFigures = True

    def tearDown(self):
        pass

    def test4(self):
        # 2D problem: three 1D user-defined functions f(x) are fitted,
        # and a neural network

        df = pd.DataFrame({'x': [13, 21, 32, 33, 43, 55, 59, 60, 62, 82],
                           'y': [.56, .65, .7, .7, 2.03, 1.97, 1.92, 1.81,
                                 2.89, 7.83],
                           'u': [-0.313, -0.192, -0.145, -0.172, -0.563,
                                 -0.443, -0.408, -0.391, -0.63, -1.701]})

        def f1(x, c0=0, c1=0, c2=0, c3=0, c4=0, c5=0):
            """
            Computes polynomium: u(x) = c0 + c1*x + ... + c5*x^5
            """
            return c0+x*(c1+x*(c2+x*(c3+x*(c4+x*c5))))

        def f2(x, c0=0, c1=0, c2=0, c3=0, c4=0, c5=0):
            y = c0 / (x[0]*x[0]) + c1 / x[1] + c2 + c3*x[1] + c4 * x[0]*x[0]
            return [y]

        def f3(x, c0=0, c1=0, c2=0, c3=0, c4=0, c5=0):
            return c0 * x*x + c1 / x + c2 + c3 * x

        definitions = [f1, f2, f3,
                       [50, 10, 2],
                       f2]

        # neural network options
        opt = {'trainers': 'bfgs rprop', 'neurons': []}

        Y = np.array(df.loc[:, ['u']])            # extracts an 2D array
        for f in definitions:
            blk = Black('test4')

            if hasattr(f, '__call__'):
                print(f.__name__)
                print('f1==f', f1 == f, id(f) == id(f1))
                print('f2==f', f2 == f, id(f) == id(f2))
                print('f3==f', f3 == f, id(f) == id(f3))
            if hasattr(f, '__call__') and f2 != f:
                X = np.array(df.loc[:, ['x']])
            else:
                X = np.array(df.loc[:, ['x', 'y']])

            blk.train(X, Y, **opt)
            y = blk.predict(X)
            dy = y - Y

            # console output
            print('    ' + 76 * '-')
            su = '[j:0..' + str(Y.shape[1] - 1) + '] '
            print('    i   X[j:0..' + str(X.shape[1] - 1) + ']' +
                  'U' + su + 'u' + su + 'du' + su + 'rel' + su + '[%]:')
            for i in range(X.shape[0]):
                print('{:5d} '.format(i), end='')
                for a in X[i]:
                    print('{:f} '.format(a), end='')
                for a in Y[i]:
                    print('{:f} '.format(a), end='')
                for a in y[i]:
                    print('{:f} '.format(a), end='')
                for j in range(Y.shape[1]):
                    print('{:f} '.format(dy[i][j]), end='')
                for j in range(Y.shape[1]):
                    print('{:f} '.format(dy[i][j] / Y[i][j] * 100), end='')
                print()
            print('    ' + 76 * '-')

            # graphic presentation
            if X.shape[1] == 1 and Y.shape[1] == 1:
                plt.title('Approximation')
                plt.xlabel('$x$')
                plt.ylabel('$u$')
                plt.scatter(X, Y, label='$u$', marker='x')
                plt.scatter(X, Y, label=r'$\tilde u$', marker='o')
                if y is not None:
                    plt.plot(X, y, label=r'$\tilde u$ (cont)')
                plt.plot(X, dy, label=r'$\tilde u - u$')
                plt.legend(bbox_to_anchor=(0, 0), loc='lower left')
                plt.show()
                if 1:
                    plt.title('Absolute error')
                    plt.ylabel(r'$\tilde u - u$')
                    plt.plot(X, dy)
                    plt.show()
                if 1:
                    plt.title('Relative error')
                    plt.ylabel('E [%]')
                    plt.plot(X, dy / Y * 100)
                    plt.show()
            else:
                if isinstance(f, str):
                    s = ' (' + f + ') '
                elif not hasattr(f, '__call__'):
                    s = ' $(neural: ' + str(f) + ')$ '
                else:
                    s = ''

                plot_isomap(X[:, 0], X[:, 1], Y[:, 0],
                            labels=['$x$', '$y$', r'$u$' + s])
                plot_isomap(X[:, 0], X[:, 1], Y[:, 0],
                            labels=['$x$', '$y$', r'$\tilde u$' + s])
                plot_isomap(X[:, 0], X[:, 1], dy[:, 0],
                            labels=['$x$', '$y$', r'$\tilde u - u$' + s])

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
