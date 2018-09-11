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
      2018-08-30 DWW
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('..'))
from grayboxes.black import Black
from grayboxes.plotarrays import plot_isomap
from grayboxes.neural import Neural
from grayboxes.boxmodel import grid, noise
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
               'epochs': 500, 'methods': 'bfgs rprop'}

        best_trn = blk(X=X, Y=Y, **opt)
        y = blk(x=x)
        best_tst = blk.error(x, White(f)(x=x))

        plt.title('$neurons:' + str(opt['neurons']) +
                  ', L_{2}^{train}:' + str(round(best_trn['L2'], 4)) +
                  ', L_{2}^{test}:' + str(round(best_tst['L2'], 4)) + '$')
        plt.cla()
        plt.ylim(min(-2, Y.min(), y.min()), max(2, Y.max(), Y.max()))
        plt.yscale('linear')
        plt.xlim(-0.1 + x.min(), 0.1 + x.max())
        plt.scatter(X, Y, marker='x', c='r', label='training data')
        plt.plot(x, y, c='b', label='prediction')
        plt.plot(x, f(x), linestyle=':', label='analytical')
        i_abs_trn = best_trn['iAbs']
        plt.scatter([X[i_abs_trn]], [Y[i_abs_trn]], marker='o', color='r',
                    s=66, label='max abs train')
        i_abs_tst = best_tst['iAbs']
        plt.scatter([x[i_abs_tst]], [y[i_abs_tst]], marker='o', color='b',
                    s=66, label='max abs test')
        plt.legend(bbox_to_anchor=(1.1, 0), loc='lower left')
        plt.grid()
        plt.show()

        self.assertTrue(True)

    def test3(self):
        # 1D problem sin(x) with noise, variation of network structure
        path = Black().path

        print('*' * 30, 'path:', path)

        file = 'sin_x_-3..3.5pi'
        n_point = 20
        max_neurons_in_layer = 2  # 16
        n_hidden_layers = 1  # 3
        MAX_HIDDEN_LAYERS = 6  # MUST NOT be modified
        assert MAX_HIDDEN_LAYERS >= n_hidden_layers
        max_noise = 0.0

        # compute training target ('X', 'U') and test data ('x', 'uAna')
        def f(x, *args):
            c0, c1 = args if len(args) > 0 else 1, 1
            return np.sin(x) * c0 + c1
        X = np.linspace(-2*np.pi, +2*np.pi, n_point)    # X of train data
        if 0:
            X = np.flipud(X)                        # reverse order of X
        x = np.linspace(-3*np.pi, 3*np.pi, 100)
        dx = (X.max() - X.min()) / n_point
        x = x + 0.5 * dx                          # shift x of test data

        Y = f(X)
        if max_noise > 0.0:
            Y += np.random.normal(-max_noise, +max_noise, X.size)
        X, Y = X.reshape(X.size, 1), Y.reshape(Y.size, 1)
        x = x.reshape(x.size, 1)
        y_ref = f(x)

        # define 'collect' as DataFrame for result collection
        columns = ['n' + str(i+1) for i in range(MAX_HIDDEN_LAYERS)]
        columns.extend(['L2Train', 'absTrain', 'iAbsTrain',
                        'L2Test', 'absTest', 'iAbsTest',
                        'mse', 'method', 'epochs'])
        collect = pd.DataFrame(columns=columns)
        definition_max = [max_neurons_in_layer for _ in range(n_hidden_layers)]
        definition_max = definition_max + [0] * (MAX_HIDDEN_LAYERS -
                                               n_hidden_layers)
        print('definition_max:', definition_max)

        definitions = []
        for n1 in range(1, 1+definition_max[0]):
            for n2 in range(0, 1+min(n1, definition_max[1])):
                for n3 in range(0, 1+min(n2, definition_max[2])):
                    for n4 in range(0, 1+min(n3, definition_max[3])):
                        for n5 in range(0, 1+min(n4, definition_max[4])):
                            for n6 in range(0, 1+min(n5, definition_max[5])):
                                definition = list([int(n1)])
                                if n2 > 0:
                                    definition.append(int(n2))
                                if n3 > 0:
                                    definition.append(int(n3))
                                if n4 > 0:
                                    definition.append(int(n4))
                                if n5 > 0:
                                    definition.append(int(n5))
                                if n6 > 0:
                                    definition.append(int(n6))
                                print('+++ generate hidden:', definition)
                                definitions.append(definition)
        print('+++ definitions:', definitions)

        l2_tst_best, i_def_best = 1.0, 0
        for i_def, definition in enumerate(definitions):
            definition_copy = definition.copy()
            print('+++ hidden layers:', definition)

            # network training
            blk = Black()
            best_trn = blk(X=X, Y=Y, neurons=definition, trials=5, epochs=500,
                          show=500, algorithms='bfgs', goal=1e-5)

            # network prediction
            y = blk.predict(x=x)

            best_tst = blk.error(x, y_ref, silent=False)
            if l2_tst_best > best_tst['L2']:
                l2_tst_best = best_tst['L2']
                i_def_best = i_def
            row = definition_copy.copy()
            row = row + [0]*(MAX_HIDDEN_LAYERS - len(row))
            row.extend([best_trn['L2'], best_trn['abs'], best_trn['iAbs'],
                        best_tst['L2'], best_tst['abs'], best_tst['iAbs'],
                        0, 0, 0  # mse training method epochs
                        ])
            # print('row:', row, len(row), 'columns:', collect.keys)
            collect.loc[collect.shape[0]] = row

            if isinstance(blk._empirical, Neural):
                print('+++ neural network definition:', definition)
            plt.title('$' + str(definition_copy) + '\ \ L_2(tr/te):\ ' +
                      str(round(best_trn['L2'], 5)) + r', ' +
                      str(round(best_tst['L2'], 4)) +
                      '$')
            plt.xlim(x.min() - 0.25, x.max() + 0.25)
            plt.ylim(-2, 2)
            plt.grid()
            plt.scatter(X, Y, marker='>', c='g', label='training data')
            plt.plot(x, y, linestyle='-', label='prediction')
            plt.plot(x, y_ref, linestyle=':', label='analytical')
            plt.scatter([X[best_trn['iAbs']]], [Y[best_trn['iAbs']]], marker='o',
                        color='c', s=60, label='max err train')
            plt.scatter([x[best_tst['iAbs']]], [y[best_tst['iAbs']]], marker='o',
                        color='r', s=60, label='max err test')
            plt.legend(bbox_to_anchor=(1.15, 0), loc='lower left')
            if self.saveFigures:
                f = file
                for s in definition_copy:
                    f += '_' + str(s)
                p = os.path.join(path, f + '.png')
                plt.savefig(p)
            plt.show()

            print('+++ optimum: definition: ', definitions[i_def_best],
                  ' index: [', i_def_best, '], L2: ',
                  round(l2_tst_best, 5), sep='')
            if i_def % 10 == 0 or i_def == len(definitions) - 1:
                print('+++ collect:\n', collect)
                collect.to_csv(os.path.join(path, file + '.collect.csv'))
                collect.to_pickle(os.path.join(path, file + '.collect.pkl'))
                collect.plot(y='L2Test', use_index=True)

        self.assertTrue(True)

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
        opt = {'methods': 'bfgs rprop', 'neurons': []}

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
