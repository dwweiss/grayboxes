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
      2018-05-20 DWW
"""

import numpy as np
import pandas as pd

from Model import Model
from Neural import Neural


class Black(Model):
    """
    Black box model y = F^*(x, w), w = train(X, Y, f(x))

        - Neural network is employed if argument 'neurons' is passed by kwargs
          Best neural network of all trials is saved as 'self._empirical._net'
        - Splines are employed if the argument 'splines' is passed by kwargs

    Example:
        X = np.linspace(0.0, 1.0, 20)
        x = X * 2
        Y = X**2

        # black box, neural network, expanded:
        model = Black()
        model(X=X, Y=Y, neurons=[2, 3])
        yTrn = model(x=X)
        yTst = model(x=x)

        # black box, neural network, compact:
        y = Black()(XY=(X, Y), neurons=[2, 3], x=x)
    """

    def __init__(self, identifier='Black'):
        """
        Args:
            identifier (str, optional):
                object identifier
        """
        super().__init__(f=None, identifier=identifier)
        self._empirical = None         # holds instance of Neural, splines etc.

    @property
    def silent(self):
        return self._silent

    @silent.setter
    def silent(self, value):
        self._silent = value
        if self._empirical is not None:
            self._empirical._silent = value

    def train(self, X, Y, **kwargs):
        """
        Trains model, stores X and Y as self.X and self.Y, and stores result
        of best training trial as self.best

        Args:
            X (2D or 1D array_like of float):
                training input, shape: (nPoint, nInp) or shape: (nPoint)

            Y (2D or 1D array_like of float):
                training target, shape: (nPoint, nOut) or shape: (nPoint)

            kwargs (dict, optional):
                keyword arguments:

                neurons (int or 1D array_like of int):
                    number of neurons in the hidden layers of neural network

                splines (1D array_like of float:
                    not specified yet

                ... additional training options of network, see Neural.train()

        Returns:
            see Model.train()
        """
        if X is None or Y is None:
            return None

        self.X, self.Y = np.atleast_2d(X), np.atleast_2d(Y)
        if self._X.shape[0] == 1:
            self._X = self._X.T
        if self._Y.shape[0] == 1:
            self._Y = self._Y.T
        assert self.X.shape[0] == self.Y.shape[0], \
            str(self.X.shape) + str(self.Y.shape)
        assert self.X.shape[0] > 2, str(self.X.shape)
        assert self.Y.shape[0] > 2, str(self.Y.shape)

        neurons = kwargs.get('neurons', None)
        splines = kwargs.get('splines', None) if neurons is None else None

        if neurons is not None:
            self.write('+++ train neural, hidden neurons:', neurons)

            if self._empirical is not None:
                del self._empirical
            self._empirical = Neural()
            self._empirical.silent = self.silent

            self.best = self._empirical.train(self.X, self.Y, **kwargs)
            self.ready = self._empirical.ready

        elif splines is not None:
            # ...
            self.best = None
            self.ready = False
            assert 0, 'splines not implemented'

        else:
            self.best = None
            self.ready = False
            assert 0, 'unknown empirical method'

        return self.best

    def predict(self, x, **kwargs):
        """
        Executes Model, stores input x as self.x and output as self.y

        Args:
            x (2D or 1D array_like of float):
                prediction input, shape: (nPoint, nInp) or shape: (nInp)

            kwargs (dict, optional):
                keyword arguments

        Returns:
            (2D array of float):
                prediction output, shape: (nPoint, nOut)
        """
        assert self.ready

        self.x = x
        self.y = self._empirical.predict(self.x, **kwargs)
        return self.y


# Examples ###################################################################

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from White import White
    import Model as md

    print('*** main')

    ################

    def example1():
        noise = 0.1
        X = md.grid(-20, [0, 1])
        Y = md.noise(White(lambda x: [x[0]**2])(x=X), absolute=noise)

        y = Black()(X=X, Y=Y, x=X, neurons=[], silent=True)

        plt.plot(X, Y, '.', X, y, '-')
        plt.show()

    ################

    def example2():
        # neural network, 1D problem sin(x) with noise
        def f(x, *args):
            a, b = args if len(args) > 0 else 1, 1
            return np.sin(x) + a * x + b

        # training data
        nPointTrn = 20
        noise = 0.01
        X = np.linspace(-1 * np.pi, +1 * np.pi, nPointTrn)
        Y = f(X)
        X, Y = np.atleast_2d(X).T, np.atleast_2d(Y).T
        Y_nse = Y.copy()
        if noise > 0.0:
            Y_nse += np.random.normal(-noise, +noise, Y_nse.shape)

        # test data
        dx = 0.5 * np.pi
        nPointTst = nPointTrn
        x = np.atleast_2d(np.linspace(X.min()-dx, X.max()+dx, nPointTst)).T

        blk = Black()
        opt = {'neurons': [10, 10], 'trials': 5, 'goal': 1e-6,
               'epochs': 500, 'methods': 'bfgs rprop'}

        bestTrn = blk(X=X, Y=Y, **opt)
        y = blk(x=x)
        bestTst = blk.error(x, White(f)(x=x))

        plt.title('$neurons:' + str(opt['neurons']) +
                  ', L_{2}^{train}:' + str(round(bestTrn['L2'], 4)) +
                  ', L_{2}^{test}:' + str(round(bestTst['L2'], 4)) + '$')
        plt.cla()
        plt.ylim(min(-2, Y.min(), y.min()), max(2, Y.max(), Y.max()))
        plt.yscale('linear')
        plt.xlim(-0.1 + x.min(), 0.1 + x.max())
        plt.scatter(X, Y, marker='x', c='r', label='training data')
        plt.plot(x, y, c='b', label='prediction')
        plt.plot(x, f(x), linestyle=':', label='analytical')
        iAbsTrn = bestTrn['iAbs']
        plt.scatter([X[iAbsTrn]], [Y[iAbsTrn]], marker='o', color='r',
                    s=66, label='max abs train')
        iAbsTst = bestTst['iAbs']
        plt.scatter([x[iAbsTst]], [y[iAbsTst]], marker='o', color='b',
                    s=66, label='max abs test')
        plt.legend(bbox_to_anchor=(1.1, 0), loc='lower left')
        plt.grid()
        plt.show()

    ################

    def example3():
        # 1D problem sin(x) with noise, variation of neural network structure
        saveFigures = True
        path = Black().path
        file = 'sin_x_-3..3.5pi'
        nPoint = 20
        maxNeuronsInLayer = 2  # 16
        nHiddenLayers = 1  # 3
        MAX_HIDDEN_LAYERS = 6  # MUST NOT be modified
        assert MAX_HIDDEN_LAYERS >= nHiddenLayers
        maxNoise = 0.0

        # compute training target ('X', 'U') and test data ('x', 'uAna')
        def f(x, *args):
            c0, c1 = args if len(args) > 0 else 1, 1
            return np.sin(x) * c0 + c1
        X = np.linspace(-2*np.pi, +2*np.pi, nPoint)   # argument of train data
        if 0:
            X = np.flipud(X)                   # reverse order of elements of X
        x = np.linspace(-3*np.pi, 3*np.pi, 100)
        dx = (X.max() - X.min()) / nPoint
        x = x + 0.5 * dx          # shift argument for calculation of test data

        Y = f(X)
        if maxNoise > 0.0:
            Y += np.random.normal(-maxNoise, +maxNoise, X.size)
        X, Y = X.reshape(X.size, 1), Y.reshape(Y.size, 1)
        x = x.reshape(x.size, 1)
        yRef = f(x)

        # define 'collect' as DataFrame for result collection
        columns = ['n' + str(i+1) for i in range(MAX_HIDDEN_LAYERS)]
        columns.extend(['L2Train', 'absTrain', 'iAbsTrain',
                        'L2Test', 'absTest', 'iAbsTest',
                        'mse', 'method', 'epochs'])
        collect = pd.DataFrame(columns=columns)
        definitionMax = [maxNeuronsInLayer for i in range(nHiddenLayers)]
        definitionMax = definitionMax + [0] * (MAX_HIDDEN_LAYERS -
                                               nHiddenLayers)
        print('definitionMax:', definitionMax)

        definitions = []
        for n1 in range(1, 1+definitionMax[0]):
            for n2 in range(0, 1+min(n1, definitionMax[1])):
                for n3 in range(0, 1+min(n2, definitionMax[2])):
                    for n4 in range(0, 1+min(n3, definitionMax[3])):
                        for n5 in range(0, 1+min(n4, definitionMax[4])):
                            for n6 in range(0, 1+min(n5, definitionMax[5])):
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

        L2TstBest, iDefBest = 1.0, 0
        for iDef, definition in enumerate(definitions):
            definitionCopy = definition.copy()
            print('+++ hidden layers:', definition)

            # network training
            blk = Black()
            bestTrn = blk(X=X, Y=Y, neurons=definition, trials=5, epochs=500,
                          show=500, algorithms='bfgs', goal=1e-5)

            # network prediction
            y = blk.predict(x=x)

            bestTst = blk.error(x, yRef, silent=False)
            if L2TstBest > bestTst['L2']:
                L2TstBest = bestTst['L2']
                iDefBest = iDef
            row = definitionCopy.copy()
            row = row + [0]*(MAX_HIDDEN_LAYERS - len(row))
            row.extend([bestTrn['L2'], bestTrn['abs'], bestTrn['iAbs'],
                        bestTst['L2'], bestTst['abs'], bestTst['iAbs'],
                        0, 0, 0  # mse training method epochs
                        ])
            # print('row:', row, len(row), 'columns:', collect.keys)
            collect.loc[collect.shape[0]] = row

            if isinstance(blk._empirical, Neural):
                print('+++ neural network definition:', definition)
            plt.title('$' + str(definitionCopy) + '\ \ L_2(tr/te):\ ' +
                      str(round(bestTrn['L2'], 5)) + r', ' +
                      str(round(bestTst['L2'], 4)) +
                      '$')
            plt.xlim(x.min() - 0.25, x.max() + 0.25)
            plt.ylim(-2, 2)
            plt.grid()
            plt.scatter(X, Y, marker='>', c='g', label='training data')
            plt.plot(x, y, linestyle='-', label='prediction')
            plt.plot(x, yRef, linestyle=':', label='analytical')
            plt.scatter([X[bestTrn['iAbs']]], [Y[bestTrn['iAbs']]], marker='o',
                        color='c', s=60, label='max err train')
            plt.scatter([x[bestTst['iAbs']]], [y[bestTst['iAbs']]], marker='o',
                        color='r', s=60, label='max err test')
            plt.legend(bbox_to_anchor=(1.15, 0), loc='lower left')
            if saveFigures:
                f = path + '/' + file
                for s in definitionCopy:
                    f += '_' + str(s)
                print('file:', f)
                plt.savefig(f + '.png')
            plt.show()

            print('+++ optimum: definition: ', definitions[iDefBest],
                  ' index: [', iDefBest, '], L2: ',
                  round(L2TstBest, 5), sep='')
            if (iDef % 10 == 0 or iDef == len(definitions) - 1):
                print('+++ collect:\n', collect)
                collect.to_csv(path + file + '.collect.csv')
                collect.to_pickle(path + file + '.collect.pkl')
                collect.plot(y='L2Test', use_index=True)

    ################

    def example4():
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

        Y = np.array(df.loc[:, ['u']])                   # extracts an 2D array
        for f in definitions:
            blk = Black()

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

            # print('    shapes X:', X.shape, 'U:', U.shape, 'u:', u.shape,
            #      'du:', du.shape)

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

                try:
                    from plotArrays import plotIsoMap
                except ImportError:
                    print("??? import of 'plotArrays' failed")
                plotIsoMap(X[:, 0], X[:, 1], Y[:, 0],
                           labels=['$x$', '$y$', r'$u$' + s])
                plotIsoMap(X[:, 0], X[:, 1], Y[:, 0],
                           labels=['$x$', '$y$', r'$\tilde u$' + s])
                plotIsoMap(X[:, 0], X[:, 1], dy[:, 0],
                           labels=['$x$', '$y$', r'$\tilde u - u$' + s])


# main ########################################################################

    example1()
    example2()
    example3()  # loops over network structure
    example4()

    print('*** main.')
