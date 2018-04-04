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
      2018-02-04 DWW
"""

import numpy as np
import pandas as pd

from Model import Model
try:
    from Neural import Neural
    _hasNeural = True
except IOError:
    print('!!! Neural not imported')
    _hasNeural = False


class Empirical(Model):
    """
    Wraps empirical models based on:
        - neural network training
        - other (not implemented yet)

    Note:
        Array definition: input and output arrays are 2D, first index is sample
            index. Use 'X[:,0]', or 'X.T[0]' for extracting 1D

        The whole neural network is saved as 'self._net'

    Example:
        X0 = np.linspace(0, 1, 20)
        X1 = X0 * 2
        Y = X**2

        # neural network
        foo = Empirical(X, Y, definition=[])
        y = foo.predict(X)
    """

    def __init__(self, identifier='Empirical', trainer=None, f=None):
        """
        Args:
            identifier (string):
                class identifier

            trainer (method):
                training method

            f (method):
                white box model f(x)
        """
        super().__init__(identifier=identifier, trainer=trainer, f=f)

        self._net = None

    def isNeural(self):
        """
        Checks if self.definition equals 'neural' or None, or it it is a scalar

        Returns:
            True if a neural network is defined
        """
        return self.definition == 'neural' or self.definition is None or \
            isinstance(self.definition, (int, list, tuple))

    def train(self, X=None, Y=None, **kwargs):
        """
        Trains model

        Args:
            X (1D or 2D array_like of float, optional):
                training input X[0..nPoint-1, j=0..nInp-1]

            Y (1D or 2D array_like of float, optional):
                training target Y[i=0..nPoint-1, k=0..nOut-1]

            kwargs (dict, optional):
                keyword arguments:

                definition (string or function or list of int or None):
                    1) build-in curve selection as string, or
                    2) fit curve passed as function, or
                    3) neural network definition as 1D array, string or None
                ...
                ...
                training options of neural network, see Neural.train()

        Returns:
            (three float):
                (L2-norm, max(abs(Delta_Y)), index(max(abs(Delta_Y)))

        Note:
            - L2_norm = np.sqrt(np.mean(np.square(y - Y)))
            - If training fails, self._weights is set to 'None'
            - Hidden layers of neural network passed as argument 'definition'
                are of type 'list', 'tuple' or 'np.array',
                e.g. [5, 2] stands for two hidden layers with 5 and 2 neurons
        """
        self.definition = kwargs.get('definition', None)
        if self.definition == 'neural':
            self.definition = None

        # if 1D arrays are passed, it is assumed that shape is: (1, nInp/nOut)
        if np.asanyarray(X).ndim == 1:
            self._X = np.atleast_2d(X).T
        else:
            self._X = np.atleast_2d(X)
        if np.asanyarray(Y).ndim == 1:
            self._Y = np.atleast_2d(Y).T
        else:
            self._Y = np.atleast_2d(Y)
        assert self.X is not None and self.Y is not None
        assert X.shape[0] > 2, 'X.shape: ' + str(X.shape)

        if self.isNeural():
            self.write('+++ neural network')
            if self._net is not None:
                del self._net
            self._net = Neural()
            self._net.importArrays(self.X, self.Y)
            self._net.train(hidden=self.definition, f=self.f,
                            **self.kwargsDel(kwargs, ('hidden', 'f')))
            ok = True
            if not ok:
                del self._net
                self._net = None
        else:
            self.write('no training for definition: ', str(self.definition))

        return self.error(self.X, self.Y)

    def predict(self, x=None, **kwargs):
        """
        predicts empirical model

        Args:
            x (1D or 2D array_like of float, optional):
                prediction input, x[i=0..nPoint-1, j=0..nInp-1]

            kwargs (dict, optional):
                keyword arguments

        Returns:
            (2D array of float):
                prediction result, y[i=0..nPoint-1, k=0..nOut-1]
        """
        self.x = x if x is not None else self.x

        if self.isNeural():
            assert self._net is not None
            self.y = self._net(self.x)
        else:
            self.y = np.asfarray([self.definition(_x) for _x in self.x])

        return self.y


# Examples ###################################################################

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print('*** main')

    ################

    def example1():
        X = np.linspace(0, 1, 20)
        noise = 0.1
        Y = X**2 + np.random.rand(X.size) * noise

        if 1:
            # neural network
            print('X.shape:', X.shape, 'Y.shape:', Y.shape)
            foo = Empirical()
            foo.train(X, Y, definition=[])
            y = foo.predict(X)[:, 0]

            plt.plot(X, Y, X, y)
            plt.show()

    ################

    def example2():
        # neural network, 1D problem sin(x) with noise
        def f(x, a=-0.2, b=0, c=0, d=0, e=0):
            return np.sin(x) + a * x + b

        # hidden layers
        definition = [10, 10]

        # training data
        nSamples = 20
        maxNoise = 0.01
        X = np.linspace(-1 * np.pi, +1 * np.pi, nSamples)
        Y = f(X)
        if maxNoise > 0.0:
            Y += np.random.normal(-maxNoise, +maxNoise, Y.size)
        X, Y = X.reshape(X.size, 1), Y.reshape(Y.size, 1)

        # test data
        dx = 0.5 * np.pi
        x = np.linspace(X.min()-dx, X.max()+dx, nSamples*2)

        foo = Empirical()
        (L2Train, abs, iAbsTrain) = \
            foo.train(X, Y, definition=definition, trials=5, goal=1e-6,
                      epochs=500, trainers='gdx bfgs rprop')
        y = foo.predict(x)

        print('Emp ex2 x:', x.shape, 'y:', y.shape)
        L2Test, abs, iAbsTest = foo.error(x, f(x))

        plt.title('$hidden:' + str(definition) +
                  ', L_{2}^{train}:' + str(round(L2Train, 4)) +
                  ', L_{2}^{test}:' + str(round(L2Test, 4)) + '$')
        plt.cla()
        plt.ylim(min(-2, Y.min(), y.min()), max(2, Y.max(), Y.max()))
        plt.yscale('linear')
        plt.xlim(-0.1 + x.min(), 0.1 + x.max())
        plt.scatter(X, Y, marker='x', c='r', label='training data')
        plt.plot(x, y, c='b', label='prediction')
        plt.plot(x, f(x), linestyle=':', label='analytical')
        plt.scatter([X[iAbsTrain]], [Y[iAbsTrain]], marker='o', color='r',
                    s=66, label='max abs train')
        plt.scatter([x[iAbsTest]], [y[iAbsTest]], marker='o', color='b',
                    s=66, label='max abs test')
        plt.legend(bbox_to_anchor=(1.1, 0), loc='lower left')
        plt.grid()
        plt.show()

    ################

    def example3():
        # 1D problem sin(x) with noise, variation of neural network structure
        saveFigures = True
        path = 'c:/temp/'
        file = 'sin_x_-3..3.5pi'
        nPoint = 20
        maxNeuronsInLayer = 2  # 16
        nHiddenLayers = 1  # 3
        MAX_HIDDEN_LAYERS = 6  # MUST NOT be modified
        assert MAX_HIDDEN_LAYERS >= nHiddenLayers
        maxNoise = 0.0

        # compute training target ('X', 'U') and test data ('x', 'uAna')
        def f(x, c0=1, c1=1, c2=1, c3=1, c4=1, c5=1, c6=1, c7=1):
            return np.sin(x) * c0 + c1
        X = np.linspace(-2*np.pi, +2*np.pi, nPoint)   # argument of train data
        if 0:
            X = np.flipud(X)                   # reverse order of elements of X
        x = np.linspace(-3*np.pi, 3*np.pi, 100)
        dx = (X.max() - X.min()) / nPoint
        x = x + 0.5 * dx          # shift argument for calculation of test data
        U = f(X)
        if maxNoise > 0.0:
            U += np.random.normal(-maxNoise, +maxNoise, X.size)
        X, U = X.reshape(X.size, 1), U.reshape(U.size, 1)
        x = x.reshape(x.size, 1)
        uAna = f(x)

        # define 'collect' as DataFrame for result collection
        columns = ['n' + str(i+1) for i in range(MAX_HIDDEN_LAYERS)]
        columns.extend(['L2Train', 'absTrain', 'iAbsTrain',
                        'L2Test', 'absTest', 'iAbsTest',
                        'mse', 'trainer', 'epochs'])
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

        L2TestBest, iDefBest = 1.0, 0
        for iDef, definition in enumerate(definitions):
            definitionCopy = definition.copy()
            print('+++ hidden layers:', definition)

            # network training
            em = Empirical()
            L2Train, absTrain, iAbsTrain = \
                em.train(X, U, definition=definition, trials=5, epochs=500,
                         show=500, algorithms='bfgs', goal=1e-5)

            # network prediction
            u = em.predict(x)

            L2Test, absTest, iAbsTest = em.error(x, uAna, silent=False)
            if L2TestBest > L2Test:
                L2TestBest = L2Test
                iDefBest = iDef
            row = definitionCopy.copy()
            row = row + [0]*(MAX_HIDDEN_LAYERS - len(row))
            row.extend([L2Train, absTrain, int(iAbsTrain),
                        L2Test,  absTest,  int(iAbsTest),
                        0, 0, 0  # mse trainer epochs
                        ])
            # print('row:', row, len(row), 'columns:', collect.keys)
            collect.loc[collect.shape[0]] = row

            if isinstance(em._net, Neural):
                print('+++ neural network definition:', em.definition)
            plt.title('$' + str(definitionCopy) + '\ \ L_2(tr/te):\ ' +
                      str(round(L2Train, 5)) + r', ' + str(round(L2Test, 4)) +
                      '$')
            plt.xlim(x.min() - 0.25, x.max() + 0.25)
            plt.ylim(-2, 2)
            plt.grid()
            plt.scatter(X, U, marker='>', c='g', label='training data')
            plt.plot(x, u, linestyle='-', label='prediction')
            plt.plot(x, uAna, linestyle=':', label='analytical')
            plt.scatter([X[iAbsTrain]], [U[iAbsTrain]], marker='o', color='c',
                        s=60, label='max err train')
            plt.scatter([x[iAbsTest]], [u[iAbsTest]], marker='o', color='r',
                        s=60, label='max err test')
            plt.legend(bbox_to_anchor=(1.15, 0), loc='lower left')
            if saveFigures:
                f = file
                for s in definitionCopy:
                    f += '_' + str(s)
                print('file:', f)
                plt.savefig(path + f + '.png')
            plt.show()

            print('+++ optimum: definition: ', definitions[iDefBest],
                  ' index: [', iDefBest, '], L2: ',
                  round(L2TestBest, 5), sep='')
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
        kwargs = {'trainers': 'bfgs rprop'}

        U = np.array(df.loc[:, ['u']])                   # extracts an 2D array
        for f in definitions:
            em = Empirical()

            if hasattr(f, '__call__'):
                print(f.__name__)
                print('f1==f', f1 == f, id(f) == id(f1))
                print('f2==f', f2 == f, id(f) == id(f2))
                print('f3==f', f3 == f, id(f) == id(f3))
            if hasattr(f, '__call__') and f2 != f:
                X = np.array(df.loc[:, ['x']])
            else:
                X = np.array(df.loc[:, ['x', 'y']])

            em.train(X, U, definition=f, **kwargs)
            u = em.predict(X)
            du = u - U

            #print('    shapes X:', X.shape, 'U:', U.shape, 'u:', u.shape,
            #      'du:', du.shape)

            # console output
            print('    ' + 76 * '-')
            su = '[j:0..' + str(U.shape[1] - 1) + '] '
            print('    i   X[j:0..' + str(X.shape[1] - 1) + ']' +
                  'U' + su + 'u' + su + 'du' + su + 'rel' + su + '[%]:')
            for i in range(X.shape[0]):
                print('{:5d} '.format(i), end='')
                for a in X[i]:
                    print('{:f} '.format(a), end='')
                for a in U[i]:
                    print('{:f} '.format(a), end='')
                for a in u[i]:
                    print('{:f} '.format(a), end='')
                for j in range(U.shape[1]):
                    print('{:f} '.format(du[i][j]), end='')
                for j in range(U.shape[1]):
                    print('{:f} '.format(du[i][j] / U[i][j] * 100), end='')
                print()
            print('    ' + 76 * '-')

            # graphic presentation
            if X.shape[1] == 1 and U.shape[1] == 1:
                plt.title('Approximation')
                plt.xlabel('$x$')
                plt.ylabel('$u$')
                plt.scatter(X, U, label='$u$', marker='x')
                plt.scatter(X, U, label=r'$\tilde u$', marker='o')
                if em.isNeural():
                    plt.plot(X, u, label=r'$\tilde u$ (cont)')
                plt.plot(X, du, label=r'$\tilde u - u$')
                plt.legend(bbox_to_anchor=(0, 0), loc='lower left')
                plt.show()
                if 1:
                    plt.title('Absolute error')
                    plt.ylabel(r'$\tilde u - u$')
                    plt.plot(X, du)
                    plt.show()
                if 1:
                    plt.title('Relative error')
                    plt.ylabel('E [%]')
                    plt.plot(X, du / U * 100)
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
                plotIsoMap(X[:, 0], X[:, 1], U[:, 0],
                           labels=['$x$', '$y$', r'$u$' + s])
                plotIsoMap(X[:, 0], X[:, 1], U[:, 0],
                           labels=['$x$', '$y$', r'$\tilde u$' + s])
                plotIsoMap(X[:, 0], X[:, 1], du[:, 0],
                           labels=['$x$', '$y$', r'$\tilde u - u$' + s])


# main ########################################################################

    example1()
    example2()
    example3()  # loops over network structure
    example4()

    print('*** main.')
