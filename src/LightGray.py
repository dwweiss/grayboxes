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
      2018-05-21 DWW
"""

import numpy as np
from scipy.optimize import curve_fit

from Model import Model


class LightGray(Model):
    """
    Light gray box model y=f(x, C)

    Extends the functionality of class Model by a train() method which fits
    the theoretical submodel f(x) with constant fit parameters 'C'

    Notes:
        - CInit (int, 1D or 2D array_like of float) is mandatory
        - The curve fit is currently limited to nOut=1, see self._nMaxOut

    Examples:
        def function(x, c0, c1, c1, c3):
            return [c0 + c1 * (c2 * np.sin(x[0]) + c3 * (x[1] - 1)**2)]

        def function2(x, *args):
            c0, c1, c2, c3 = args if len(args) > 0 else np.ones(4)
            return [c0 + c1 * (c2 * np.sin(x[0]) + c3 * (x[1] - 1)**2)]

        def method(self, x, *args):
            c0, c1, c2, c3 = args if len(args) > 0 else np.ones(4)
            return [c0 + c1 * (c2 * np.sin(x[0]) + c3 * (x[1] - 1)**2)]

        ### compact form:
        y = LightGray(function)(X=X, Y=Y, x=x, CInit=4, trainers='lm')

        ### expanded form:
        # assign theoretical submodel f(x) as function or method (with 'self')
        model = LightGray(function)  or
        model = LightGray(method)

        # (X, Y): training data
        X = [(1,2), (2,3), (4,5), (6,7), (7,8)]
        Y = [(1), (2), (3), (4), (5)]          # alternatively: [1, 2, 3, 4, 5]

        # x: test data
        x = [(1, 4), (6, 6)]

        # before training, result of theoretical submodel f(x) is returned
        y = model(x=x)                           # predict with white box model

        # train light gray box model with data (X, Y)
        model(X=X, Y=Y, C0=4)                                           # train

        # after model is trained, it keeps its weights for further preddictions
        y = model(x=x)                      # predict with light gray box model

        # combined traing and prediction
        y = model(X=X, Y=Y, C0=4, x=x)                      # train and predict
    """

    def __init__(self, f, identifier='LightGray'):
        """
        Args:
            f (method or function):
                theoretical submodel f(self, x) or f(x) for single data point

            identifier (string, optional):
                object identifier
        """
        super().__init__(identifier=identifier, f=f)

        self._nMaxOut = 1        # max nOut is '1' due to implementation limits
        self._nMaxWeights = 8           # number of arguments of f() except 'x'

    def train(self, X, Y, **kwargs):
        """
        Trains model, stores X and Y as self.X and self.Y, and stores result
        of best training trial as self.best.
        Fitted coefficients are stored as self._weights

        Args:
            X (2D or 1D array_like of float):
                training input, shape: (nPoint, nInp) or shape: (nPoint)

            Y (2D or 1D array_like of float):
                training target, shape: (nPoint, nOut) or shape: (nPoint)

            C0 (2D or 1D array_like of float, MANDATORY):
                sequence of initial x (1D array), CInit.shape[1] is number
                of fit parameters
                [IS PASSED IN KWARGS to be compatible to parallel.py]

            kwargs (dict, optional):
                keyword arguments:

                bounds (2-tuple of float or 2-tuple of 1D array_like of float):
                    list of pairs (xMin, xMax) limiting x

                trainers (string or 1D array_like of string):
                    optimizer method of scipy.optimizer.curve_fit()
                    valid trainers: ('lm', 'trf', 'dogbox')
                    default: 'trf'

        Returns:
            see Model.train()
        """
        self.X = X if X is not None and Y is not None else self.X
        self.Y = Y if X is not None and Y is not None else self.Y

        CInit = kwargs.get('C0', None)
        if CInit is None:
            CInit = kwargs.get('CInit', None)
        if CInit is None:
            CInit = [[]]
        if isinstance(CInit, int):
            CInit = np.ones(CInit)
        CInit = np.atleast_2d(CInit)
        nInp = CInit.shape[1]

        scipyOptimizeCurveFitTrainers = ('trf', 'lm', 'dogbox')
        validTrainers = scipyOptimizeCurveFitTrainers
        trainers = kwargs.get('trainers', None)
        if trainers is None:
            trainers = kwargs.get('trainer', None)
        if trainers is None:
            trainers = validTrainers[0]
        trainers = np.atleast_1d(trainers)
        if trainers[0].lower() == 'all':
            trainers = validTrainers

        kw = self.kwargsDel(kwargs, 'x')
        # xT.shape: (nInp, nPoint), xT.T.shape: (nPoint, nInp)
        if nInp == 1:
            def fScipyOptimizeCurveFit(xT, c0):
                return Model.predict(self, xT.T, c0, **kw).ravel()
        elif nInp == 2:
            def fScipyOptimizeCurveFit(xT, c0, c1):
                return Model.predict(self, xT.T, c0, c1, **kw).ravel()
        elif nInp == 3:
            def fScipyOptimizeCurveFit(xT, c0, c1, c2):
                return Model.predict(self, xT.T, c0, c1, c2, **kw).ravel()
        elif nInp == 4:
            def fScipyOptimizeCurveFit(xT, c0, c1, c2, c3):
                return Model.predict(self, xT.T, c0, c1, c2, c3, **kw).ravel()
        elif nInp == 5:
            def fScipyOptimizeCurveFit(xT, c0, c1, c2, c3, c4):
                return Model.predict(self, xT.T, c0, c1, c2, c3, c4,
                                     **kw).ravel()
        elif nInp == 6:
            def fScipyOptimizeCurveFit(xT, c0, c1, c2, c3, c4, c5):
                return Model.predict(self, xT.T, c0, c1, c2, c3, c4, c5,
                                     **kw).ravel()
        elif nInp == 7:
            def fScipyOptimizeCurveFit(xT, c0, c1, c2, c3, c4, c5, c6):
                return Model.predict(self, xT.T, c0, c1, c2, c3, c4, c5, c6,
                                     **kw).ravel()
        elif nInp == 8:
            def fScipyOptimizeCurveFit(xT, c0, c1, c2, c3, c4, c5, c6, c7):
                return Model.predict(self, xT.T, c0, c1, c2, c3, c4, c5, c6,
                                     c7, **kw).ravel()
        else:
            assert 0, str(nInp) + ' ' + str(CInit)

        if any([tr not in validTrainers for tr in trainers]):
            trainers = validTrainers[0]
            self.write('??? unknown trainer found, correct to:', trainers)
        # TODO activate bounds = kwargs.get('bounds', (-np.inf, np.inf))

        self.best = self.initBest()
        self._weights = None
        self.write('    fit (', None)
        for trainer in trainers:
            self.write(trainer, ', ' if trainer != trainers[-1] else '', None)

            for iTrial, cInit in enumerate(CInit):
                #self.write('+++ cInit: ', np.round(cInit, 3))

                if trainer in scipyOptimizeCurveFitTrainers:
                    try:
                        self.ready = True         # required by Model.predict()
                        c, cov = curve_fit(f=fScipyOptimizeCurveFit,
                                           xdata=self.X.T,     # (nInp, nPoint)
                                           ydata=self.Y.ravel(),     # (nPoint)
                                           p0=cInit, sigma=None,
                                           absolute_sigma=False,
                                           # TODO activate bounds=(-inf,+inf),
                                           method=trainer)
                        # TODO check for failure of curve_fit()
                        self.ready = True
                        if self.ready:
                            self._weights = np.array(c)
                    except RuntimeError:
                        self.ready = False
                        print('\n??? max epochs exceeded,cont.with next trial')
                        continue
                else:
                    assert 0, 'branch for curve fit optimizer'

                if self.ready:
                    #print('self._weights:', self._weights)
                    actual = self.error(X=X, Y=Y, silent=True)
                    if self.best['L2'] > actual['L2']:
                        self.best = actual
                        self.best['trainer'] = trainer
                        self.best['epochs'] = -1
                        self.best['iTrial'] = iTrial

        assert self._weights is not None
        assert self._nMaxWeights >= self._weights.size, str(self._weights.size)
        assert CInit.size >= self._weights.size, str(self._weights.size)
        self.write('), w: ', None)
        self.write(np.round(self._weights, 4))
        self.write('    best trainer: ', "'", self.best['trainer'], "'",
                   ', L2: ', float(str(round(self.best['L2'], 4))),
                   ', abs: ', float(str(round(self.best['abs'], 4))),
                   ', iTrial: ', self.best['iTrial'])

        return self.best

    def predict(self, x, *args, **kwargs):
        """
        Executes Model, stores input x as self.x and output as self.y

        Args:
            x (2D or 1D array_like of float):
                prediction input, shape: (nPoint, nInp) or shape: (nInp)

            args(list arguments, optional):
                constant fit parameters if self._weights is None

            kwargs (dict, optional):
                keyword arguments

        Returns:
            (2D array of float):
                prediction output, shape: (nPoint, nOut)
        """
        args = self._weights if self._weights is not None else args
        return Model.predict(self, x, *args, **self.kwargsDel(kwargs, 'x'))


# Examples ####################################################################

if __name__ == '__main__':
    ALL = 0

    from plotArrays import plot_X_Y_Yref
    import Model as md
    from White import White

    def fUser(self, x, *args, **kwargs):
        c0, c1, c2, c3 = args if len(args) > 0 else np.ones(4)
        x0, x1 = x[0], x[1]

        y0 = c0 + c1 * np.sin(c2 * x0) + c3 * (x1 - 1.5)**2

        return [y0]

    if 0 or ALL:
        s = 'Light gray: add noise to Y_exa, fit, compare: y with Y_exa'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        noise = 10e-2
        X = md.grid(8, [-1, 2], [0, 2])
        y_exa = White(fUser)(x=X, silent=True)
        Y = md.noise(y_exa, relative=noise)
        plot_X_Y_Yref(X, Y, y_exa, ['X', 'Y_{nse}', 'y_{exa}'])

        # train with (X, Y_noise) and predict for x=X, variant with xKeys/yKeys
        model = LightGray(fUser)
        model.silent = True
        best = model(X=X, Y=Y, C0=4)
        y = model(x=X)
        plot_X_Y_Yref(X, y, y_exa, ['X', 'y', 'y_{exa}'])
        print('+++ best:', best)
        # df = model.xy2frame()
        # print('df:\n', df)

    if 1 or ALL:
        s = 'Creates exact Y(X), add noise, fit model, compare: y_fit vs Y_exa'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        noise = 0.25
        X = md.grid(8, [-1, 8], [0, 3])
        y_exa = White(fUser)(x=X)
        Y = md.noise(y_exa, absolute=noise)
        plot_X_Y_Yref(X, Y, y_exa, ['X', 'Y_{nse}', 'y_{exa}'])

        # train with (X, Y_noise) and predict for x=X, variant with xKeys/yKeys
        C0 = md.rand(9, [-100, 100], [-20, 20], [-10, 10], [0, 1000])
        y = LightGray(fUser)(X=X, Y=Y, C0=C0, x=X, silent=True)

        plot_X_Y_Yref(X, y, y_exa, ['X', 'y', 'y_{exa}'])

        # df = lgr.xy2frame()
        # print('df containing y=f(x):\n', df)
