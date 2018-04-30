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
      2018-04-30 DWW
"""

import numpy as np
from scipy.optimize import curve_fit

from Model import Model


class LightGray(Model):
    """
    Extends the functionality of class Model by a train() method which fits
    the theoretical model f(x) with constant coefficients

    1) Computes y = f(x) through predict(x, **kwargs)
    2) Model training is defined in train(X, Y, **kwargs)
    3) self.ready returns True if model is trained

    Upper case 'X' and 'Y' are 2D training input and target
    Lower case 'x' and 'y' are 2D input and prediction

    Notes:
        Array definition: Input array X[i,j] and output array Y[i,k] are 2D.
            First index is data point index, second index is input/output index
            If X or Y passed as 1D array_like, they are transformed to:
            X = np.atleast_2d(X).T and/or Y = np.atleast_2d(Y).T

        The curve fit is currently limited to nOut=1, see self._nMaxOut

    Examples:
        def function(x, c0=1, c1=1, c2=1, c3=1, c4=1, c5=1, c6=1, c7=1):
            return c0 + c1 * np.array(c2 * np.sin(x[0]) + c3 * (x[1] - 1)**2)

        # method with access to 'self' attributes
        def method(self, x, c0=1, c1=1, c2=1, c3=1, c4=1, c5=1, c6=1, c7=1):
            return function(x, c0, c1, c2, c3, c4, c5, c6, c7)

        # assign theoretical model f(x) as method (with 'self') or function.
        # alternatively, method can be passed as argument for train or predict
        foo = LightGray(f=f)

        # (X, Y): training data
        X = [(1,2), (2,3), (4,5), (6,7), (7,8)]
        Y = [(1), (2), (3), (4), (5)]                 # or: Y = [1, 2, 3, 4, 5]

        # x: test data
        x = [(1, 4), (6, 6)]

        # without training, result of theoretical model f(x) is returned
        y = foo(x)

        # train light gray box model with training data (X, Y)
        y = foo(X=X, Y=Y, x=x)    # train and predict with light gray box model
        y = foo(X=X, Y=Y)                          # train light gray box model

        # after model is trained, it keeps its weights for further preddictions
        y = foo(x=x)                        # predict with light gray box model

        # compact form
        y = LightGray(f=f)(X=X, Y=Y, x=x, fitMethod='lm')
    """

    def __init__(self, f, identifier='LightGray'):
        """
        Args:
            f (method or function):
                theoretical model y = f(x) for single data point

            identifier (string, optional):
                object identifier
        """
        super().__init__(identifier=identifier, f=f)

        self._weights = None
        self._nMaxOut = 1        # max nOut is '1' due to implementation limits
        self._nMaxWeights = 8           # number of arguments of f() except 'x'

    def train(self, X=None, Y=None, **kwargs):
        """
        Fits coefficients of light gray box model and save weights as
        self._weights

        Args:
            X (2D array_like of float, optional):
                input X[i=0..nPoint-1, j=0..nInp-1]

            Y (2D array_like of float, optional):
                target Y[i=0..nPoint-1, k=0..nOut-1]

            kwargs (dict, optional):
                keyword arguments:

                fitMethod (string of of:('lm', 'trf', or 'dogbox')):
                    optimizer method of scipy.optimizer.curve_fit()

                bounds (2-tuple of float or 2-tuple of 1D array_like of float):
                    pairs (xMin, xMax) limiting x

                silent (bool):
                    If True, L2-norm printing is suppressed

        Returns:
            (3-tuple of float):
                (L2-norm, max(abs(Delta_Y)), index(max(abs(Delta_Y)))

        Note:
            L2-norm = np.sqrt(np.mean(np.square(y - Y)))
        """
        self.X = X if X is not None and Y is not None else self.X
        self.Y = Y if X is not None and Y is not None else self.Y

        f = kwargs.get('f', None)
        if f is not None:
            self.f = f
        assert self.f is not None

        fitMethod = kwargs.get('trainer', 'lm')
        # bounds = kwargs.get('bounds', (-np.inf, np.inf))

        # fWrapper(x) is a wrapper around self.predict(), xT=x.T
        def fWrapper(xT, c0=1, c1=1, c2=1, c3=1, c4=1, c5=1, c6=1, c7=1):
            kw = {k: v for k, v in locals().items() if k not in ('self', 'xT')}
            assert len(kw) == self._nMaxWeights, \
                str(len(kw)) + ' ' + str(self._nMaxWeights)

            # xT.shape: (nInp, nPoint), xT.T.shape: (nPoint, nInp)
            return self.predict(x=xT.T, **kw).ravel()

        self.write('+++ curve_fit: ', None)
        self._weights = None

        # Option 'method' can be a string out of ('lm', 'trf', 'dogbox')
        #     default is 'lm': Levenberg-Marquardt through leastsq() for
        #     unconstrained problems, use 'trf' for constrained ones
        #
        #     'lm' requires nPoint >= nInp

        # X.T.shape: (nInp, nPoint)  Y.ravel().shape: (nPoint,)
        c, cov = curve_fit(f=fWrapper, xdata=self.X.T, ydata=self.Y.ravel(),
                           p0=None, sigma=None, absolute_sigma=False,
                           # TODO activate bounds: bounds=(-np.inf, np.inf),
                           method=fitMethod)

        # TODO check if curve_fit() failed
        self.ready = True

        if self.ready:
            self._weights = np.array(c)

            assert self._nMaxWeights == self._weights.size, \
                str(self._nMaxWeights) + ' ' + str(self._nWeights)
            self.write([float(str(round(_c, 3))) for _c in c])

        error = self.error(X=X, Y=Y, silent=kwargs.get('silent', None))
        self._L2norm = error[0]
        return error

    def predict(self, x=None, **kwargs):
        """
        Executes light gray box model

        Args:
            x (1D or 2D array_like of float, optional):
                arguments, x.shape: (nPoint, nInp) or (nInp,)

            kwargs (dict, optional):
                keyword arguments

                c0, c1, ... (float):
                    weights

        Returns:
            self.y (2D array of float):
                model output, y.shape: (nPoint, nOut)
        """
        if self._weights is None:
            validKeys = ['c'+str(i) for i in range(self._nMaxWeights)]
            kw = {k: v for k, v in kwargs.items() if k in validKeys}
        else:
            kw = {'c'+str(i): w for i, w in enumerate(self._weights)}

        return super().predict(x, **kw)


# Examples ####################################################################

if __name__ == '__main__':
    ALL = 1

    from plotArrays import plot_X_Y_Yref
    from Model import gridInit
    from White import White

    def fUser(self, x, c0=1, c1=1, c2=1, c3=1, c4=1, c5=1, c6=1, c7=1):
        x0, x1 = x[0], x[1]
        y0 = c0 + c1 * np.sin(c2 * x0) + c3 * (x1 - 1.5)**2
        return [y0]

    if 0 or ALL:
        s = 'Light gray: add noise to Y_exa, fit, compare: y_fit with Y_exa'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        noise = 0.25
        X = gridInit(ranges=((-1, 8), (0, 3)), n=(8, 8))
        Y_exa = White(fUser)(x=X)
        Y_nse = Y_exa + np.random.normal(-noise, noise, size=Y_exa.shape)
        plot_X_Y_Yref(X, Y_nse, Y_exa, ['X', 'Y_{nse}', 'Y_{exa}'])

        # train with (X, Y_noise) and predict for x=X, variant with xKeys/yKeys
        lgr = LightGray(fUser)
        y_fit = lgr(XY=(X, Y_nse, ['xx0', 'xx1'], ['yy']), x=X)

        plot_X_Y_Yref(X, y_fit, Y_exa, ['X', 'y_{fit}', 'Y_{exa}'])
        df = lgr.xy2frame()
        print('df:\n', df)

    if 1 or ALL:
        s = 'Creates exact Y(X), add noise, fit model, compare: y_fit vs Y_exa'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        noise = 0.25
        X = gridInit(ranges=((-1, 8), (0, 3)), n=(8, 8))
        Y_exa = White(fUser)(x=X)
        Y_nse = Y_exa + np.random.normal(-noise, noise, size=Y_exa.shape)
        plot_X_Y_Yref(X, Y_nse, Y_exa, ['X', 'Y_{nse}', 'Y_{exa}'])

        # train with (X, Y_noise) and predict for x=X, variant with xKeys/yKeys
        L2norm, bestMethod = np.inf, 'lm'
        lgr = LightGray(fUser)
        for fitMethod in ('lm', 'trf', 'dogbox'):
            print('+++ fitMethod:', fitMethod)

            y = lgr(XY=(X, Y_nse, ['xx0', 'xx1'], ['yy']), x=X,
                    fitMethod=fitMethod)

            print('x:', lgr.x.shape)
            print('y:', lgr.y.shape)
            print('X:', lgr.X.shape)
            print('Y:', lgr.Y.shape)
            print('+++ L2-norm(' + str(fitMethod) + '):', lgr._bestTrain[0],
                  '\n\n')
            if L2norm > lgr._bestTrain[0]:
                L2norm = lgr._bestTrain[0]
                Y_fit = y
                fitMethod = fitMethod

        print("bestMethod: '" + bestMethod + "', L2norm:", L2norm)
        plot_X_Y_Yref(X, Y_fit, Y_exa, ['X', 'Y_{fit}', 'Y_{exa}'])

        df = lgr.xy2frame()
        print('df containing y=f(x):\n', df)
