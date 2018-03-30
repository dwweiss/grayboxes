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
      2018-02-07 DWW
"""

import numpy as np
from scipy.optimize import curve_fit

from Model import Model


class Theoretical(Model):
    """
    1) Computes y = f(x) through predict(x, **kwargs)
    2) Model training is defined in train(X, Y, **kwargs)
    3) ready() returns True if model is trained or not subject to training

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

        # assign white box model f(x) as method (with 'self') or function.
        # alternatively, method can be passed as argument for train or predict
        foo = Theoretical(f=function) or:
        foo = Theoretical(f=method)

        # (X, Y): training data
        X = [(1,2), (2,3), (4,5), (6,7), (7,8)]
        Y = [(1), (2), (3), (4), (5)]                 # or: Y = [1, 2, 3, 4, 5]

        # x: test data
        x = [(1, 4), (6, 6)]

        # without training, the result of the white box model f(x) is returned
        y = foo(x=x)                             # predict with white box model

        # train light gray box model with training data (X, Y)
        y = foo(X=X, Y=Y, x=x)    # train and predict with light gray box model
        y = foo(X=X, Y=Y)                          # train light gray box model

        # after model is trained, it keeps its weights for further preddictions
        y = foo(x=x)                        # predict with light gray box model

        # alternative passing of method as argument
        foo = Theoretical()
        y = foo(X=X, Y=Y, x=x, f=method)
    """

    def __init__(self, identifier='Theoretical', f=None):
        """
        Args:
            identifier (string):
                class identifier

            f (method):
                white box model f(x)
        """
        super().__init__(identifier=identifier, f=f)

        self._weights = None
        self._nMaxOut = 1        # max nOut is '1' due to implementation limits
        self._nMaxWeights = 8           # number of arguments of f() except 'x'

    def train(self, X=None, Y=None, **kwargs):
        """
        Fits coefficients of light gray box model and stores weights

        Args:
            X (2D array_like of float, optional):
                input X[i=0..nPoint-1, j=0..nInp-1]

            Y (2D array_like of float, optional):
                target Y[i=0..nPoint-1, k=0..nOut-1]

            kwargs (dict, optional):
                keyword arguments:

                fitMethod (string: 'lm', 'trf', or 'dogbox'):
                    optimizer method of scipy.optimizer.curve_fit()

                bounds (2-tuple of float or of array_like of float):
                    pairs (xMin, xMax) limiting x

                silent (bool):
                    If True, L2-norm printing is suppressed

        Returns:
            (three float):
                (L2-norm, max(abs(Delta_Y)), index(max(abs(Delta_Y)))

        Note:
            L2-norm = np.sqrt(np.mean(np.square(y - Y)))
        """
        self.X = X if X is not None and Y is not None else self.X
        self.Y = Y if X is not None and Y is not None else self.Y

        # fWrapper(x) is a wrapper around self.predict()
        def fWrapper(xT, c0=1, c1=1, c2=1, c3=1, c4=1, c5=1, c6=1, c7=1):
            kw = {k: v for k, v in locals().items() if k not in ('self', 'xT')}
            assert len(kw) == self._nMaxWeights, \
                str(len(kw)) + ' ' + str(self._nMaxWeights)

            # xT.shape: (nInp, nPoint), xT.T.shape: (nPoint, nInp)
            return self.predict(x=xT.T, **kw).ravel()

        self.write('+++ curve_fit: ', None)
        self._weights = None

        # X.T.shape: (nInp, nPoint)  Y.ravel().shape: (nPoint,)

        # Option 'method' can be a string out of ('lm', 'trf', 'dogbox')
        #     default is 'lm': Levenberg-Marquardt through leastsq() for
        #     unconstrained problems, use 'trf' for constrained ones
        # 'lm' requires nPoint >= nInp
        c, cov = curve_fit(f=fWrapper, xdata=self.X.T, ydata=self.Y.ravel(),
                           p0=None, sigma=None, absolute_sigma=False,
                           # TODO activate bounds
                           # bounds=(-np.inf, np.inf),
                           method=kwargs.get('fitMethod', None))

        # TODO check if curve_fit() failed
        self._ready = True

        if self._ready:
            self._weights = np.array(c)

            assert self._nMaxWeights == self._weights.size, \
                str(self._nMaxWeights) + ' ' + str(self._nWeights)
            self.write([float(str(round(_c, 3))) for _c in c])

        err = self.error(X=X, Y=Y, silent=kwargs.get('silent', None))
        self._L2norm = err[0]
        return err

    def predict(self, x=None, **kwargs):
        """
        Executes light gray box model

        Args:
            x (1D or 2D array_like of float):
                arguments, x[j=0..nInp-1]

            kwargs (dict, optional):
                keyword arguments

                c0, c1, ... (float):
                    weights

        Returns:
            self.y (2D array of float):
                model output, y[i=0, k=0..nOut-1]
        """
        if self._weights is None:
            validKeys = ['c'+str(i) for i in range(self._nMaxWeights)]
            kw = {k: v for k, v in kwargs.items() if k in validKeys}
        else:
            kw = {'c'+str(i): w for i, w in enumerate(self._weights)}

        return Model.predict(self, x, **kw)


# Examples ####################################################################

if __name__ == '__main__':
    ALL = 1

    from plotArrays import plot_X_Y_Yref

    # defines new train() method
    def train(self, X=None, Y=None, **kwargs):
        print('new train() silent:', self.silent)

    def user(self, x, c0=1, c1=1, c2=1, c3=1, c4=1, c5=1, c6=1, c7=1):
        x0, x1 = x[0], x[1]
        y0 = c0 + (c1 * np.sin(c2 * x0) + c3 * (x1 - 1.5)**2 + c4)
        return [y0]

    if 0 or ALL:
        s = 'Creates exact Y(X), add noise, fit model, compare: Y_fit vs Y_exa'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        noise = 0.25
        X0 = np.linspace(-1, 5, 8)
        X1 = np.linspace(0, 3, X0.size)
        X0, X1 = np.meshgrid(X0, X1)
        X = np.asfarray([X0.ravel(), X1.ravel()]).T
        Y_exa = np.array([np.array(user('dummy', x)) for x in X])
        Y_noise = Y_exa + np.random.normal(-noise, noise, size=Y_exa.shape)
        plot_X_Y_Yref(X, Y_noise, Y_exa, ['X', 'Y_{nos}', 'Y_{exa}'])

        # train with (X, Y_noise) and predict for x=X, variant with xKeys/yKeys
        foo = Theoretical(f=user)
        Y_fit = foo(XY=(X, Y_noise, ['xx0', 'xx1'], ['yy']), x=X)

        plot_X_Y_Yref(X, Y_noise, Y_exa, ['X', 'Y_{fit}', 'Y_{exa}'])
        df = foo.xyToFrame()
        print('Theoretical df:\n', df)

    if 1 or ALL:
        s = 'Creates exact Y(X), add noise, fit model, compare: Y_fit vs Y_exa'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        noise = 0.25
        X0 = np.linspace(-1, 5, 8)
        X1 = np.linspace(0, 3, X0.size)
        X0, X1 = np.meshgrid(X0, X1)
        X = np.asfarray([X0.ravel(), X1.ravel()]).T

        Y_exa = np.array([np.array(user('dummy', x)) for x in X])
        Y_noise = Y_exa + np.random.normal(-noise, noise, size=Y_exa.shape)
        plot_X_Y_Yref(X, Y_noise, Y_exa, ['X', 'Y_{nos}', 'Y_{exa}'])

        # train with (X, Y_noise) and predict for x=X, variant with xKeys/yKeys
        L2norm, bestMethod = np.inf, 'lm'
        for fitMethod in ('lm', 'trf', 'dogbox'):
            print('+++ fitMethod:', fitMethod)

            foo = Theoretical(f=user)
            y = foo(XY=(X, Y_noise, ['xx0', 'xx1'], ['yy']), x=X.copy(),
                    fitMethod=fitMethod)

            print('+++ L2-norm(' + str(fitMethod) + '):', foo.L2norm(), '\n\n')
            if L2norm > foo.L2norm():
                L2norm = foo.L2norm()
                Y_fit = y
                bestMethod = fitMethod

        print('bestMethod:', bestMethod, 'L2norm:', L2norm)
        plot_X_Y_Yref(X, Y_fit, Y_exa, ['X', 'Y_{fit}', 'Y_{exa}'])
        df = foo.xyToFrame()
        print('Theoretical df:\n', df)

    if 0 or ALL:
        s = 'Assigns new method to instance with access to self.* members'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        from xyz import xyz
        from Loop import Loop
        from Base import Base
        from Move import Move

        def f3(self, x, c0=1, c1=1, c2=1, c3=1, c4=1, c5=1, c6=1, c7=1):
            def moving_task(self):
                Base.task(self)
                standing = self.getFollower('standing')
                self.write('    moving_task, standing.id: ',
                           standing.identifier)

            def standing_task(self):
                Base.task(self)
                moving = self.getFollower('moving')
                self.write('    standing_task, moving.id: ', moving.identifier)

            if x is not None:
                self._x = x
            x0 = x[0]
            x1 = x[1]

            # create instance of Base and modify it
            standing = Base('standing')
            standing.task = standing_task.__get__(standing, Base)

            # create instance of Move and modify it
            moving = Move('moving')
            moving.task = moving_task.__get__(moving, Move)
            way = [xyz(0.0,  0.0,  0.0),   #    ^
                   xyz(0.2,  0.1,  0.0),   #   1|      /\
                   xyz(0.2,  0.2,  0.0),   #    |    /    \
                   xyz(0.3,  0.19, 0.0),   #    |  /        \             0.8
                   xyz(0.4,  0.0,  0.0),   #  0 |/----0.2-----\----0.6-----/-->
                   xyz(0.5, -0.1,  0.0),   #    |            0.4\        /    x
                   xyz(0.6, -0.23, 0.0),   #    |                 \    /
                   xyz(0.7, -0.17, 0.0),   #  -1|                   \/
                   xyz(0.81, 0.0,  0.0)]   #    | trajectory W=W(t)
            moving.setTrajectory(way, speed=1.5)

            # create leader, add the followers 'standing' and 'moving' and call
            mod = Loop()
            mod.setFollower([standing, moving])
            mod.setTransient(tEnd=0.7, n=5)
            mod()

            mod.getFollower('moving').plot()
            print('new predict x:', str(self.x))

            y0 = x0 * 2
            y1 = x1**2
            self.y = [y0, y1]

            # return value of f3()
            return self.y

        foo = Theoretical(f=f3)
        y = foo(x=[1, 2])
        print('y:', y)
