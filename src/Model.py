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
      2018-05-08 DWW
"""

import inspect
import random
from collections import OrderedDict
import numpy as np
from pandas import DataFrame
from Base import Base
try:
    import parallel
except ImportError:
    print("!!! Module 'parallel' not imported")


def grid(n, *ranges):
    """
    Sets intial (uniformly spaced) grid input, for instance for 2 input
    with 4 nodes per axis: grid(n=4, [1, 3], [-7, -5])

            x---x---x---x
            |   |   |   |
            x---x---x---x
            |   |   |   |
            x---x---x---x
            |   |   |   |
            x---x---x---x
    Args:
        n (int or 1D array_like of int):
            number of nodes per axis for which initial values are generated

        ranges (variable length argument list of pairs of float):
            list of (min, max) pairs

    Returns:
        (2D array of float):
            Grid-like initial values, first index is point index, second
            index is input index
    """
    ranges = list(ranges)
    N = list(np.atleast_1d(n))
    n = N + [N[-1]] * (len(ranges) - len(N))
    assert len(n) == len(ranges), 'n:' + str(n) + ' ranges:' + str(ranges)

    ranges = np.asfarray(ranges)
    xVar = []
    for rng, _n in zip(ranges, n):
        rngMin = min(rng[0], rng[1])
        rngMax = max(rng[0], rng[1])
        xVar.append(np.linspace(rngMin, rngMax, _n))

    if ranges.shape[0] == 1:
        x = xVar[0]
    elif ranges.shape[0] == 2:
        x0, x1 = np.meshgrid(xVar[0], xVar[1])
        x = [(a0, a1) for a0, a1 in zip(x0.ravel(), x1.ravel())]
    elif ranges.shape[0] == 3:
        x0, x1, x2 = np.meshgrid(xVar[0], xVar[1], xVar[2])
        x = [(a0, a1, a2) for a0, a1, a2 in
             zip(x0.ravel(), x1.ravel(), x2.ravel())]
    elif ranges.shape[0] == 4:
        x0, x1, x2, x3 = \
            np.meshgrid(xVar[0], xVar[1], xVar[2], xVar[3])
        x = [(a0, a1, a2, a3) for a0, a1, a2, a3 in
             zip(x0.ravel(), x1.ravel(), x2.ravel(),
                 x3.ravel())]
    elif ranges.shape[0] == 5:
        x0, x1, x2, x3, x4 = \
            np.meshgrid(xVar[0], xVar[1], xVar[2], xVar[3], xVar[4])
        x = [(a0, a1, a2, a3, a4) for a0, a1, a2, a3, a4 in
             zip(x0.ravel(), x1.ravel(), x2.ravel(),
                 x3.ravel(), x4.ravel())]
    elif ranges.shape[0] == 6:
        x0, x1, x2, x3, x4, x5 = \
            np.meshgrid(xVar[0], xVar[1], xVar[2], xVar[3], xVar[4],
                        xVar[5])
        x = [(a0, a1, a2, a3, a4, a5) for a0, a1, a2, a3, a4, a5 in
             zip(x0.ravel(), x1.ravel(), x2.ravel(),
                 x3.ravel(), x4.ravel(), x5.ravel())]
    else:
        assert 0, 'ranges: ' + str(ranges)
    return np.asfarray(x)


def cross(n, *ranges):
    """
    Sets intial (uniformly spaced) cross input, for instance for 2 input
    with 5 nodes per axis: cross(n=5, (1, 2), (-4, -3))
                  x
                  |
                  x
                  |
            x--x--x--x--x
                  |
                  x
                  |
                  x
    Args:
        ranges (variable length argument list of pairs of float):
            list of (min, max) pairs

        n (int):
            number of nodes per axis for which initial values are generated
            n is corrected to an odd number if n is even

    Returns:
        (2D array of float):
            Cross-like initial values, first index is point index, second
            index is input index
    """
    ranges = list(ranges)
    N = list(np.atleast_1d(n))

    # ensures odd number of nodes per axis
    N = [2 * (n // 2) + 1 for n in N]
    n = N + [N[-1]] * (len(ranges) - len(N))
    assert len(n) == len(ranges), 'n:' + str(n) + ' ranges:' + str(ranges)

    x = []
    xCenter = [np.mean(rng) for rng in ranges]
    x.append(xCenter)
    for i, rng in enumerate(ranges):
        if rng[0] != rng[1]:
            xPoint = xCenter.copy()
            rngMin = min(rng[0], rng[1])
            rngMax = max(rng[0], rng[1])
            xVar = np.linspace(rngMin, rngMax, n[i])
            for j in range(0, n[i]):
                if j != n[i] // 2:
                    xPoint[i] = xVar[j]
                    x.append(xPoint.copy())
    return np.asfarray(x)


def rand(n, *ranges):
    """
    Sets intial (uniformly distributed) random input, for instance for 2
    input with 12 trials: rand(n=12, [1, 3], [-7, -5])

           -------------
          |  x  x  x    |
          |    x x      |
          |   x     x   |
          |  x    x     |
          |    x  x  x  |
           -------------
    Args:
        n (int):
            number of trials for which initial values are random generated

        ranges (variable length argument list of pairs of float):
            list of (min, max) pairs


    Returns:
        (2D array of float):
            Random initial values, first index is trial index, second index
            is input index
    """
    ranges = list(ranges)
    ranges = np.atleast_2d(ranges)
    assert ranges.shape[1] == 2, 'ranges: ' + str(ranges)
    assert n > 0, 'n: ' + str(n)
    assert all(x[0] <= x[1] for x in ranges), 'ranges: ' + str(ranges)

    x = np.array([[random.uniform(min(rng[0], rng[1]), max(rng[0], rng[1]))
                  for rng in ranges] for i in range(n)])
    return x


def noise(x, relative=0.0, absolute=None, uniform=True):
    """
    Adds noise to multi-dimensional arrays

        |
        |                      **   *
        |                *    *===*==
        |      *   *  =**=*===*    *
        |     *=*=*==*     **
        |*==**   *
        | **                    === x
        |                       *** x + noise
        +------------------------------------

    Args:
        x (array_like of float):
            initial array

        relative (float, optional):
            maximum noise added, relative to difference between maximum and
            minimum of array x. only effective if 'absolute' is None

        absolute (float, optional):
            maximum of absolute noise added to x

        uniform (bool):
            if True then uniformely distributed random numbers. Otherwise
            noise in normally distributed

    Returns:
        (array of float):
            array of same shape as x with noise or copy of x if no noise

    Note:
        'absolute' superseeds 'relative' argument
        In case of negative 'relative' and 'absolute', x returns unchanged
    """

    x = np.asfarray(x)
    if absolute is None:
        dx = relative * (x.max() - x.min()) if relative is not None else 0.0
    else:
        dx = absolute
    if dx <= 0.:
        return x.copy()
    else:
        if not uniform:
            return x + np.random.normal(loc=0.0, scale=dx, size=x.shape)
        else:
            return x + np.random.uniform(low=-dx, high=dx, size=x.shape)


class Model(Base):
    """
    Parent class of White, LightGray, MediumGray, DarkGray and Black

    - Array definition: input and output arrays are 2D, first index is data
      point index. Extract 1D arrays with 'X[:, 0]', or 'X.T[0]'
    - If X or Y passed as 1D array_like, they are transformed to:
      X = np.atleast_2d(X).T    Y = np.atleast_2d(Y).T

    - Upper case 'X' is 2D training   input and 'Y' is 2D training   target
    - Lower case 'x' is 2D prediction input and 'y' is 2D prediction result
    - XY = (X, Y, xKeys, yKeys) is combination of X and Y with array keys

    Decision tree
    -------------
        if f == 'demo'
            f = f_demo()

        if f is not None:
            if XY is None:
                White()
            else:
                if method.startswith('light'):
                    LightGray()
                elif method.startswith('medium'):
                    if '-l' in method.lower():
                        MediumGray() - local
                    else:
                        MediumGray() - glocal
                else:
                    DarkGray()
        else:
            Black()
    """

    def __init__(self, f, identifier='Model'):
        """
        Args:
            f (method or function):
                theoretical submodel y=f(x) for single data point if f not None

            identifier (string, optional):
                object identifier
        """
        super().__init__(identifier=identifier)
        self.f = f                   # user-defined function (if not black box)

        self._X = None               # input to training (2D array)
        self._Y = None               # target for training (2D array)
        self._x = None               # input to prediction (1D or 2D array)
        self._y = None               # output from prediction  (1D or 2D array)
        self._xKeys = None           # x-keys for data sel. (1D list of string)
        self._yKeys = None           # y-keys for data sel. (1D list of string)

        self._best = None            # (3-tuple) best training: (L2, abs, iAbs)

    @property
    def f(self):
        """
        Returns:
            (method):
                Theoretical model y=f(x) for calculation of single data point
        """
        return self._f

    @f.setter
    def f(self, value):
        """
        Args:
            value (method):
                Theoretical model y=f(x) for calculation of single data point
        """
        if not isinstance(value, str):
            f = value
        else:
            if value.lower().endswith(('demo', 'demo0')):
                f = self.f_demo
            elif value.lower().endswith('demo1'):
                f = self.f_demo1
            else:
                f = None
        if f is not None:
            firstArg = list(inspect.signature(f).parameters.keys())[0]
            if firstArg == 'self':
                f = f.__get__(self, self.__class__)
        self._f = f

    def f_demo(self, x, c0=1, c1=1, c2=1, c3=1, c4=1, c5=1, c6=1, c7=1):
        """
        Function y = f(x): demo calculation of single data point

        Args:
            x (1D array_like of float):
                input x with x.shape: (nInp, )

            c0, c1, ..., c7 (float, optional):
                coefficients

        Returns:
            (1D array_like of float):
                output with shape: (nOut, )
        """
        assert x is not None and len(x) > 1

        # input is 1D array_like with x.shape = (nInp, )
        y0 = c0 * np.sin(c1 * x[0]) + c2 * (x[1] - 1)**2 + c3

        # output is 1D array_like with y.shape = (nOut, )
        return [y0]

    def f_demo1(self, x, **kwargs):
        """
        Function y = f(x): demo calculation of single data point with 'kwargs'

        Args:
            x (1D array_like of float):
                input x with x.shape: (nInp)

            kwargs (dict, optional):
                keyword arguments:

                c0, c1, ..., c7 (float, optional):
                    coefficients

        Returns:
            (1D array_like of float):
                output with shape: (nOut)
        """
        assert x is not None and len(x) > 1

        c0 = kwargs.get('c0', 1)
        c1 = kwargs.get('c1', 1)
        c2 = kwargs.get('c3', 1)
        c3 = kwargs.get('c3', 1)

        # input is 1D array_like, np.array(x).shape: (nInp, )
        y0 = c0 * np.sin(c1 * x[0]) + c2 * (x[0] - 1)**2 + c3

        # output is 1D array_like, np.array(y).shape: (nOut, )
        return [y0]

    @property
    def X(self):
        """
        Returns:
            (2D array of float):
                X array of training input
        """
        return self._X

    @X.setter
    def X(self, value):
        """
        Args:
            value (2D array_like of float):
                X array of training input
        """
        if value is None:
            self._X = None
        else:
            self._X = np.atleast_2d(value)

            assert self._Y is None or self._X.shape[0] == self._Y.shape[0], \
                str(self._X.shape) + str(self._Y.shape)

    @property
    def Y(self):
        """
        Returns:
            (2D array of float):
                Y array of training target
        """
        return self._Y

    @Y.setter
    def Y(self, value):
        """
        Args:
            value (2D array_like of float):
                Y array of training target
        """
        if value is None:
            self._Y = None
        else:
            self._Y = np.atleast_2d(value)

            assert self._X is None or self._X.shape[0] == self._Y.shape[0], \
                str(self._X.shape) + str(self._Y.shape)

    @property
    def x(self):
        """
        Returns:
            (2D array of float):
                x array of prediction input
        """
        return self._x

    @x.setter
    def x(self, value):
        """
        Args:
            value(2D or 1D array_like of float):
                x array of prediction input
        """
        if value is None:
            self._x = None
        else:
            self._x = np.atleast_2d(value)

    @property
    def y(self):
        """
        Returns:
            (2D array of float):
                y array of prediction output
        """
        return self._y

    @y.setter
    def y(self, value):
        """
        Args:
            value(2D or 1D array_like of float):
                y array of prediction result
        """
        if value is None:
            self._y = None
        else:
            self._y = np.atleast_2d(value)

    @property
    def XY(self):
        """
        Returns:
            X (2D array of float):
                X array of training input

            Y (2D array of float):
                Y array of training target

            xKeys (1D list of string):
                list of column keys for data selection
                use self._xKeys keys if xKeys is None,
                default: ['x0', 'x1', ... ]

            yKeys (1D list of string):
                list of column keys for data selection
                use self._yKeys keys if yKeys is None,
                default: ['y0', 'y1', ... ]
        """
        return self._X, self._X, self._xKeys, self._yKeys

    @XY.setter
    def XY(self, value):
        """
        Args:
            value (4-tuple of two arrays of float and two arrays of string):
                X (2D or 1D array_like of float):
                    Input X will be converted to 2D-array
                    (first index is data point index)

                Y (2D or 1D array_like of float):
                    Target Y will be converted to 2D-array
                    (first index is data point index)

                xKeys (1D array_like of string, optional):
                    list of column keys for data selection
                    use self._xKeys keys if xKeys is None,
                    default: ['x0', 'x1', ... ]

                yKeys (1D array_like of string, optional):
                    list of column keys for data selection
                    use self._yKeys keys if yKeys is None,
                    default: ['y0', 'y1', ... ]

        Side effects:
            self._X, self._Y, self._xKeys, self._yKeys will be overwritten
        """
        value = list(value)
        for i in range(len(value), 4):
            value.append(None)
        X, Y, xKeys, yKeys = value
        assert X is not None and Y is not None, str(X is not None)

        self._X = np.atleast_2d(X)
        self._Y = np.atleast_2d(Y)

        assert self._X.shape[0] == self._Y.shape[0], \
            str(self._X.shape) + str(self._Y.shape)

        if xKeys is None:
            self._xKeys = ['x' + str(i) for i in range(self._X.shape[1])]
        else:
            self._xKeys = list(xKeys)
        if yKeys is None:
            self._yKeys = ['y' + str(i) for i in range(self._Y.shape[1])]
        else:
            self._yKeys = list(yKeys)
        assert not any(x in self._yKeys for x in self._xKeys), \
            str(xKeys) + str(yKeys)

    def XY2frame(self, X=None, Y=None):
        """
        Args:
            X (2D or 1D array_like of float, optional):
                X training input. If None, self._X is exported to DataFrame

            Y (2D or 1D array_like of float, optional):
                Y tarining target. If None, assign self._Y to Y

        Returns:
            (pandas.DataFrame):
                Data frame created from X and Y arrays. If X or Y is None,
                self._X and self._Y are used
        """
        if X is None:
            X = self._X
        if Y is None:
            Y = self._Y
        X, Y = np.atleast_2d(X, Y)
        xKeys, yKeys = self._xKeys, self._yKeys
        assert X is not None
        assert Y is not None
        assert X.shape[0] == Y.shape[0], str(X.shape) + ', ' + str(Y.shape)

        if xKeys is None:
            if X is not None:
                xKeys = ['x' + str(i) for i in range(X.shape[1])]
            else:
                xKeys = []
        else:
            xKeys = list(self._xKeys)
        if yKeys is None:
            if Y is not None:
                yKeys = ['y' + str(i) for i in range(Y.shape[1])]
            else:
                yKeys = []
        else:
            yKeys = list(self._yKeys)
        dic = OrderedDict()
        for j in range(len(xKeys)):
            dic[xKeys[j]] = X[:, j]
        for j in range(len(yKeys)):
            dic[yKeys[j]] = Y[:, j]
        return DataFrame(dic)

    def xy2frame(self):
        """
        Returns:
            (pandas.DataFrame);
                data frame created from self._x and self._y
        """
        return self.XY2frame(self._x, self._y)

    def frame2arrays(self, df, keys0, keys1=None, keys2=None, keys3=None,
                     keys4=None, keys5=None, keys6=None, keys7=None):
        """
        Args:
            df (pandas.DataFrame of float):
                data object

            keys0 (1D array_like of string):
                keys for data selection

            keys1..keys7 (1D array_like of string, optional):
                keys for data selection

        Returns:
            (multiple 1D arrays of float):
                column arrays or None if none of the keys found in df
        """
        keysList = [keys0, keys1, keys2, keys3, keys4, keys5, keys6, keys7]
        keysList = [x for x in keysList if x is not None]
        if not keysList:
            return None

        assert all(key in df for keys in keysList for key in keys), \
            'unknown key in: ' + str(keysList) + ', valid keys: ' + df.columns

        col = []
        for keys in keysList:
            col.append(np.asfarray(df.loc[:, keys]))
        n = len(col)
        if n == 1:
            return col[0]
        if n == 2:
            return col[0], col[1]
        if n == 3:
            return col[0], col[1], col[2]
        if n == 4:
            return col[0], col[1], col[2], col[3]
        if n == 5:
            return col[0], col[1], col[2], col[3], col[4]
        if n == 6:
            return col[0], col[1], col[2], col[3], col[4], col[5]
        if n == 7:
            return col[0], col[1], col[2], col[3], col[4], col[5], col[6]
        if n == 8:
            return col[0], col[1], col[2], col[3], col[4], col[5], col[6], \
                   col[7]

    def train(self, X, Y, **kwargs):
        """
        Trains model, this method has to be overwritten in derived classes

        Args:
            X (2D array of float):
                training input, X.shape: (nPoint, nInp)

            Y (2D array of float):
                training target, Y.shape: (nPoint, nOut)

            kwargs (dict, optional):
                keyword arguments

        Returns:
            (3-tuple of float):
                (||y-Y||_2, max{|y-Y|}, index(max{|y-Y|}) if X and Y not None
            or (None):
                if X is None or Y is None or training fails
        """
        if X is None or Y is None:
            self._best = None
            return self._best

        self.ready = False
        print('!!! train() is not implemented in: ' + self.__class__.__name__)
        self.ready = True

        self._best = (np.inf, np.inf, -1)
        return self._best

    def predict(self, x, **kwargs):
        """
        Executes model. If MPI is available, execution is distributed.

        Args:
            x (2D or 1D array_like of float):
                prediction input, shape: (nPoint, nInp) or shape: (nInp)

            kwargs (dict, optional):
                keyword arguments

                c0, c1, ... (float):
                    coefficients for LightGray

                ... options for neural network etc

        Returns:
            self.y (2D array of float):
                prediction result, shape: (nPoint, nOut) if x is not None
            or (3-tuple of (float, float, int)):
                ||y-Y||_2, max{|y-Y|}, index(max{|y-Y|} if x is None
            or (None):
                if model is not ready
        """
        assert self.ready

        if x is None:
            return self._best

        self.x = x                 # self.x is a setter ensuring 2D numpy array
        if not parallel.communicator() or x.shape[0] <= 1:
            opt = {k: v for k, v in kwargs.items()
                   if k.startswith(('c', 'C', 'x')) and k not in ('cross')}

            # self.y is a setter ensuring 2D numpy array
            self.y = [np.atleast_1d(self.f(x, **opt)) for x in self.x]
        else:
            self.y = parallel.predict_scatter(self.f, self.x, **opt)

        return self.y

    def error(self, X, Y, **kwargs):
        """
        Evaluates difference between prediction y(X) and given ref. array Y(X)

        Args:
            X (2D array_like of float):
                reference input, shape: (nPoint, nInp)

            Y (2D array_like of float):
                reference output, shape: (nPoint, nOut)

            kwargs (dict, optional):
                keyword arguments:

                silent (bool):
                    if True then print of norm is suppressed
        Returns:
            (3-tuple of (float, float, int)):
                (||y-Y||_2, max{|y-Y|}, index(max{|y-Y|})
            or (None):
                if X is None or Y is None

        Note:
            maximum index is 1D index, e.g. yAbsMax = Y.ravel()[iAbsMax]
        """
        if X is None or Y is None:
            return None

        assert X.shape[0] == Y.shape[0], str(X.shape) + str(Y.shape)

        y = self.predict(x=X, **self.kwargsDel(kwargs, 'x'))

        try:
            dy = y.ravel() - Y.ravel()
        except ValueError:
            print('X Y y:', X.shape, Y.shape, y.shape)
            assert 0
        L2_norm = np.sqrt(np.mean(np.square(dy)))
        iAbsMax = np.abs(dy).argmax()

        if not kwargs.get('silent', True):
            self.write('    L2-norm: ', np.round(L2_norm, 4))
            self.write('    max(abs(err)): ',
                       np.round(dy[iAbsMax], 5), ' @ (X,Y)+[',
                       str(iAbsMax), ']: (', np.round(X.ravel()[iAbsMax], 3),
                       ', ', np.round(Y.ravel()[iAbsMax], 3), ')')

        return L2_norm, Y.ravel()[iAbsMax], iAbsMax

    def pre(self, **kwargs):
        """
        Args:
            kwargs (dict, optional):
                keyword arguments:

                XY (4-tuple of two 2D array_like of float, and optionally
                    two list of string):
                    training input and train. target, optional keys of X and Y

                X, Y (two 2D array_like of float):
                    training input and training target,
                    X.shape: (nPoint, nInp) and Y.shape: (nPoint, nOut)
        Returns:
            (3-tuple of (float, float, int) or None):
                self._best if 'XY' or ('X' and 'Y') in 'kwargs'
            or (None):
                if 'XY' or ('X' a nd 'Y') not in 'kwargs'

        Side effects:
            self.X and self.Y are overwritten with XY or (X and Y)
            self._best contains result of train() or None if failure
        """
        super().pre(**kwargs)

        XY = kwargs.get('XY', None)
        if XY is not None:
            self.XY = XY
        else:
            self.X, self.Y = kwargs.get('X', None), kwargs.get('Y', None)

        # trains model if self.X is not None and self.Y is not None
        if self.X is not None and self.Y is not None:
            opt = self.kwargsDel(kwargs, ['X', 'Y'])
            self._best = self.train(X=self.X, Y=self.Y, **opt)
        else:
            self._best = None

        return self._best

    def task(self, **kwargs):
        """
        Args:
            kwargs (dict, optional):
                keyword arguments:

                x (2D or 1D array_like of float):
                    prediction input

        Returns:
            self.y (2D array of float):
                prediction result if 'x' in 'kwargs'
            or (3-tuple of (float, float, int) or None):
                self._best if 'x' not in 'kwargs'

        Side effects:
            self.x is overwritten with x
            self.y is overwritten with result of predict()
        """
        super().task(**kwargs)

        x = kwargs.get('x', None)
        if x is None:
            return self._best

        self.x = x                    # self.x is setter ensuring correct shape
        opt = self.kwargsDel(kwargs, 'x')
        self.y = self.predict(x=self.x, **opt)

        return self.y


# Examples ####################################################################

if __name__ == '__main__':
    ALL = 1

    from White import White
    from plotArrays import plotIsoMap

    def fUser(self, x, c0=1, c1=1, c2=1, c3=1, c4=1, c5=1, c6=1, c7=1):
        """
        Customized single point calculation method for Model
        """
        x = np.asfarray(x)
        assert x.ndim == 1, 'x.ndim: ' + str(x.ndim)
        assert x.shape[0] >= 2, 'x.shape: ' + str(x.shape)

        y0 = c0 * x[0]*x[0] + c1 * x[1] + c2
        y1 = x[1] * 2.1
        y = np.asfarray([y0, y1])
        return y

    if 0 or ALL:
        X = grid(5, [-1, 2], [3, 4])
        print('X:', X)

        Y_exa = White(fUser)(x=X)
        plotIsoMap(X[:, 0], X[:, 1], Y_exa[:, 0], title='Y_exa[:,0]')
        plotIsoMap(X[:, 0], X[:, 1], Y_exa[:, 1], title='Y_exa[:,1]')
        print('Y_exa:', Y_exa)

        Y = noise(Y_exa, absolute=0.1, uniform=True)
        plotIsoMap(X[:, 0], X[:, 1], Y[:, 0], title='Y[:,0]')
        plotIsoMap(X[:, 0], X[:, 1], Y[:, 1], title='Y[:,1]')
        print('Y:', Y)

        dY = Y - Y_exa
        plotIsoMap(X[:, 0], X[:, 1], dY[:, 0], title='dY[:,0]')
        plotIsoMap(X[:, 0], X[:, 1], dY[:, 1], title='dY[:,1]')
        print('dY:', dY)

    if 0 or ALL:
        # creates instance of Model
        foo = Model(fUser)

        y = foo.f(x=[2, 3], c0=2, c1=0, c2=1)
        print('y:', y)

        # sets input
        foo.x = [1, 2]
        print('1: foo.x:', foo.x, 'foo.y:', foo.y)

        print('test data frame import/export')
        df = foo.xy2frame()
        print('4: df:', df)

        df = foo.XY2frame()
        print('5: df:', df)

        foo.X = [[2, 3], [4, 5]]
        foo.Y = [[22, 33], [44, 55]]
        df = foo.XY2frame()
        print('6: df:', df)

        y0, y1 = foo.frame2arrays(df, ['y0'], ['y1'])
        print('7 y0:', y0, 'y1:', y1)
        y01 = foo.frame2arrays(df, ['y0', 'y1'])
        print('8 y01:', y01)
        y12, x0 = foo.frame2arrays(df, ['y0', 'y1'], ['x0'])
        print('9 y12:', y12, 'x0', x0)

    if 1 or ALL:
        import matplotlib.pyplot as plt

        foo = Model(fUser)
        nPoint = 20
        X = rand(nPoint, [0, 10], [0, 10])
        Y = noise(White(fUser)(x=X), absolute=0.1)
        x = rand(nPoint, [0, 10], [0, 10])
        y_exa = White(fUser)(x=x)

        # trains light white box model. If an instance of Theoretical is not
        # trained, it provides via predict() the white box model f(x)
        combinedTrainAndPredict = False
        trainWhiteBox = False
        if not combinedTrainAndPredict:
            if trainWhiteBox:
                # trains light white box model. If an instance of Theoretical
                # is not trained, it provides later with predict() the return
                # of f(x)
                foo(X=X, Y=Y)
                # predicts light weight model or white box model (if not train)
            y_fit = foo(x=x)
        else:
            # trains and executes light white box model
            y_fit = foo(X=X, Y=Y, x=x)

        y_exa = np.atleast_2d([foo.f(_x) for _x in x])

        plt.scatter(X.T[0], Y.T[0], marker='o', label='Y(X) train')
        plt.scatter(X.T[0], y_exa.T[0], marker='s', label='y(x) exa')
        plt.scatter(foo.x.T[0], y_fit.T[0], marker='v', label='y(x) fit')
        plt.legend()
        plt.show()
        plt.scatter(foo.x.T[0], y_fit.T[0] - y_exa.T[0], marker='s',
                    label='Y_fit - y_exa[0]')
        plt.scatter(foo.x.T[1], y_fit.T[1] - y_exa.T[1], marker='s',
                    label='Y_fit - y_exa[1]')
        plt.legend()
        plt.show()
