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
      2018-05-24 DWW
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
    with 4 nodes per axis: grid(4, [1, 3], [-7, -5])

            x---x---x---x
            |   |   |   |
            x---x---x---x
            |   |   |   |
            x---x---x---x
            |   |   |   |
            x---x---x---x
    Args:
        n (int or 1D array_like of int):
            number of nodes per axis for which initial values are generated.
            If n is single and negative, the array will be transformed to 2D
            and transposed: grid(-5, [0, 1]) ==> [[0], [.25], [.5], [.75], [1]]

        ranges (variable length argument list of pairs of float):
            list of (min, max) pairs

    Returns:
        (2D array of float):
            Grid-like initial values, first index is point index, second
            index is input index
    """
    ranges = list(ranges)
    N = list(np.atleast_1d(n))
    n = N + [N[-1]] * (len(ranges) - len(N))  # fill n-array up to: len(ranges)
    assert len(n) == len(ranges), 'n:' + str(n) + ' ranges:' + str(ranges)

    ranges = np.asfarray(ranges)
    xVar = []
    for rng, _n in zip(ranges, n):
        rngMin = min(rng[0], rng[1])
        rngMax = max(rng[0], rng[1])
        xVar.append(np.linspace(rngMin, rngMax, abs(_n)))

    if ranges.shape[0] == 1:
        x = xVar[0]
        if _n < 0:
            x = np.atleast_2d(x).T
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
    with 5 nodes per axis: cross(5, [1, 2], [-4, -3])
                   x
                   |
                   x
                   |
            x--x--ref--x--x
                   |
                   x
                   |
                   x
    Args:
        n (int):
            number of nodes per axis for which initial values are generated
            n is corrected to an odd number if n is even

        ranges (variable length argument list of pairs of float):
            list of (min, max) pairs

    Returns:
        (2D array of float):
            Cross-like initial values, shape: (nPoint, nInp). First point is
            reference point in cross center, see figure
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
    input with 12 trials: rand(12, [1, 3], [-7, -5])

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


def noise(y, absolute=0.0, relative=0.0, uniform=True):
    """
    Adds noise to an array_like argument 'y'. The noise can be:
        - normally distributed or
        - uniformly distributed

    The addition to 'y' can be:
        - noise from the interval [-absolute, +absolute] independently of the
          actual value of 'y' or
        - noise from the interval [-relative, +relative] proportional to the
          actual value of 'y'


        y
        |                      **   *
        |                *    *===*==
        |      *   *  =**=*===*    *
        |     *=*=*==*     **
        |*==**   *
        | **                    === y
        |                       *** y + noise
        +---------------------------------------index

    Args:
        y (array_like of float):
            initial array of any shape

        absolute (float, optional):
            upper boundary of interval of absolute values of noise to be added.
            The lower boundary is the opposite of 'abolute'
            default: 0.0

        relative (float, optional):
            upper boundary of interval of the relative noise to be added.
            The lower boundary is the opposite of 'relative'.
            The addition is relative to the actual value of y.
            default: 0.0

        uniform (bool, optional):
            if True then noise is uniformely distributed between the upper and
            lower boundaries given by 'absolute' and/or 'relative'.
            Otherwise these upper boundaries represent the standard deviation
            of a Gaussian distribution (at given boundarynoise value is 60.7%
            of max noise )
            default: True

    Returns:
        (array of float):
            copy of 'y' plus noise if y is not None
        or
        (None):
            if y is None

    Note:
        Result can be clipped with: y = np.clip(y, [lo0, up0], [lo1, up1], ...)
    """
    if y is None:
        return None
    y = np.asfarray(y).copy()

    if absolute is not None and absolute > 0.:
        if uniform:
            y += np.random.uniform(low=-absolute, high=absolute, size=y.shape)
        else:
            y += np.random.normal(loc=0., scale=absolute, size=y.shape)

    if relative is not None and relative > 0.:
        if uniform:
            y *= 1. + np.random.uniform(low=-relative, high=relative,
                                        size=y.shape)
        else:
            y *= 1. + np.random.normal(loc=0., scale=relative, size=y.shape)
    return y


class Model(Base):
    """
    Parent class of White, LightGray, MediumGray, DarkGray and Black

    - Array definition: input and output are 2D arrays. First index is data
      point index
    - If X or Y passed as 1D array_like, then they are transformed to:
      self.X = np.atleast_2d(X).T and self.Y = np.atleast_2d(Y).T

    - Upper case 'X' is 2D training   input and 'Y' is 2D training   target
    - Lower case 'x' is 2D prediction input and 'y' is 2D prediction output
    - XY = (X, Y, xKeys, yKeys) is combination of X and Y with array keys

    Decision tree
    -------------
        if f == 'demo'
            f = f_demo()

        if f is not None:
            if XY is None:
                White()
            else:
                if color.startswith('light'):
                    LightGray()
                elif color.startswith('medium'):
                    if '-loc' in color.lower():
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
                theoretical submodel f(self, x) or f(x) for single data point

            identifier (str, optional):
                object identifier
        """
        super().__init__(identifier=identifier)
        self.f = f                       # theoretical submodel if not Black

        self._X = None                   # input to training (2D array)
        self._Y = None                   # target for training (2D array)
        self._x = None                   # input to prediction (1D or 2D array)
        self._y = None                   # output of pred.  (1D or 2D array)
        self._xKeys = None               # x-keys for data sel.(1D list of str)
        self._yKeys = None               # y-keys for data sel.(1D list of str)
        self._best = self.initResults()  # initialize best result set
        self._weights = None             # weights of empirical submodels

    def initResults(self, keys=None, values=None):
        """
        Sets default values to results dict

        Args:
            keys (str or 1D array_like of str):
                list of keys to be updated or added

            values (float or int or str, or 1D array_like of float & int& str):
                list of values to be updated or added

        Returns:
            default for results of best training trial,
            see Model.train()
        """
        results = {'method': None,
                   'L2': np.inf, 'abs': np.inf, 'iAbs': -1,
                   'iterations': -1, 'evaluations': -1, 'epochs': -1,
                   'weights': None}
        if keys is not None and values is not None:
            for key, value in zip(np.atleast_1d(keys), np.atleast_1d(values)):
                results[key] = value
        return results

    @property
    def f(self):
        """
        Returns:
            (method or function):
                theoretical submodel f(self, x) or f(x) for single data point
        """
        return self._f

    @f.setter
    def f(self, value):
        """
        If value is 'demo' or 'rosen' then assign f_demo() to self.f
        Otherwise assign 'value'

        Args:
            value (method or function or str):
                theoretical submodel f(self, x) or f(x) for single data point
        """
        if not isinstance(value, str):
            f = value
        else:
            f = self.f_demo if value.startswith(('demo', 'rosen')) else None
        if f is not None:
            firstArg = list(inspect.signature(f).parameters.keys())[0]
            if firstArg == 'self':
                f = f.__get__(self, self.__class__)
        self._f = f

    def f_demo(self, x, *args, **kwargs):
        """
        Demo function f(self, x) for single data point (Rosenbrook function)

        Args:
            x (1D array_like of float):
                input, shape: (nInp)

            args (argument list, optional):
                positional arguments

            kwargs (dict, optional):
                keyword arguments:

                c0, c1, ... (float, optional):
                    coefficients

            kwargs (dict, optional):
                keyword arguments:

        Returns:
            (1D array_like of float):
                output, shape: (nOut)
        """
        # minimum at f(a,a**2)=f(1,1)=0
        a, b = args if len(args) > 0 else (1, 100)
        y0 = (a - x[0])**2 + b * (x[1] - x[0]**2)**2
        return [y0]

    @property
    def X(self):
        """
        Returns:
            (2D array of float):
                X array of training input, shape: (nPoint, nInp)
        """
        return self._X

    @X.setter
    def X(self, value):
        """
        Args:
            value (2D or 1D array_like of float):
                X array of training input, shape: (nPoint, nInp) or (nPoint)
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
                Y array of training target, shape: (nPoint, nOut)
        """
        return self._Y

    @Y.setter
    def Y(self, value):
        """
        Args:
            value (2D or 1D array_like of float):
                Y array of training target, shape: (nPoint, nOut) or (nPoint)
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
                x array of prediction input, shape: (nPoint, nInp)
        """
        return self._x

    @x.setter
    def x(self, value):
        """
        Args:
            value(2D or 1D array_like of float):
                x array of prediction input, shape: (nPoint, nInp) or (nInp,)
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
                y array of prediction output, shape: (nPoint, nOut)
        """
        return self._y

    @y.setter
    def y(self, value):
        """
        Args:
            value(2D or 1D array_like of float):
                y array of prediction output, shape: (nPoint, nOut) or (nOut,)
        """
        if value is None:
            self._y = None
        else:
            self._y = np.atleast_2d(value)

    @property
    def XY(self):
        """
        Returns:
            X (2D or 1D array_like of float):
                training input, shape: (nPoint, nInp) or shape: (nPoint)

            Y (2D or 1D array_like of float):
                training target, shape: (nPoint, nOut) or shape: (nPoint)

            xKeys (1D list of str):
                list of column keys for data selection
                use self._xKeys keys if xKeys is None,
                default: ['x0', 'x1', ... ]

            yKeys (1D list of str):
                list of column keys for data selection
                use self._yKeys keys if yKeys is None,
                default: ['y0', 'y1', ... ]
        """
        return self._X, self._X, self._xKeys, self._yKeys

    @XY.setter
    def XY(self, value):
        """
        Args:
            value (4-tuple of two arrays of float and two arrays of str):
                X (2D or 1D array_like of float):
                    training input, shape: (nPoint, nInp) or shape: (nPoint)

                Y (2D or 1D array_like of float):
                    training target, shape: (nPoint, nOut) or shape: (nPoint)

                xKeys (1D array_like of str, optional):
                    list of column keys for data selection
                    use self._xKeys keys if xKeys is None,
                    default: ['x0', 'x1', ... ]

                yKeys (1D array_like of str, optional):
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
                training input, shape: (nPoint, nInp) or shape: (nPoint)
                default: self.X

            Y (2D or 1D array_like of float, optional):
                training target, shape: (nPoint, nOut) or shape: (nPoint)
                default: self.Y

        Returns:
            (pandas.DataFrame):
                Data frame created from X and Y arrays
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

            keys0 (1D array_like of str):
                keys for data selection

            keys1..keys7 (1D array_like of str, optional):
                keys for data selection

        Returns:
            (tuple of 1D arrays of float):
                column arrays. Size of tuple equals number of keys0..7 which
                is not None
            or
            (None):
                if all(keys0..7 not in df)
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

    @property
    def weights(self):
        """
        Returns:
            (array of float):
                array of weights
        """
        return self._weights

    @weights.setter
    def weights(self, value):
        """
        Args:
            value(2D or 1D array_like of float):
                array of weights
        """
        if value is None:
            self._weights = None
        else:
            self._weights = np.array(value)

    def train(self, X, Y, **kwargs):
        """
        Trains model. This method has to be overwritten in derived classes.
        X and Y are stored as self.X and self.Y if both are not None

        Args:
            X (2D or 1D array_like of float):
                training input, shape: (nPoint, nInp) or shape: (nPoint)

            Y (2D or 1D array_like of float):
                training target, shape: (nPoint, nOut) or shape: (nPoint)

            kwargs (dict, optional):
                keyword arguments

                methods (str or list of str):
                    training methods

                epochs (int):
                    maximum number of epochs

                goal (float):
                    residuum to be met

                trials (int):
                    number of repetitions of training with same method
                ...

        Returns:
            (dict {str: float or str or int}):
                result of best training trial:
                    'method'     (str): best training method
                    'L2'       (float): sqrt{sum{(net(x)-Y)^2}/N} of best train
                    'abs'      (float): max{|net(x) - Y|} of best training
                    'iAbs'       (int): index of Y where absolute error is max
                    'epochs'     (int): number of epochs of best (neural) train
                    'iterations' (int): number of iterations
                    'weights'    (arr): weights if not White box model

        Note:
            If X or Y is None, or training fails then self.best['method']=None
        """
        self.ready = True
        self._weights = None
        self.best = self.initResults()

        if X is not None and Y is not None:
            self.X, self.Y = X, Y
            #
            # ... INSERT TRAINING IN DERIVED CLASSES ...
            #
            # SUCCESS = ...
            # self.ready = SUCCESS
            # if self.ready:
            #    self._weights = ...
            #    self.best = {'method': ..., 'L2': ..., 'abs': ...,
            #                 'iAbs': ..., 'epochs': ...}

        return self.best

    def predict(self, x, *args, **kwargs):
        """
        Executes model. If MPI is available, execution is distributed. x and
        y=f(x) is stored as self.x and self.y

        Args:
            x (2D or 1D array_like of float):
                prediction input, shape: (nPoint, nInp) or shape: (nInp,)

            args (list, optional):
                positional arguments

            kwargs (dict, optional):
                keyword arguments

        Returns:
            (2D array of float):
                if x is not None and self.ready: prediction output
            or
            (None):
                otherwise
        """
        self.x = x                 # self.x is a setter ensuring 2D numpy array

        if not self.ready:
            self.y = None
        elif not parallel.communicator() or x.shape[0] <= 1:
            # self.y is a setter ensuring 2D numpy array
            self.y = [np.atleast_1d(self.f(x, *args)) for x in self.x]
        else:
            self.y = parallel.predict_scatter(
                self.f, self.x, *args, **self.kwargsDel(kwargs, 'x'))
        return self.y

    def error(self, X, Y, *args, **kwargs):
        """
        Evaluates difference between prediction y(X) and reference array Y(X)

        Args:
            X (2D array_like of float):
                reference input, shape: (nPoint, nInp)

            Y (2D array_like of float):
                reference output, shape: (nPoint, nOut)

            args (argument list, optional):
                positional arguments

            kwargs (dict, optional):
                keyword arguments:

                silent (bool):
                    if True then print of norm is suppressed
                    default: False
        Returns:
            (dict: {str: float or str or int})
                result of evaluation
                    'L2'    (float): sqrt{sum{(net(x)-Y)^2}/N} of best training
                    'abs'   (float): max{|net(x) - Y|} of best training
                    'iAbs'    (int): index of Y where absolute error is maximum
        Note:
            - maximum abs index is 1D index, e.g. yAbsMax = Y.ravel()[iAbsMax]
        """
        if X is None or Y is None:
            best = self.initResults()
            best = {key: best[key] for key in ('L2', 'abs', 'iAbs')}
        else:
            assert X.shape[0] == Y.shape[0], str(X.shape) + str(Y.shape)

            y = self.predict(x=X, *args, **self.kwargsDel(kwargs, 'x'))

            try:
                dy = y.ravel() - Y.ravel()
            except ValueError:
                print('X Y y:', X.shape, Y.shape, y.shape)
                assert 0
            best = {'L2': np.sqrt(np.mean(np.square(dy)))}
            best['iAbs'] = np.abs(dy).argmax()
            best['abs'] = dy.ravel()[best['iAbs']]

            if not kwargs.get('silent', True):
                self.write('    L2: ', np.round(best['L2'], 4),
                           ' max(abs(y-Y)): ', np.round(best['abs'], 5),
                           ' [', best['iAbs'], ']',
                           ' x,y:(', np.round(X.ravel()[best['iAbs']], 3),
                           ', ', np.round(Y.ravel()[best['iAbs']], 3), ')')
        return best

    def pre(self, **kwargs):
        """
        Args:
            kwargs (dict, optional):
                keyword arguments:

                XY (4-tuple of two 2D array_like of float, and
                    optionally two list of str):
                    training input & training target, optional keys of X and Y
                    'XY' supersede 'X' and 'Y'

                X (2D or 1D array_like of float, optional):
                    training input, shape: (nPoint, nInp) or shape: (nPoint)
                    'XY' supersede 'X' and 'Y'
                    default: self.X

                Y (2D or 1D array_like of float, optional):
                    training target, shape: (nPoint, nOut) or shape: (nPoint)
                    'XY' supersede 'X' and 'Y'
                    default: self.Y

         Returns:
            (dict: {str: float or str or int})
                result of best training trial:
                    'method '    (str): best training method
                    'L2'       (float): sqrt{sum{(net(x)-Y)^2}/N} of best train
                    'abs'      (float): max{|net(x) - Y|} of best training
                    'iAbs'       (int): index of Y where absolute error is max
                    'epochs'     (int): number of epochs of best (neural) train
                    'iterations' (int): number of iterations
                    'weights'    (arr): weights if not White box model
                if 'XY' or ('X' and 'Y') in 'kwargs'

        Side effects:
            self.X and self.Y are overwritten with setter: XY or (X, Y)
        """
        super().pre(**kwargs)

        XY = kwargs.get('XY', None)
        if XY is not None:
            self.XY = XY
        else:
            self.X, self.Y = kwargs.get('X', None), kwargs.get('Y', None)

        # trains model if self.X is not None and self.Y is not None
        if self.X is not None and self.Y is not None:
            kw = self.kwargsDel(kwargs, ['X', 'Y'])
            self.best = self.train(X=self.X, Y=self.Y, **kw)
        else:
            self.best = self.initResults()

        return self.best

    def task(self, **kwargs):
        """
        Args:
            kwargs (dict, optional):
                keyword arguments:

                x (2D or 1D array_like of float):
                    prediction input, shape: (nPoint, nInp) or shape: (nInp,)

        Returns:
            (2D array of float):
                if x is not None and self.read: predict y=model(x)
            or
            (dict: {str: float or str or int})
                otherwise: result of train() with best training trial:
                    'method'     (str): best training method
                    'L2'       (float): sqrt{sum{(net(x)-Y)^2}/N} of best train
                    'abs'      (float): max{|net(x) - Y|} of best training
                    'iAbs'       (int): index of Y where absolute error is max
                    'epochs'     (int): number of epochs of best (neural) train
                    'iterations' (int): number of iterations
                    'weights'    (arr): weights if not White box model

        Side effects:
            x is stored as self.x and prediction output as self.y
        """
        super().task(**kwargs)

        x = kwargs.get('x', None)
        if x is None:
            return self.best

        self.x = x                       # self.x is a setter ensuring 2D shape
        self.y = self.predict(x=self.x, **self.kwargsDel(kwargs, 'x'))
        return self.y


# Examples ####################################################################

if __name__ == '__main__':
    ALL = 1

    import matplotlib.pyplot as plt
    from White import White
    from LightGray import LightGray
    from plotArrays import plotIsoMap, plotSurface, plotIsolines

    def fUser(self, x, *args, **kwargs):
        """
        theoretical submodel y=f(x,c) for single data point
        """
        nFit = 3
        if x is None:
            return nFit
        c0, c1, c2 = args if len(args) > 0 else np.ones(nFit)

        y0 = c2 * x[0]**2 + c1 * x[1] + c0
        y1 = c2 * x[1]
        return [y0, y1]

    if 1 or ALL:
        x = grid(100, [0.9, 1.1], [0.9, 1.1])
        y_exa = White('demo')(x=x)
        y = noise(y_exa, relative=20e-2)

        plotIsoMap(x[:, 0], x[:, 1], y_exa[:, 0], title='$y_{exa,0}$')
        plotSurface(x[:, 0], x[:, 1], y_exa[:, 0], title='$y_{exa,0}$')
        plotIsolines(x[:, 0], x[:, 1], y_exa[:, 0], title='$y_{exa,0}$',
                     levels=[0, 1e-4, 5e-4, .003, .005, .01, .02, .05, .1, .2])
        plotIsoMap(x[:, 0], x[:, 1], y[:, 0], title='$y_0$')
        plotIsoMap(x[:, 0], x[:, 1], (y-y_exa)[:, 0], title='$y_0-y_{exa,0}$')

    if 0 or ALL:
        x = grid(4, [0, 12], [0, 10])
        y_exa = White(fUser)(x=x)
        y = noise(y_exa, relative=20e-2)

        plotIsoMap(x[:, 0], x[:, 1], y_exa[:, 0], title='$y_{exa,0}$')
        plotIsoMap(x[:, 0], x[:, 1], y_exa[:, 1], title='$y_{exa,1}$')
        plotSurface(x[:, 0], x[:, 1], y_exa[:, 0], title='$y_{exa,0}$')
        plotSurface(x[:, 0], x[:, 1], y_exa[:, 1], title='$y_{exa,1}$')
        plotIsoMap(x[:, 0], x[:, 1], y[:, 0], title='$y_0$')
        plotIsoMap(x[:, 0], x[:, 1], y[:, 1], title='$y_1$')
        plotIsoMap(x[:, 0], x[:, 1], (y-y_exa)[:, 0], title='$y_0-y_{exa,0}$')
        plotIsoMap(x[:, 0], x[:, 1], (y-y_exa)[:, 1], title='$y_1-y_{exa,1}$')

    if 0 or ALL:
        X = grid(5, [-1, 2], [3, 4])
        print('X:', X)

        Y_exa = White(fUser)(x=X)
        plotIsoMap(X[:, 0], X[:, 1], Y_exa[:, 0], title='$Y_{exa,0}$')
        plotIsoMap(X[:, 0], X[:, 1], Y_exa[:, 1], title='$Y_{exa,1}$')
        print('Y_exa:', Y_exa)

        Y = noise(Y_exa, absolute=0.1, uniform=True)
        plotIsoMap(X[:, 0], X[:, 1], Y[:, 0], title='$Y_{0}$')
        plotIsoMap(X[:, 0], X[:, 1], Y[:, 1], title='$Y_{1}$')
        print('Y:', Y)

        dY = Y - Y_exa
        plotIsoMap(X[:, 0], X[:, 1], dY[:, 0], title='$Y - Y_{exa,0}$')
        plotIsoMap(X[:, 0], X[:, 1], dY[:, 1], title='$Y - Y_{exa,1}$')
        print('dY:', dY)

    if 0 or ALL:
        model = Model(fUser)
        y = model.f([2, 3], 2, 0, 1)
        print('y:', y)

        # sets input
        model.x = [1, 2]
        print('1: model.x:', model.x, 'model.y:', model.y)

        print('test data frame import/export')
        df = model.xy2frame()
        print('4: df:', df)

        df = model.XY2frame()
        print('5: df:', df)

        model.X = [[2, 3], [4, 5]]
        model.Y = [[22, 33], [44, 55]]
        df = model.XY2frame()
        print('6: df:', df)

        y0, y1 = model.frame2arrays(df, ['y0'], ['y1'])
        print('7 y0:', y0, 'y1:', y1)
        y01 = model.frame2arrays(df, ['y0', 'y1'])
        print('8 y01:', y01)
        y12, x0 = model.frame2arrays(df, ['y0', 'y1'], ['x0'])
        print('9 y12:', y12, 'x0', x0)

    if 0 or ALL:
        model = LightGray(fUser)
        nPoint = 20
        X = rand(nPoint, [0, 10], [0, 10])
        Y = noise(White(fUser)(x=X), absolute=0.1)

        x = rand(nPoint, [0, 10], [0, 10])
        y_exa = White(fUser)(x=x)
        y = model(X=X, Y=Y, x=x)

        plt.scatter(X[:, 0], Y[:, 0], marker='o', label='$Y_0(X_0)$ train')
        plt.scatter(x[:, 0], y_exa[:, 0], marker='s', label='$y_{exa,0}(x_0)$')
        plt.scatter(model.x[:, 0], y[:, 0], marker='v', label='$y_0(x_0)$')
        plt.legend()
        plt.show()
        plt.scatter(model.x[:, 0], y[:, 0] - y_exa[:, 0], marker='s',
                    label='$y_0 - y_{exa,0}$')
        plt.scatter(model.x[:, 0], y[:, 1] - y_exa[:, 1], marker='s',
                    label='$y_1 - y_{exa,1}$')
        plt.legend()
        plt.show()
