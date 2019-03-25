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
      2019-03-22 DWW
"""

__all__ = ['BoxModel', 'grid', 'cross', 'rand', 'noise', 'frame2arr']

import inspect
import random
from collections import OrderedDict
import numpy as np
from pandas import DataFrame
from typing import Any, Callable, Dict, Optional, List, Sequence, Tuple, Union

from grayboxes.base import Base
from grayboxes.parallel import communicator, predict_scatter


def grid(n: Union[int, Sequence[int]], *ranges: Tuple[float, float]) \
        -> np.ndarray:
    """
    Sets initial (uniformly spaced) grid input, for instance for 2 input
    with 4 nodes per axis: grid(4, [3., 6.], [-7., -5.5])


       -5.5  x-----x-----x-----x
             |     |     |     |
         -6  x-----x-----x-----x
             |     |     |     |
       -6.5  x-----x-----x-----x
             |     |     |     |
         -7  x-----x-----x-----x
             3     4     5     6

    Args:
        n:
            Number of nodes per axis for which initial values are
            generated. If n is single and NEGATIVE, the array will be
            transformed to 2D and transposed:
            grid(-5, [0, 1]) ==> [[0], [.25], [.5], [.75], [1]]

        ranges:
            Variable length argument list of (min, max) pairs

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
    xVar: List[np.ndarray] = []
    for rng, _n in zip(ranges, n):
        rng_min = min(rng[0], rng[1])
        rng_max = max(rng[0], rng[1])
        xVar.append(np.linspace(rng_min, rng_max, abs(_n)))

    if ranges.shape[0] == 1:
        x = xVar[0]
        # if argument n is a negative int:
        if n[0] < 0:
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


def cross(n: Union[int, Sequence[int]], *ranges: Tuple[float, float]) \
        -> np.ndarray:

    """
    Sets initial (uniformly spaced) cross input, for instance for 2 
    input with 5 nodes per axis: cross(5, [3., 7.], [-4., -2.])

                 -2.0
                   |
                 -2.5
                   |
      3.0---4.0---ref---6.0---7.0      ref = (5.0, -3.0)
                   |
                 -3.5
                   |
                 -4.0
    Args:
        n:
            number of nodes per axis for which initial values generated
            n is corrected to the next odd number if n is even

        ranges (variable length argument list of pairs of float):
            list of (min, max) pairs

    Returns:
        (2D array of float):
            Cross-like initial values, shape: (n_point, n_inp). First 
            point is reference point in cross center, see figure
    """
    ranges = list(ranges)
    N = list(np.atleast_1d(n))

    # ensures odd number of nodes per axis
    N = [2 * (n // 2) + 1 for n in N]
    n = N + [N[-1]] * (len(ranges) - len(N))
    assert len(n) == len(ranges), 'n:' + str(n) + ' ranges:' + str(ranges)

    x: List[np.ndarray] = []
    x_center = [np.mean(rng) for rng in ranges]
    x.append(x_center)
    for i, rng in enumerate(ranges):
        if rng[0] != rng[1]:
            x_point = x_center.copy()
            rng_min = min(rng[0], rng[1])
            rng_max = max(rng[0], rng[1])
            x_var = np.linspace(rng_min, rng_max, n[i])
            for j in range(0, n[i]):
                if j != n[i] // 2:
                    x_point[i] = x_var[j]
                    x.append(x_point.copy())
    return np.asfarray(x)


def rand(n: Union[int, Sequence[int]], *ranges: Tuple[float, float]) \
        -> np.ndarray:

    """
    Sets initial (uniformly distributed) random input, for instance for
    2 input with 12 trials: rand(12, [1., 3.], [-7., -5.])

      -5.0 ---------------
           |  x  x  x    |
           |    x x      |
           |   x     x   |
           |  x    x     |
           |    x  x  x  |
      -7.0 ---------------
           1.0         3.0

    Args:
        n:
            number of trials for which initial values random generated

        ranges (variable length argument list of pairs of float):
            list of (min, max) pairs

    Returns:
        (2D array of float):
            Random initial values, first index is trial index,
            second index is input index
    """
    ranges = list(ranges)
    ranges = np.atleast_2d(ranges)
    assert ranges.shape[1] == 2, 'ranges: ' + str(ranges)
    assert n > 0, 'n: ' + str(n)
    assert all(x[0] <= x[1] for x in ranges), 'ranges: ' + str(ranges)

    x = np.array([[random.uniform(min(rng[0], rng[1]), max(rng[0], rng[1]))
                  for rng in ranges] for _ in range(n)])
    return x


def noise(y: np.ndarray, absolute: float=0.0, relative: float=0e-2,
          uniform: bool=True) -> Optional[np.ndarray]:
    """
    Adds noise to an array_like argument 'y'. The noise can be:
        - normally distributed or
        - uniformly distributed

    The addition to 'y' can be:
        - noise from the interval [-absolute, +absolute] independently of
          actual value of 'y' or
        - noise from the interval [-relative, +relative] proportional to
          actual value of 'y', 'relative' is not in percent


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
        y:
            initial array of any shape

        absolute:
            upper boundary of interval of absolute values of noise to be 
            added. The lower boundary is the opposite of 'absolute'
            
            default: 0.0

        relative:
            upper boundary of interval of relative noise to be added.
            The lower boundary is the opposite of 'relative'.
            The addition is relative to actual value of y, for instance 
            'relative = 0.2' adds noise out of the range from -20% to 
            20% to 'y'

            default: 0.0

        uniform:
            if True then noise is uniformly distributed between the upper 
            and lower boundaries given by 'absolute' and/or 'relative'.
            Otherwise these upper boundaries represent the standard 
            deviation of a Gaussian distribution (at given boundary 
            noise value is 60.7% of max noise )
            
            default: True

    Returns:
        (array of float of same shape as y):
            result is a copy of y plus noise if y is not None
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


def frame2arr(df: DataFrame,
              keys0: Union[str, Sequence[str]],
              keys1: Union[str, Sequence[str], None]=None,
              keys2: Union[str, Sequence[str], None]=None,
              keys3: Union[str, Sequence[str], None]=None,
              keys4: Union[str, Sequence[str], None]=None,
              keys5: Union[str, Sequence[str], None]=None,
              keys6: Union[str, Sequence[str], None]=None,
              keys7: Union[str, Sequence[str], None]=None) \
        -> Optional[List[np.ndarray]]:
    """
    Extracts 1D arrays of float from columns of a pandas DataFrame

    Args:
        df:
            data object

        keys0:
            key(s) of column 0 for data selection

        keys1:
            keys(s) of column 1 for data selection

        keys2:
            keys(s) of column 2 for data selection

        keys3:
            keys(s) of column 3 for data selection

        keys4:
            keys(s) of column 4 for data selection

        keys5:
            keys(s) of column 5 for data selection

        keys6:
            keys(s) of column 6 for data selection

        keys7:
            keys(s) of column 7 for data selection

    Returns:
        (list of 1D arrays of float):
            column arrays of shape: (n_points, len(key?)). Size of tuple equals
            number of keys 'keys0, key1,  .., keys7' which are not None
        or
        (None):
            if all(keys0..7 not in df)
    """
    keys_list = [keys0, keys1, keys2, keys3, keys4, keys5, keys6, keys7]
    keys_list = [x for x in keys_list if x is not None]
    if not keys_list:
        return None

    assert all(key in df for keys in keys_list for key in keys), \
        'unknown key in: ' + str(keys_list) + ', valid keys: ' + df.columns

    col: List[np.ndarray] = []
    for keys in keys_list:
        col.append(np.asfarray(df.loc[:, keys]))
    n = len(col)
    if n == 1:
        return [col[0]]
    if n == 2:
        return [col[0], col[1]]
    if n == 3:
        return [col[0], col[1], col[2]]
    if n == 4:
        return [col[0], col[1], col[2], col[3]]
    if n == 5:
        return [col[0], col[1], col[2], col[3], col[4]]
    if n == 6:
        return [col[0], col[1], col[2], col[3], col[4], col[5]]
    if n == 7:
        return [col[0], col[1], col[2], col[3], col[4], col[5], col[6]]
    if n == 8:
        return [col[0], col[1], col[2], col[3], col[4], col[5], col[6], col[7]]


class BoxModel(Base):
    """
    Parent class of White, LightGray, MediumGray, DarkGray and Black

    - Input and output are 2D arrays. First index is data point index
    - If X or Y is passed as 1D array_like, then they are transformed to
          self.X = np.atleast_2d(X).T and self.Y = np.atleast_2d(Y).T

    - Upper case 'X' is 2D   training input and 'Y' is 2D train. target
    - Lower case 'x' is 2D prediction input and 'y' is 2D pred. output
    - XY = (X, Y, x_keys, y_keys) is union of X and Y,optional with keys

    Decision tree
    -------------
        if f == 'demo'
            f = f_demo()

        if f is not None:
            if XY is None:
                White()
            else:
                if model_color.startswith('light'):
                    LightGray()
                elif model_color.startswith('medium'):
                    if '-loc' in model_color:
                        MediumGray() - local tuning pars from empirical
                    else:
                        MediumGray() - global training
                else:
                    DarkGray()
        else:
            Black()
    """

    def __init__(self, f: Callable, identifier: str='BoxModel') -> None:
        """
        Args:
            f:
                Theoretical submodel f(self, x, *args, **kwargs) or
                f(x, *args, **kwargs) for single data point

            identifier:
                Unique object identifier
        """
        super().__init__(identifier=identifier)
        self.f: Callable = f  # theoretical submodel if not black box
        self._X: Optional[np.ndarray] = None  # training input  (2D arr)
        self._Y: Optional[np.ndarray] = None  # training target (2D arr)
        self._x: Optional[np.ndarray] = None  # pred. input  (1D/2D arr)
        self._y: Optional[np.ndarray] = None  # pred. output (1D/2D arr)
        self._x_keys: Optional[List[str]] = None  # x-keys for data selec
        self._y_keys: Optional[List[str]] = None  # y-keys for data selec
        self._best: Dict[str, Any] = self.init_results()  # init best
        self._weights: Optional[np.ndarray] = None  # weights of emp.mod

    def init_results(self, keys: Union[str, Sequence[str], None]=None,
                     values: Union[Any, Sequence[Any], None]=None) \
            -> Dict[str, Any]:
        """
        Sets default values to result dictionary

        Args:
            keys:
                list of keys to be updated or added

            values:
                list of values to be updated or added

        Returns:
            default for results of best training trial,
            see BoxModel.train()
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
    def f(self) -> Callable:
        """
        Returns:
            Theoretical submodel f(self, x, *args, **kwargs)
        """
        return self._f

    @f.setter
    def f(self, value: Union[Callable, str]) -> None:
        """
,        Args:
            value:
                theoretical submodel f(self, x, *args, **kwargs) or
                    f(x, *args, **kwargs) for single data point
                or
                if value is 'demo' or 'rosen', then f_demo() is assigned
                    to self.f
        """
        if not isinstance(value, str):
            f = value
        else:
            f = self.f_demo if value.startswith(('demo', 'rosen')) else None
        if f is not None:
            first_arg = list(inspect.signature(f).parameters.keys())[0]
            if first_arg == 'self':
                f = f.__get__(self, self.__class__)
        self._f = f

    def f_demo(self, x: np.ndarray, *args: float, **kwargs: Any) \
            -> List[float]:
        """
        Demo function f(self, x) for single data point (Rosenbrook function)

        Args:
            x (1D array of float):
                input, shape: (n_inp,)

            args:
                tuning parameters as positional arguments

        Kwargs:
            c0, c1, ... (float):
                coefficients

        Returns:
            (1D list of float):
                output, shape: (n_out,)
        """
        # minimum at f(a, a**2) = f(1, 1) = 0
        a, b = args if len(args) > 0 else (1, 100)
        y0 = (a - x[0])**2 + b * (x[1] - x[0]**2)**2
        return [y0]

    @property
    def X(self) -> np.ndarray:
        """
        Returns:
            (2D array of float):
                X array of training input, shape: (n_point, n_inp)
        """
        return self._X

    @X.setter
    def X(self, value: np.ndarray) -> None:
        """
        Args:
            value (2D or 1D array of float):
                X array of training input, shape: (n_point, n_inp)
                or (n_point,)
        """
        if value is None:
            self._X = None
        else:
            self._X = np.atleast_2d(value)

            assert self._Y is None or self._X.shape[0] == self._Y.shape[0], \
                str(self._X.shape) + str(self._Y.shape)

    @property
    def Y(self) -> np.ndarray:
        """
        Returns:
            (2D array of float):
                Y array of training target, shape: (n_point, n_out)
        """
        return self._Y

    @Y.setter
    def Y(self, value: np.ndarray) -> None:
        """
        Args:
            value (2D or 1D array of float):
                Y array of training target, shape: (n_point, n_out)
                or (n_out,)
        """
        if value is None:
            self._Y = None
        else:
            self._Y = np.atleast_2d(value)

            assert self._X is None or self._X.shape[0] == self._Y.shape[0], \
                str(self._X.shape) + str(self._Y.shape)

    @property
    def x(self) -> np.ndarray:
        """
        Returns:
            (2D array of float):
                x array of prediction input, shape: (n_point, n_inp)
        """
        return self._x

    @x.setter
    def x(self, value: np.ndarray) -> None:
        """
        Args:
            value(2D or 1D array of float):
                x array of prediction input, shape: (n_point, n_inp)
                or (n_inp,)
        """
        if value is None:
            self._x = None
        else:
            self._x = np.atleast_2d(value)

    @property
    def y(self) -> np.ndarray:
        """
        Returns:
            (2D array of float):
                y array of prediction output, shape: (n_point, n_out)
        """
        return self._y

    @y.setter
    def y(self, value: np.ndarray) -> None:
        """
        Args:
            value(2D or 1D array of float):
                y array of prediction output, shape: (n_point, n_out)
                or (n_out,)
        """
        if value is None:
            self._y = None
        else:
            self._y = np.atleast_2d(value)

    @property
    def XY(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Returns:
            X (2D or 1D array of float):
                training input, shape: (n_point, n_inp) or (n_point,)

            Y (2D or 1D array of float):
                training target, shape: (n_point, n_out) or (n_point,)

            x_keys (1D list of str):
                list of column keys for data selection
                use self._x_keys keys if x_keys is None,
                default: ['x0', 'x1', ... ]

            y_keys (1D list of str):
                list of column keys for data selection
                use self._y_keys keys if y_keys is None,
                default: ['y0', 'y1', ... ]
        """
        return self._X, self._X, self._x_keys, self._y_keys

    @XY.setter
    def XY(self, value: Tuple[np.ndarray, np.ndarray, Optional[Sequence[str]],
                              Optional[Sequence[str]]]) -> None:
        """
        Args:
            value (4-tuple of two arrays of float and two arrays of str):
                X (2D or 1D array_like of float):
                    training input, shape: (n_point, n_inp) or (n_point,)

                Y (2D or 1D array of float):
                    training target, shape: (n_point, n_out) or (n_point,)

                x_keys (1D array of str or None):
                    list of column keys for data selection
                    use self._x_keys keys if x_keys is None,
                    default: ['x0', 'x1', ... ]

                y_keys (1D array of str or None):
                    list of column keys for data selection
                    use self._y_keys keys if y_keys is None,
                    default: ['y0', 'y1', ... ]

        Side effects:
            self._X, self._Y, self._x_keys, self._y_keys will be overwritten
        """
        value = list(value)
        for _ in range(len(value), 4):
            value.append(None)
        X, Y, x_keys, y_keys = value
        assert X is not None and Y is not None, str(X is not None)

        self._X = np.atleast_2d(X)
        self._Y = np.atleast_2d(Y)

        assert self._X.shape[0] == self._Y.shape[0], \
            str(self._X.shape) + str(self._Y.shape)

        if x_keys is None:
            self._x_keys = ['x' + str(i) for i in range(self._X.shape[1])]
        else:
            self._x_keys = list(x_keys)
        if y_keys is None:
            self._y_keys = ['y' + str(i) for i in range(self._Y.shape[1])]
        else:
            self._y_keys = list(y_keys)
        assert not any(x in self._y_keys for x in self._x_keys), \
            str(x_keys) + str(y_keys)

    def XY2frame(self, X: Optional[np.ndarray]=None,
                 Y: Optional[np.ndarray]=None) -> DataFrame:
        """
        Args:
            X (2D or 1D array of float):
                training input, shape: (n_point, n_inp) or (n_point,)
                default: self.X

            Y (2D or 1D array of float):
                training target, shape: (n_point, n_out) or (n_point,)
                default: self.Y

        Returns:
            Data frame created from X and Y arrays
        """
        if X is None:
            X = self._X
        if Y is None:
            Y = self._Y
        X, Y = np.atleast_2d(X, Y)
        x_keys, y_keys = self._x_keys, self._y_keys
        assert X is not None
        assert Y is not None
        assert X.shape[0] == Y.shape[0], str(X.shape) + ', ' + str(Y.shape)

        if x_keys is None:
            if X is not None:
                x_keys = ['x' + str(i) for i in range(X.shape[1])]
            else:
                x_keys: List[str] = []
        else:
            x_keys = list(self._x_keys)
        if y_keys is None:
            if Y is not None:
                y_keys = ['y' + str(i) for i in range(Y.shape[1])]
            else:
                y_keys: List[str] = []
        else:
            y_keys = list(self._y_keys)
        dic = OrderedDict()
        for j in range(len(x_keys)):
            dic[x_keys[j]] = X[:, j]
        for j in range(len(y_keys)):
            dic[y_keys[j]] = Y[:, j]
        return DataFrame(dic)

    def xy2frame(self) -> DataFrame:
        """
        Returns:
            Data frame created from self._x and self._y
        """
        return self.XY2frame(self._x, self._y)

    @property
    def weights(self) -> np.ndarray:
        """
        Returns:
            Array of weights
        """
        return self._weights

    @weights.setter
    def weights(self, value: Optional[np.ndarray]) -> None:
        """
        Args:
            value (2D or 1D array of float):
                Array of weights
        """
        if value is None:
            self._weights = None
        else:
            self._weights = np.array(value)

    def train(self, X: np.ndarray, Y: np.ndarray, **kwargs: Any) \
            -> Dict[str, Any]:
        """
        Trains model. This method has to be overwritten in derived classes.
        X and Y are stored as self.X and self.Y if both are not None

        Args:
            X (2D or 1D array of float):
                training input, shape: (n_point, n_inp) or (n_point,)

            Y (2D or 1D array of float):
                training target, shape: (n_point, n_out) or (n_point,)

        Kwargs:
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
            (dictionary):
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
        self.best = self.init_results()

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

    def predict(self, x: np.ndarray, *args: float, **kwargs: Any) \
            -> Optional[np.ndarray]:
        """
        Executes model. If MPI is available, execution is distributed.
        x and y=f(x) is stored as self.x and self.y

        Args:
            x (2D or 1D array of float):
                prediction input, shape: (n_point, n_inp) or (n_inp,)

            args (*float):
                positional arguments to be passed to theoretical
                submodel f()

        Kwargs:
            Keyword arguments to be passed to theoretical submodel f()

        Returns:
            (2D array of float):
                if x is not None and self.ready: prediction output
            or
            (None):
                otherwise
        """
        self.x = x          # self.x is a setter ensuring 2D numpy array

        if not self.ready:
            self.y = None
        elif not communicator() or x.shape[0] <= 1:
            # self.y is a setter ensuring 2D numpy array
            self.y = [np.atleast_1d(self.f(x, *args)) for x in self.x]
        else:
            self.y = predict_scatter(
                self.f, self.x, *args, **self.kwargs_del(kwargs, 'x'))
        return self.y

    def error(self, X: np.ndarray, Y: np.ndarray, *args: float,
              **kwargs: Any) -> Dict[str, Any]:
        """
        Evaluates difference between prediction y(X) and reference Y(X)

        Args:
            X (2D array of float):
                reference input, shape: (n_point, n_inp)

            Y (2D array of float):
                reference output, shape: (n_point, n_out)

            args (*float):
                positional arguments

        Kwargs:
                silent (bool):
                    if True then print of norm is suppressed
                    default: False
        Returns:
            (dictionary):
                result of evaluation
                    'L2'  (float): sqrt{sum{(net(x)-Y)^2}/N} best train
                    'abs' (float): max{|net(x) - Y|} of best training
                    'iAbs'  (int): index of Y where absolute err is max
        Note:
            maximum abs index is 1D index, eg yAbsMax=Y.ravel()[iAbsMax]
        """
        if X is None or Y is None:
            best = self.init_results()
            best = {key: best[key] for key in ('L2', 'abs', 'iAbs')}
        else:
            assert X.shape[0] == Y.shape[0], str(X.shape) + str(Y.shape)

            y = self.predict(x=X, *args, **self.kwargs_del(kwargs, 'x'))

            try:
                dy = y.ravel() - Y.ravel()
            except ValueError:
                print('X Y y:', X.shape, Y.shape, y.shape)
                assert 0
            best = {'L2': np.sqrt(np.mean(np.square(dy))),
                    'iAbs': np.abs(dy).argmax()}
            best['abs'] = dy.ravel()[best['iAbs']]

            if not kwargs.get('silent', True):
                self.write('    L2: ' + str(np.round(best['L2'], 4)) +
                           ' max(abs(y-Y)): ' + str(np.round(best['abs'], 5)) +
                           ' [' + str(best['iAbs']) + '] x,y:(' +
                           str(np.round(X.ravel()[best['iAbs']], 3)) + ', ' +
                           str(np.round(Y.ravel()[best['iAbs']], 3)) + ')')
        return best

    def pre(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Kwargs:
            XY (tuple of two 2D array of float, and
                optionally two list of str):
                training input & training target, optional keys of X and Y
                'XY' supersede 'X' and 'Y'

            X (2D or 1D array of float):
                training input, shape: (n_point, n_inp) or (n_point,)
                'XY' supersede 'X' and 'Y'
                default: self.X

            Y (2D or 1D array of float):
                training target, shape: (n_point, n_out) or (n_point,)
                'XY' supersede 'X' and 'Y'
                default: self.Y

         Returns:
            (dictionary):
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
            kw = self.kwargs_del(kwargs, ['X', 'Y'])
            self.best = self.train(X=self.X, Y=self.Y, **kw)
        else:
            self.best = self.init_results()

        return self.best

    def task(self, **kwargs: Any) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Kwargs:
            x (2D or 1D array of float):
                prediction input, shape: (n_point, n_inp) or (n_inp,)

        Returns:
            (2D array of float):
                if x is not None and self.read: predict y=model(x)
            or
            (dictionary):
                result of train() with best training trial:
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

        self.x = x                # self.x is a setter ensuring 2D shape
        self.y = self.predict(x=self.x, **self.kwargs_del(kwargs, 'x'))
        return self.y
