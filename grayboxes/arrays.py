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
      2019-08-21 DWW
"""

__all__ = ['grid', 'cross', 'rand', 'noise', 'xy_rand_split', \
           'xy_thin_out', 'frame_to_arrays', 'scale', ]

import random
import numpy as np
from pandas import DataFrame
from typing import Optional, List, Sequence, Tuple, Union


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
            Random initial values, first index is trial index, second 
            index is input index
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
            'relative = 20e-2' adds noise out of the range from -20% to 
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


def frame_to_arrays(df: DataFrame,
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


def xy_rand_split(x: np.ndarray, y: Optional[np.ndarray]=None,
                  fractions: Optional[Sequence[float]]=None) \
        -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Splits randomly one or two 2D arrays into sub-arrays, size of the sub 
    arrays is defined by a list of 'fractions'

    Example:
        from arrays.xy_rand_split import xy_rand_split
        x = [1, 2, 3, 4, 5, 6]
        y = [3, 4, 5, 6, 7, 8]
        x = np.atleast_2d(x).T  # 2D array with one column
        y = np.atleast_2d(y).T  # 2D array with one column
        fractions = [0.3, 0.1, 0.2]
        X, Y = xy_rand_split(x=x, y=y, fractions=fractions)
        # sum(fractions) is 0.6, size (shape[0]) of x and y is 6 
        # -> sub-array sizes are 0.3/0.6*6=3, 0.1/0.6*6=1 and 0.2/0.6*6=2
        
        return values:
            X = [[4, 6, 2], [1], [3, 5]] 
            Y = [[6, 8, 4], [3], [5, 7]]
 
    Args:
        x (2D array of float):
            data array, first index is point index

        y (2D array of float):
            data array, first index is point index

        fractions:
            size of sub arrays is defined by ratios of elements of fractions 
            list relative to sum of fractions
            
    Returns:
        Pair of lists of 2D sub-arrays of float
    """
    assert len(x.shape) == 2, str(x.shape)
    if y is not None:
        assert len(y.shape) == 2, str((x.shape, y.shape))
        assert x.shape[0] == y.shape[0], str((x.shape, y.shape))

    if fractions is None:
        fractions = (0.8, 0.2)
    fractions = [f / sum(fractions) for f in fractions]
    subset_sizes = [round(f * x.shape[0]) for f in list(fractions)]
    defect = x.shape[0] - sum(subset_sizes)
    subset_sizes[0] += defect

    all_indices = np.random.permutation(x.shape[0])
    begin = 0
    subset_indices = [] # all_indices[:subset_sizes[0]]
    for i in range(len(subset_sizes)):
        end = begin + subset_sizes[i]
        subset_indices.append(all_indices[begin:end])
        begin += subset_sizes[i]
    x_split = []
    y_split = [] if y is not None else None
    for indices in subset_indices:
        x_split.append(x[indices, :])
        if y is not None:
            y_split.append(y[indices, :])

    return x_split, y_split


def xy_thin_out(x: np.ndarray, y: np.ndarray, bins: int=32) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Thinning out an a fine array of (x, y) points -> to coarse array
    
    Args
        x (1D array of float):
            arguments of fine array of (x, y) points
            
        y (1D array of float):
            values of fine array of (x, y) points
            
        bins:
            number of (x, y) points of coarse array
            
    Returns
        (1D array of float, 1D array of float):
            array of (x, y) points of coarse array (edge/center) 
    """
    x, y = np.asfarray(x), np.asfarray(y)
    x_thin_corner, y_thin_center = [], []
    delta_i = len(x) // bins
    i_beg = 0
    for i_bin in range(bins):
        i_beg += delta_i
        i_beg = min(i_beg, len(x)-1)
        i_end = min(i_beg + delta_i-1, len(x)-1)
        x_thin_corner.append(x[i_beg])
        y_thin_center.append(np.mean(y[i_beg:i_end+1]))
    if i_end < len(x):
        x_thin_corner.append(x[-2])
        y_thin_center.append(np.mean(y[i_end:-1]))
    
    return np.asfarray(x_thin_corner), np.asfarray(y_thin_center)


def scale(X: np.ndarray, lo: float=0., hi: float=1., 
          axis: Optional[int]=None) -> np.ndarray:
    """
    Normalizes elements of array to [lo, hi] interval (linear)

    Args:
        X:
            array of data to be normalised

        lo:
            minimum of returned array

        hi:
            maximum of returned array

        axis:
            if not None, minimum and maximum are taken from axis given by 
            axis index (for 2D array: column=0, row=1)

    Returns:
        (array of float):
            normalized array
    """
    np.asarray(X)
    _max = np.max(X, axis=axis)
    delta = _max - np.min(X, axis=axis)
    assert not np.isclose(delta, 0), str(delta)

    return hi - (((hi - lo) * (_max - X)) / delta)
