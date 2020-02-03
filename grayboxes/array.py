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
      2020-02-03 DWW
"""

__all__ = ['grid', 'cross', 'rand', 'noise', 'xy_rand_split', \
           'xy_thin_out', 'frame_to_arrays', 'scale', 'smooth', 
           'convert_to_2d']

import numpy as np
from pandas import DataFrame
import random
from statsmodels.nonparametric.smoothers_lowess import lowess
from typing import Optional

try:
    from grayboxes.datatypes import Float1D, Float2D
except ImportError:
    try:
        from datatypes import Float1D, Float2D
    except ImportError:
        print('    continue with unauthorized definition of Float1D, ' +
              'Float2D')        
        Float1D = Optional[np.ndarray]
        Float2D = Optional[np.ndarray]

from typing import Iterable, Optional, List, Tuple, Union


def grid(size: Union[int, Iterable[int]], 
         *ranges: Union[Tuple[float, float], Iterable[float]]) -> Float2D:
    """
    Sets initial (uniformly spaced) grid input, for instance for 2 input
    with 4 nodes on all axes: grid(n=4, [3., 6.], [-7., -5.5])

       -5.5  x-----x-----x-----x
             |     |     |     |
         -6  x-----x-----x-----x
             |     |     |     |
       -6.5  x-----x-----x-----x
             |     |     |     |
         -7  x-----x-----x-----x
             3     4     5     6

    If the number of nodes is 4 x 3, then grid() is called as:   
    grid(size=(4, 3), [3., 6.], [-7., -6.])
    
         -6  x-----x-----x-----x
             |     |     |     |
       -6.5  x-----x-----x-----x
             |     |     |     |
         -7  x-----x-----x-----x
             3     4     5     6
             
    Args:
        size:
            Number of nodes per axis for which initial values are
            generated. 
            
            if size is a tuple, the number of array elements is the 
            product of the shape dimensions 
            
            If size is single and NEGATIVE, the array will be
            transformed to 2D and transposed:
            grid(size=-5, [0, 1])     ==> [[0], [.25], [.5], [.75], [1]]
            grid(size=(1, 5), [0, 1]) ==> [[0], [.25], [.5], [.75], [1]]
            
            if size is array_like, it defines the number of points per axis 

        ranges:
            Variable length argument list of (min, max) pairs

    Returns:
        Grid-like initial values, first index is point index, second
        index is input index
    """
    N = list(np.atleast_1d(size))
    ranges_ = np.asfarray(list(ranges))
    
    if ranges_.shape[0] == 1 and len(N) == 2:
        if N[1] == 1:
            # grind((N, 1), [rng_min, rng_max]) ==> [[x0], [x1], ... ]
            size = [-N[0]]
        if N[0] == 1:
            # grind((1, N), [rng_min, rng_max]) ==> [x0, x1, ... ]
            size = [N[1]]
    else:
        # fill array up n-array to: len(ranges)
        size = N + [N[-1]] * (len(ranges_) - len(N))  
        assert len(size) == len(ranges_), str((size, ranges_))

    x_var: List[Float1D] = []
    for rng, _n in zip(ranges_, size):
        rng_min = min(rng[0], rng[1])
        rng_max = max(rng[0], rng[1])
        x_var.append(np.linspace(rng_min, rng_max, abs(_n)))

    if ranges_.shape[0] == 1:
        x = x_var[0]
        # if argument n is a negative int:
        if size[0] < 0.:
            x = np.atleast_2d(x).T
    elif ranges_.shape[0] == 2:
        x0, x1 = np.meshgrid(x_var[0], x_var[1])
        x = [(a0, a1) for a0, a1 in zip(x0.ravel(), x1.ravel())]
    elif ranges_.shape[0] == 3:
        x0, x1, x2 = np.meshgrid(x_var[0], x_var[1], x_var[2])
        x = [(a0, a1, a2) for a0, a1, a2 in
             zip(x0.ravel(), x1.ravel(), x2.ravel())]
    elif ranges_.shape[0] == 4:
        x0, x1, x2, x3 = \
            np.meshgrid(x_var[0], x_var[1], x_var[2], x_var[3])
        x = [(a0, a1, a2, a3) for a0, a1, a2, a3 in
             zip(x0.ravel(), x1.ravel(), x2.ravel(),
                 x3.ravel())]
    elif ranges_.shape[0] == 5:
        x0, x1, x2, x3, x4 = \
            np.meshgrid(x_var[0], x_var[1], x_var[2], x_var[3], x_var[4])
        x = [(a0, a1, a2, a3, a4) for a0, a1, a2, a3, a4 in
             zip(x0.ravel(), x1.ravel(), x2.ravel(),
                 x3.ravel(), x4.ravel())]
    elif ranges_.shape[0] == 6:
        x0, x1, x2, x3, x4, x5 = \
            np.meshgrid(x_var[0], x_var[1], x_var[2], x_var[3], x_var[4],
                        x_var[5])
        x = [(a0, a1, a2, a3, a4, a5) for a0, a1, a2, a3, a4, a5 in
             zip(x0.ravel(), x1.ravel(), x2.ravel(),
                 x3.ravel(), x4.ravel(), x5.ravel())]
    else:
        assert 0, 'ranges_: ' + str(ranges_)

    return np.asfarray(x)


def cross(n: Union[int, Iterable[int]], 
          *ranges: Tuple[float, float]) -> Float2D:
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

            if n is array_like, it defines the number of points per axis 

        ranges:
            Variable length argument list of (min, max) pairs

    Returns:
        Cross-like initial values, shape: (n_point, n_inp). First 
        point is reference point in cross center, see figure
    """
    ranges_ = list(ranges)
    N = list(np.atleast_1d(n))

    # ensures odd number of nodes per axis
    N = [2 * (n // 2) + 1 for n in N]
    n = N + [N[-1]] * (len(ranges_) - len(N))
    assert len(n) == len(ranges_), 'n:' + str(n) + ' ranges_:' + str(ranges_)

    x = []
    x_center = [np.mean(rng) for rng in ranges_]
    x.append(x_center)
    for i, rng in enumerate(ranges_):
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


def rand(n: int, *ranges: Tuple[float, float]) -> Float2D:
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

        ranges:
            Variable length argument list of (min, max) pairs

    Returns:
        Random initial values, first index is trial index, second 
        index is input index
    """
    ranges_ = np.atleast_2d(list(ranges))
    assert ranges_.shape[1] == 2, 'ranges_: ' + str(ranges_)
    assert n > 0, 'n: ' + str(n)
    assert all(x[0] <= x[1] for x in ranges_), 'ranges_: ' + str(ranges_)

    x = np.array([[random.uniform(min(rng[0], rng[1]), max(rng[0], rng[1]))
                  for rng in ranges_] for _ in range(n)])
    return x


def noise(y: np.ndarray, 
          absolute: float = 0.0, 
          relative: float = 0e-2,
          uniform: bool = True) -> Optional[np.ndarray]:
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

        relative:
            upper boundary of interval of relative noise to be added.
            The lower boundary is the opposite of 'relative'.
            The addition is relative to actual value of y, for instance 
            'relative = 20e-2' adds noise out of the range from -20% to 
            20% to 'y'

        uniform:
            if True then noise is uniformly distributed between the upper 
            and lower boundaries given by 'absolute' and/or 'relative'.
            Otherwise these upper boundaries represent the standard 
            deviation of a Gaussian distribution (at given boundary 
            noise value is 60.7% of max noise )

    Returns:
        copy of y plus noise if y is not None, return shape is: x.shape
        or
        None if y is None

    Note:
        Result can be clipped with: y = np.clip(y, [lo0, up0], [lo1, up1], ...)
    """
    if y is None:
        return None
    y_ = np.asfarray(y).copy()

    if absolute is not None and absolute > 0.:
        if uniform:
            y_ += np.random.uniform(low=-absolute, high=absolute, 
                                    size=y_.shape)
        else:
            y_ += np.random.normal(loc=0., scale=absolute, size=y_.shape)

    if relative is not None and relative > 0.:
        if uniform:
            y_ *= 1. + np.random.uniform(low=-relative, high=relative,
                                         size=y_.shape)
        else:
            y_ *= 1. + np.random.normal(loc=0., scale=relative, size=y_.shape)

    return y_


def frame_to_arrays(df: DataFrame, *keys: str) -> Optional[List[Float1D]]:
    """
    Extracts 1D arrays of float from columns of a pandas DataFrame

    Args:
        df:
            data object

        keys:
            key(s) for data selection

    Returns:
        Column arrays of shape: (len(keys), n_point)
        or
        None if len(keys) == 0 or invalid keys
    """
    if len(keys) == 0:
        return None
    
    keys_group = list(np.atleast_1d(keys))
    for keys_ in keys_group:
        keys_ = list(np.atleast_1d(keys_))
        for key in keys_:
            if key not in df:
                assert str((keys, df.columns))

    col: List[Float1D] = []
    for key in keys:
        col.append(np.asfarray(df.loc[:, key]))
        
    return col


def xy_rand_split(x: Float2D,
                  y: Float2D = None,
                  fractions: Optional[Iterable[float]] = None) \
        -> Tuple[Float2D, Float2D]:
    """
    Splits randomly one or two 2D arrays into sub-arrays, size of sub 
    arrays is defined by a list of 'fractions'

    Example:
        from array.xy_rand_split import xy_rand_split
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
        x:
            data array, first index is point index

        y:
            data array, first index is point index

        fractions:
            size of sub arrays is defined by ratios of elements of 
            fractions list relative to sum of fractions
            
    Returns:
        2-tuple of (list of 2D x-sub-arrays, list of 2D y-sub-arrays)
        or
        2-tuple of (list of 2D x-sub-arrays, None)
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
    x_split: List[Float2D] = []
    y_split: Optional[List[Float2D]] = [] if y is not None else None
    for indices in subset_indices:
        x_split.append(x[indices, :])
        if y is not None:
            y_split.append(y[indices, :])

    return x_split, y_split


def xy_thin_out(x: Iterable[float], 
                y: Iterable[float], 
                bins: int = 32) \
        -> Tuple[Float1D, Float1D]:
    """
    Thinning out an a fine array of (x, y) points -> to coarse array
    
    
    x x x x x x x x x x x x x x x x x x x x x x x x x  fine array
    
                            |
                            v
                            
    x   x   x   x   x   x   x   x   x   x   x   x   x  thinned-out array
    
    
    Args
        x:
            arguments of fine array of (x, y) points
            
        y:
            values of fine array of (x, y) points
            
        bins:
            number of (x, y) points of coarse array
            
    Returns:
        pair of arrays of (x, y) points of thinned-out array (edge/center) 
    """
    x, y = np.asfarray(x), np.asfarray(y)

    if len(x) < bins:
        return x, y

    assert len(x) == len(y), str((x.shape, y.shape))
    assert len(x.shape) == 1, str((x.shape, y.shape))

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


def scale(X: Iterable[float], 
          lo: float = 0., 
          hi: float = 1., 
          axis: Optional[int] = None) -> Float1D:
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
            if not None, minimum and maximum are taken from axis given 
            by axis index (for 2D array: column=0, row=1)

    Returns:
        normalized array
    """
    X = np.asfarray(X)
    max_ = np.max(X, axis=axis)
    delta = max_ - np.min(X, axis=axis)
    assert not np.isclose(delta, 0), str(delta)

    return hi - (((hi - lo) * (max_ - X)) / delta)


def convert_to_2d(value: Optional[Union[float, Iterable[float], 
                                        Float1D, Float2D]]) -> Float2D: 
    """
    Args:
        value:
            Scalar or, 1D or 2D array of float
            
    Returns:
        2D numpy array
        
    Example:
        X = convert_to_2d(2.)              => array([[2.]])
        X = convert_to_2d([2, 3, 4])       => array([[2], [3], [4]])
        X = convert_to_2d([[2], [3], [4]]) => array([[2], [3], [4]])
        X = convert_to_2d([[2, 3], [4, 5]) => array([[2, 3], [3, 4]])
    """
    if value is not None:
        value = np.atleast_1d(value)
        if len(value.shape) == 1:              
            value = value.reshape(-1, 1)
            
    return value


def smooth(x: Optional[Iterable[float]], 
           y: Iterable[float],  
           frac: float = 0.2, 
           it: int = 3) -> Float1D:
    """
    Smoothes array elements
    
    Args:                        
        x:
            optional arguments of array y(x)
            
        y:
            array to be averaged

        frac:
            fraction, see: statsmodels.org/stable/generated/statsmodels
                .nonparametric.smoothers_lowess.lowess.html
            
        it:
           number of iterations, see: statsmodels.org/stable/generated
               /statsmodels.nonparametric.smoothers_lowess.lowess.html
           
    Returns:
        smoothed array
        
    Note: 
        lowess() returns a 2D array if argument 'return_sorted' is True
        In this case, the result is returned as 
        y_smooth = lowess(y, x, return_sorted=True)[:, 1] 
    """   
    if x is None:
        y_smooth = lowess(y, range(len(y)), frac=frac, it=it, 
                          return_sorted=False)
    else:
        y_smooth = lowess(y, x, frac=frac, it=it, return_sorted=False)

    return y_smooth
