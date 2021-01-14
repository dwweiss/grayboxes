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
      2020-02-10 DWW
"""

__all__ = ['Float1D', 'Float2D', 'Float3D', 'Float4D',
           'Floats',
           'Function',
           'Int1D', 'Int2D', 'Int3D', 'Int4D',
           'None1D', 'None2D', 'None3D', 'None4D', 
           'Range', 'Ranges',
           'Scalar',
           'Str', 'Str1D', 'Str2D', 'Str3D', 'Str4D',
           ]

import numpy as np
from typing import Callable, Iterable, List, Optional, Tuple, Union

"""
    Collection of data types for type hints, see module typing
"""

# type of numpy arrays of float
Float1D = Optional[np.ndarray]
Float2D = Optional[np.ndarray]
Float3D = Optional[np.ndarray]
Float4D = Optional[np.ndarray]

# type of floats or arrays of float
Floats = Union[None, float, Iterable[float], np.ndarray]

# type of theoretical submodel for single data point
Function = Optional[Callable[..., List[float]]]

# type of numpy arrays of int
Int1D = Optional[np.ndarray]
Int2D = Optional[np.ndarray]
Int3D = Optional[np.ndarray]
Int4D = Optional[np.ndarray]

# type of numpy arrays of None 
None1D = Optional[np.ndarray]
None2D = Optional[np.ndarray]
None3D = Optional[np.ndarray]
None4D = Optional[np.ndarray]

# type of [lo,up]-ranges of float
Range = Optional[Tuple[Optional[float], Optional[float]]]
Ranges = Iterable[Range]

# type of scalar
Scalar = Union[None, int, float, str]

# type of list of str
Str = Optional[str]
Str1D = Optional[Iterable[Str]]
Str2D = Optional[Iterable[Str1D]]
Str3D = Optional[Iterable[Str2D]]
Str4D = Optional[Iterable[Str3D]]
