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

__all__ = ['Float1D', 'Float2D', 'Float3D', 
           'Function',
           'None1D', 'None2D', 
           'Range', 
           'Str1D',
           'Str1', 'Str2', 'Str3', 'Str4',
           ]

import numpy as np
from typing import Callable, Iterable, List, Optional, Tuple

# type of arrays of float
Float1D = Optional[np.ndarray]
Float2D = Optional[np.ndarray]
Float3D = Optional[np.ndarray]
Float4D = Optional[np.ndarray]

# type of theoretical submodel for single data point
Function = Optional[Callable[..., List[float]]]

# type of arrays of None 
None1D = Optional[np.ndarray]
None2D = Optional[np.ndarray]
None3D = Optional[np.ndarray]
None4D = Optional[np.ndarray]

# type of range of float
Range = Optional[Tuple[Optional[float], Optional[float]]]

# type of list of str
Str1D = Optional[Iterable[str]]
Str2D = Optional[Iterable[Iterable[str]]]

# type of tuple of str
Str1 = Optional[str]
Str2 = Optional[Tuple[Str1, Str1]]
Str3 = Optional[Tuple[Str1, Str1, Str1]]
Str4 = Optional[Tuple[Str1, Str1, Str1, Str1]]
