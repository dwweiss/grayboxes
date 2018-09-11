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
      2018-09-11 DWW
"""

from typing import Any, Callable
import numpy as np

from grayboxes.boxmodel import BoxModel


class White(BoxModel):
    """
    White box model y=f(x)
    """

    def __init__(self, f: Callable, identifier: str='White') -> None:
        """
        Args:
            f:
                Theoretical submodel f(self, x) or f(x) for single data point

            identifier:
                Unique object identifier
        """
        super().__init__(f=f, identifier=identifier)

    def train(self, X: np.ndarray, Y: np.ndarray, **kwargs: Any) -> None:
        """
        Sets self.ready to True, no further actions

        Args:
            X:
                2D array training input, not used

            Y:
                2D training target, not used

        Kwargs:
            Keyword arguments, not used
        """
        self.ready = True
        return None
