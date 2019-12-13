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
      2019-12-09 DWW
"""

from typing import Any, Union

from grayboxes.base import Float2D, Float1D
from grayboxes.minimum import Minimum


class Maximum(Minimum):
    """
    Maximizes objective function
    """

    def objective(self, x: Union[Float2D, Float1D], **kwargs: Any) -> float:
        """
        Objective function for maximization

        Args:
            x:
                input of multiple or single data points, 
                shape: (n_point, n_inp)
                
                Note: shape (n_inp, ) is tolerated

        Kwargs:
            Keyword arguments to be passed to objective() of parent

        Returns:
            Optimization objective to be maximized
        """
        return (-1) * Minimum.objective(self, x, **kwargs)
