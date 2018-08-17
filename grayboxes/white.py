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
      2018-08-17 DWW
"""

from grayboxes.boxmodel import BoxModel


class White(BoxModel):
    """
    White box model y=f(x)
    """

    def __init__(self, f, identifier='White'):
        """
        Args:
            f (method or function):
                theoretical submodel f(self, x) or f(x) for single data point

            identifier (str, optional):
                object identifier
        """
        super().__init__(f=f, identifier=identifier)

    def train(self, X, Y, **kwargs):
        """
        Sets self.ready to True, no further actions

        Args:
            X (2D array_like of float):
                training input, not used

            Y (2D array_like of float):
                training target, not used

            kwargs (dict, optional):
                keyword arguments, not used

        Returns:
            None
        """
        self.ready = True
        return None
