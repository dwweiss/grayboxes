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

import numpy as np
from typing import Any, Callable, Dict, Optional

from grayboxes.black import Black
from grayboxes.boxmodel import BoxModel


class DarkGray(BoxModel):
    """
    Dark gray box model

    Example:
        External function or method is assigned to self.f():
            def f(self, x, *args):
                c0, c1, c3 = args if len(args) > 0 else (1, 1, 1)
                y0 = c0 * x[0]*x[0] + c1 * x[1]
                y1 = x[1] * c3
                return [y0, y1]

            X = [[..], [..], ..]
            Y = [[..], [..], ..]
            x = [[..], [..], ..]

            # expanded form:
            model = DarkGray(f)
            best = model.train(X, Y, neurons=[5])
            y = model.predict(x)

            # compact form:
            y = DarkGray(f)(X=X, Y=Y, x=x, neurons=[5])
    """

    def __init__(self, f: Callable, identifier: str='DarkGray') -> None:
        """
        Args:
            f:
                theoretical submodel f(self, x) or f(x) for single data point

            identifier:
                object identifier
        """
        super().__init__(identifier=identifier, f=f)
        self._black = Black()

    @property
    def silent(self) -> bool:
        return self._silent

    @silent.setter
    def silent(self, value: bool) -> None:
        self._silent = value
        self._black._silent = value

    def train(self, X: np.ndarray, Y: np.ndarray, **kwargs: Any) \
            -> Optional[Dict[str, Any]]:
        """
        Args:
            X (2D or 1D array of float):
                training input, shape: (nPoint, nInp) or (nInp,)

            Y (2D or 1D array of float):
                training target, shape: (nPoint, nOut) or (nOut,)

        Kwargs:
            Keyword arguments to be passed to train () of this object
            and of black box model

        Returns:
            best result, see BoxModel.train()
            or
            None if X and Y are None
        """
        if X is None or Y is None:
            return None

        self.X, self.Y = X, Y
        y = BoxModel.predict(self, self.X, **self.kwargs_del(kwargs, 'x'))
        self.best = self._black.train(np.c_[self.X, y], y-self.Y, **kwargs)

        return self.best

    def predict(self, x: np.ndarray, **kwargs: Any) -> Optional[np.ndarray]:
        """
        Executes box model, stores input x as self.x and output as self.y

        Args:
            x (2D or 1D array of float):
                prediction input, shape: (nPoint, nInp) or (nInp,)

        Kwargs:
            Keyword arguments to be passed to predict() of this object
            and of black box model

        Returns:
            (2D array of float):
                prediction output, shape: (nPoint, nOut)
            or
            None if x is None
        """
        if x is None:
            return None
        assert self._black is not None and self._black.ready

        self.x = x
        self._y = BoxModel.predict(self, x, **kwargs)
        self._y -= self._black.predict(np.c_[x, self._y], **kwargs)

        return self.y
