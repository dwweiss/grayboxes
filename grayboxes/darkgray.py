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

import numpy as np
from typing import Any, Dict

from grayboxes.black import Black
from grayboxes.boxmodel import BoxModel
from grayboxes.datatypes import Float2D, Function
from grayboxes.metrics import init_metrics


class DarkGray(BoxModel):
    """
    Dark gray box model

    Example:
        External function or method is assigned to self.f():
            def f(self, x, *c):
                c0, c1, c3 = args if len(c) >= 3 else (1., 1., 1.)
                y0 = c0 * x[0]*x[0] + c1 * x[1]
                y1 = x[1] * c3
                return [y0, y1]

            X = [[..], [..], ..]
            Y = [[..], [..], ..]
            x = [[..], [..], ..]

            # expanded form:
            model = DarkGray(f)
            metrics = model.train(X, Y, neurons=[5])
            y = model.predict(x)

            # compact form:
            y = DarkGray(f)(X=X, Y=Y, neurons=[5], x=x)
    """

    def __init__(self, f: Function, identifier: str = 'DarkGray') -> None:
        """
        Args:
            f:
                theoretical submodel f(self, x) or f(x) for single point

            identifier:
                object identifier
        """
        super().__init__(identifier=identifier, f=f)
        self._empirical = Black()

    @property
    def silent(self) -> bool:
        return self._silent

    @silent.setter
    def silent(self, value: bool) -> None:
        self._silent = value
        self._empirical._silent = value

    def train(self, X: Float2D, Y: Float2D, **kwargs: Any) -> Dict[str, Any]:
        """
        Args:
            X:
                training input, shape: (n_point, n_inp)

            Y:
                training target, shape: (n_point, n_out)

        Kwargs:
            Keyword arguments to be passed to train() of this object
            and of black box model

        Returns:
            metrics of training, see BoxModel.train()
        """
        if X is None or Y is None or self._empirical is None:
            self.ready = False
            return init_metrics()

        self.set_XY(X, Y)
        
        # self.ready must be True to avoid that self.predict() returns None       
        self.ready = True  
        y = BoxModel.predict(self, self.X, **self.kwargs_del(kwargs, 'x'))
        
        X_emp = np.c_[self.X, y]
        Y_emp = y - self.Y
        self.metrics = self._empirical.train(X_emp, Y_emp, **kwargs)
        self.ready = self._empirical.ready

        return self.metrics

    def predict(self, x: Float2D, *c: float, **kwargs: Any) -> Float2D:
        """
        Executes box model, stores input x as self.x and output as self.y

        Args:
            x:
                prediction input, shape: (n_point, n_inp)
                shape: (n_inp,) is tolerated

            c:
                weigths as positional arguments to be passed to 
                theoretical submodel f()

        Kwargs:
            Keyword arguments to be passed to predict() of this object
            and of black box model

        Returns:
            prediction output, shape: (n_point, n_out)
            or
            None if x is None or model is not ready
        """
        if x is None or not self.ready:
            self.y = None
            return self.y

        self.x = x                              # ensures valid 2D array
        y = np.asfarray(BoxModel.predict(self, x, *c, **kwargs))
        
        if y is not None:
            y_delta = self._empirical.predict(np.c_[self.x, self._y], **kwargs)
            if y_delta is not None:
                y = np.asfarray(y) - np.asfarray(y_delta)
        self.y = y                              # ensures valid 2D array
        
        return self.y
