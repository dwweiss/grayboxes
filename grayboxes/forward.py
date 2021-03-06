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
      2020-11-26 DWW
"""

import numpy as np
from typing import Any, Optional, Tuple

try:
    from grayboxes.base import Base
    from grayboxes.boxmodel import BoxModel
    from grayboxes.datatype import Float2D
    from grayboxes.white import White
except:
    from base import Base
    from boxmodel import BoxModel
    from datatype import Float2D
    from white import White


class Forward(Base):
    """
    Predicts $y = \phi(x)$ for array of points, shape: (n_point, n_inp)

    Examples:
        X = [[...], [...], ...]  input of training
        Y = [[...], [...], ...]  target of training
        x = [[...], [...], ...]  input of prediction

        def function(x, c0=1, c1=1):
            return c0 * np.array(np.sin(c1*x[0]) + (x[1] - 1)**2)

        def method(self, x, c0=1, c1=1):
            return c0 * np.array(np.sin(c1*x[0]) + (x[1] - 1)**2)

        # create operation on model
        operation = Forward(White(function))
        or:
        operation = Forward(White(method))

        # training and prediction
        metrics = operation(X=X, Y=Y, trainer='lm', goal=1e-5)  # train
        x, y = operation(x=x)                                   # predict

        # alternative: compact form
        x, y = Forward(White(function))(X=X, Y=Y, x=x, trainer='lm', goal=1e-5)


    Note:
        - Forward.__call__() returns 2-tuple of 2D arrays of float

        - Forward has no own self._x or self._y attribute and employs
          therefore model.x and model.y for storing input and output

    """

    def __init__(self, model: BoxModel, identifier: str = 'Forward') -> None:
        """
        Args:
            model:
                Box type model

            identifier:
                Unique object identifier
        """
        super().__init__(identifier)
        self._model: BoxModel = model

    @property
    def model(self) -> Optional[BoxModel]:
        """
        Returns:
            Box type model
        """
        return self._model

    @model.setter
    def model(self, value: Optional[BoxModel]) -> None:
        """
        Sets box type model

        Args:
            value:
                Box type model
        """
        self._model = value
        if self._model is not None:
            assert issubclass(type(value), BoxModel), \
                'invalid model type: ' + str(type(value))

    def pre(self, **kwargs: Any) -> bool:
        """
        - Assigns box type model
        - Assigns training input and target (X, Y)
        - Assigns prediction input x
        - Trains model is not instance of White and 
          if X is not None and Y is not None

        Kwargs:
            XY (2-tuple of 2D array of float):
                input and target of training, 
                shapes: (n_point, n_inp) and (n_point, n_out)
                Note: this argument supersedes X, Y

            X (2D array of float):
                training input, 
                shape: (n_point, n_inp)
                default: self._X

            Y (2D array of float):
                training target, 
                shape: (n_point, n_out)
                default: self._Y

            x (2D or 1D array of float):
                input to forward prediction or to sensitivity analysis
                shape: (n_point, n_inp) or (n_inp,)
                default: self._x
                
        Returns:
            True on success of pre() of parent class
        """
        ok = super().pre(**kwargs)

        # trains model
        XY = kwargs.get('XY', None)
        if isinstance(XY, (list, tuple, np.ndarray)) and len(XY) > 1:
            X, Y = XY[0], XY[1]
        else:
            X, Y = kwargs.get('X', None), kwargs.get('Y', None)
        if not isinstance(self.model, White):
            if X is not None and Y is not None:
                self.metrics = self.model.train(X, Y, **self.kwargs_del(kwargs,
                                                ['X', 'Y']))

        x = kwargs.get('x', None)
        if type(self).__name__ in ('Minimum', 'Maximum', 'Inverse'):
            self.x = np.atleast_2d(x) if x is not None else None
        else:
            self.model._x = np.atleast_2d(x) if x is not None else None
            
        return ok

    def task(self, **kwargs: Any) -> Tuple[Float2D, Float2D]:
        """
        This task() method is only for Forward and Sensitivity. Minimum,
        Maximum and Inverse have different implementations of task()

        Kwargs:
            Keyword arguments passed to super.task() and to
            model.predict()

        Return:
            x, y (2-tuple of 2D arrays of float):
                input and output of model prediction,
                x.shape: (n_point, n_inp) and y.shape: (n_point, n_out)
        """
        super().task(**kwargs)

        if self.model.x is None:
            self.model._y = None
        else:
            self.model.y = self.model.predict(x=self.model.x,
                                              **self.kwargs_del(kwargs, 'x'))
        return self.model.x, self.model.y

    def post(self, **kwargs: Any) -> bool:
        """
        Kwargs:
            Keyword arguments passed to super.post()
            
        Returns:
            True on success of post() of parent class
        """
        ok = super().post(**kwargs)

        if not self.silent:
            self.plot()
            
        return ok

    def plot(self) -> None:
        pass
