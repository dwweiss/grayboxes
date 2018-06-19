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
      2018-05-19 DWW
"""

import numpy as np

from Base import Base
from Model import Model
from White import White


class Forward(Base):
    """
    Predicts $y = \phi(x)$ for series of data points, x.shape: (nPoint, nInp)

    Examples:
        X = [[... ]]  input of training
        Y = [[... ]]  target of training
        x = [[... ]]  input of prediction

        def function(x, c0=1, c1=1, c2=1, c3=1, c4=1, c5=1, c6=1, c7=1):
            return 2.2 * np.array(np.sin(x[0]) + (x[1] - 1)**2)

        def method(self, x, c0=1, c1=1, c2=1, c3=1, c4=1, c5=1, c6=1, c7=1):
            return 3.3 * np.array(np.sin(x[0]) + (x[1] - 1)**2)

        # create operation on model
        op = Forward(White(function))
        or:
        op = Forward(White(method))

        # training and prediction
        best = op(X=X, Y=Y)     # train
        x, y = op(x=x)          # predict

        # compact form
        x, y = Forward(White(function))(X=X, Y=Y, x=x)


    Note:
        Forward.__call__() returns 2-tuple of 2D arrays of float (x and y)
    """

    def __init__(self, model, identifier='Forward'):
        """
        Args:
            model (Model_like):
                box type model

            identifier (str, optional):
                object identifier
        """
        super().__init__(identifier)
        self.model = model

    @property
    def model(self):
        """
        Returns:
            (Model_like):
                box type model
        """
        return self._model

    @model.setter
    def model(self, value):
        """
        Sets box type model and assigns Forward's logFile

        Args:
            value (Model_like):
                box type model
        """
        self._model = value
        if self._model is not None:
            assert issubclass(type(value), Model), \
                'invalid model type: ' + str(type(value))
            self._model.logFile = self.logFile

    def pre(self, **kwargs):
        """
        - Assigns box type model
        - Assigns training input and target (X, Y), and prediction input x
        - Trains model if (X, Y) are not None

        Args:
            kwargs (dict, optional):
                keyword arguments:

                XY (2-tuple of 2D array_like of float, optional):
                    input and target of training, this argument supersedes X, Y

                X (2D or 1D array_like of float, optional):
                    training input, shape: (nPoint, nInp) or shape: (nPoint)
                    default: self._X

                Y (2D or 1D array_like of float, optional):
                    training target, shape: (nPoint, nOut) or shape: (nPoint)
                    default: self._Y

                x (2D or 1D array_like of float, optional):
                    input to forward prediction or to sensitivity analysis
                    shape: (nPoint, nInp) or shape: (nInp)
                    default: self._x
        """
        super().pre(**kwargs)

        # trains model
        XY = kwargs.get('XY', None)
        if isinstance(XY, (list, tuple, np.ndarray)) and len(XY) > 1:
            X, Y = XY[0], XY[1]
        else:
            X, Y = kwargs.get('X', None), kwargs.get('Y', None)
        if not isinstance(self.model, White):
            if X is not None and Y is not None:
                self.best = self.model.train(X, Y, **self.kwargsDel(kwargs,
                                             ['X', 'Y']))

        x = kwargs.get('x', None)
        if type(self).__name__ in ('Minimum', 'Maximum', 'Inverse'):
            self.x = np.atleast_2d(x) if x is not None else None
        else:
            self.model.x = np.atleast_2d(x) if x is not None else None

    def task(self, **kwargs):
        """
        This task() method is only for Forward and Sensitivity.
        Minimum, Maximum and Inverse have a different implementation of task()

        Args:
            kwargs (dict, optional):
                keyword arguments passed to super.task()

        Return:
            x, y (2-tuple of 2D arrays of float):
                input and output of model prediction,
                x.shape: (nPoint, nInp) and y.shape: (nPoint, nOut)
        """
        super().task(**kwargs)

        if self.model.x is None:
            self.model.y = None
        else:
            self.model.y = np.asfarray(self.model.predict(x=self.model.x,
                                       **self.kwargsDel(kwargs, 'x')))
        return self.model.x, self.model.y

    def post(self, **kwargs):
        """
        Args:
            kwargs (dict, optional):
                keyword arguments passed to super.post()
        """
        super().post(**kwargs)

        if not self.silent:
            self.plot()

    def plot(self):
        pass


# Examples ####################################################################

if __name__ == '__main__':
    ALL = 1

    from plotArrays import plotIsoMap
    from Model import grid, cross, rand

    from LightGray import LightGray
    from MediumGray import MediumGray
    from DarkGray import DarkGray
    from Black import Black

    # function without access to 'self' attributes
    def function(x, *args):
        print('0')
        return 3.3 * np.array(np.sin(x[0]) + (x[1] - 1)**2)

    # method with access to 'self' attributes
    def method(self, x, *args):
        print('1')
        return 3.3 * np.array(np.sin(x[0]) + (x[1] - 1)**2)

    if 1 or ALL:
        s = 'Forward() with demo function build-in into Model'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        x, y = Forward(White(function))(x=grid(3, [0, 1], [0, 1]))
        plotIsoMap(x[:, 0], x[:, 1], y[:, 0])

    if 0 or ALL:
        s = 'Forward() with demo function build-in into Model'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))
        x, y = Forward(White('demo'))(x=cross(5, [1, 2], [3, 4]))
        plotIsoMap(x[:, 0], x[:, 1], y[:, 0], scatter=True)

    if 0 or ALL:
        s = "Forward, assign external function (without self-argument) to f"
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        op = Forward(White(function))
        _, y = op(x=rand(12, [2, 3], [3, 4]))
        print('x:', op.model.x, '\ny1:', op.model.y)

    if 0 or ALL:
        s = "Forward, assign method (with 'self'-argument) to f"
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        op = Forward(White(function))
        _, y = op(x=[[2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        print('x:', op.model.x, '\ny1:', op.model.y)
