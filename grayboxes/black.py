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
      2018-08-15 DWW
"""

import numpy as np
from grayboxes.model import Model
from grayboxes.neural import Neural


class Black(Model):
    """
    Black box model y = F^*(x, w), w = train(X, Y, f(x))

        - Neural network is employed if argument 'neurons' is passed by kwargs
          Best neural network of all trials is saved as 'self._empirical._net'

        - Splines are employed if the argument 'splines' is passed by kwargs

    Example:
        X = np.linspace(0.0, 1.0, 20)
        x = X * 2
        Y = X**2

        # black box, neural network, expanded:
        model = Black()
        model(X=X, Y=Y, neurons=[2, 3])
        yTrn = model(x=X)
        yTst = model(x=x)

        # black box, neural network, compact:
        y = Black()(XY=(X, Y), neurons=[2, 3], x=x)
        
    Todo:
        Implement multivariant splines
    """

    def __init__(self, identifier='Black'):
        """
        Args:
            identifier (str, optional):
                object identifier
        """
        super().__init__(f=None, identifier=identifier)
        self._empirical = None         # holds instance of Neural, splines etc.

    @property
    def silent(self):
        return self._silent

    @silent.setter
    def silent(self, value):
        self._silent = value
        if self._empirical is not None:
            self._empirical._silent = value

    def train(self, X, Y, **kwargs):
        """
        Trains model, stores X and Y as self.X and self.Y, and stores result
        of best training trial as self.best

        Args:
            X (2D or 1D array_like of float):
                training input, shape: (nPoint, nInp) or shape: (nPoint,)

            Y (2D or 1D array_like of float):
                training target, shape: (nPoint, nOut) or shape: (nPoint,)

            kwargs (dict, optional):
                keyword arguments:

                neurons (int or 1D array_like of int):
                    number of neurons in the hidden layers of neural network

                splines (1D array_like of float):
                    not specified yet

                ... additional training options of network, see Neural.train()

        Returns:
            see Model.train()
        """
        if X is None or Y is None:
            return None

        self.X, self.Y = np.atleast_2d(X), np.atleast_2d(Y)
        if self._X.shape[0] == 1:
            self._X = self._X.T
        if self._Y.shape[0] == 1:
            self._Y = self._Y.T
        assert self.X.shape[0] == self.Y.shape[0], \
            str(self.X.shape) + str(self.Y.shape)
        assert self.X.shape[0] > 2, str(self.X.shape)

        neurons = kwargs.get('neurons', None)
        splines = kwargs.get('splines', None) if neurons is None else None

        if neurons is not None:
            self.write('+++ train neural, hidden neurons:', neurons)

            if self._empirical is not None:
                del self._empirical
            self._empirical = Neural()
            self._empirical.silent = self.silent

            self.best = self._empirical.train(self.X, self.Y, **kwargs)
            self.ready = self._empirical.ready

        elif splines is not None:
            assert self.f is None

            # TODO ...

            self.best = None
            self.ready = False
            assert 0, 'splines not implemented'

        else:
            assert self.f is None
            self.best = None
            self.ready = False
            self._empirical = None
            assert 0, 'unknown empirical method'

        return self.best

    def predict(self, x, **kwargs):
        """
        Executes Model, stores input x as self.x and output as self.y

        Args:
            x (2D or 1D array_like of float):
                prediction input, shape: (nPoint, nInp) or shape: (nInp,)

            kwargs (dict, optional):
                keyword arguments

        Returns:
            (2D array of float):
                prediction output, shape: (nPoint, nOut)
        """
        assert self.ready, str(self.ready)
        assert self._empirical is not None

        self.x = x
        self.y = self._empirical.predict(self.x, **kwargs)
        return self.y
