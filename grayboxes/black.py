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
      2019-11-20 DWW
"""

import sys
from typing import Any, Dict, Optional, Union

from grayboxes.base import Float1D, Float2D
from grayboxes.boxmodel import BoxModel
from grayboxes.neural import Neural    # , RadialBasis
try:
    from grayboxes.splines import Splines
except ImportError:
    print('!!! Module splines not imported')


class Black(BoxModel):
    """
    Black box model y = beta(x, w) with w = arg min ||beta(x, w) - Y||_2

        - Neural network is employed if kwargs contains 'neurons'

        - Splines are employed if kwargs contains 'splines'

        - Radial basis functions are employed if kwargs contains 'centers'

        - Weights of best model training trials are saved as 
          'self._empirical._weights'

    Example:
        X = np.atleast_2d(np.linspace(0.0, 1.0, 20)).T
        x = X * 2
        Y = X**2

        # black box, neural network, compact variant:
        y = Black()(XY=(X, Y), neurons=[2, 3], x=x)

        # black box, neural network, expanded variant:
        model = Black()                       # create instance of Black
        metrics = model(X=X, Y=Y, neurons=[2, 3])             # training
        y_trn = model(x=X)              # prediction with training input
        y_tst = model(x=x)                  # prediction with test input
    """

    def __init__(self, identifier: str = 'Black') -> None:
        """
        Args:
            identifier:
                Unique object identifier
        """
        super().__init__(f=None, identifier=identifier)
        self._empirical:Optional[Union[Neural, ]] = None 
                                       # Instance of Neural, Splines etc
        self.metrics: Dict[str, Any] = self.init_metrics()
                                          # measure of model performance

    @property
    def silent(self) -> bool:
        return self._silent

    @silent.setter
    def silent(self, value: bool) -> None:
        self._silent = value
        if self._empirical is not None:
            self._empirical._silent = value

    def train(self, X: Float2D, Y: Float2D, **kwargs: Any) -> Dict[str, Any]:
        """
        Trains model, stores X and Y as self.X and self.Y, and stores
        performance of best training trial as self.metrics

        Args:
            X:
                training input, shape: (n_point, n_inp)

            Y:
                training target, shape: (n_point, n_out)

        Kwargs:
            neurons (int or 1D array of int):
                number of neurons in hidden layer(s) of neural network

            splines (int or 1D array of float):
                not specified yet
               
            centers (int or 1D array of float)
                number of centers in hidden layer 
                or
                array of centers

            ... additional training options, see Neural.train()

        Returns:
            results:
                see BoxModel.train()
            or
                None
        """
        if X is None or Y is None:
            self.metrics = self.init_metrics()
            return self.metrics

        self.set_XY(X, Y)

        neurons = kwargs.get('neurons', None)
        splines = kwargs.get('splines', None) if neurons is None else None
        centers = kwargs.get('centers', None)
        if neurons:
            empirical = Neural()
        elif splines and  'splines' in sys.modules:
            empirical = Splines()
        elif centers:
            assert 0
#            empirical = RadialBasis()
        else:
            empirical = Neural()

        self.write('+++ train')

        if self._empirical is not None:
            del self._empirical
        self._empirical = empirical
        self._empirical.silent = self.silent

        self.metrics = self._empirical.train(self.X, self.Y, **kwargs)
        self.ready = self._empirical.ready

        return self.metrics

    def predict(self, x: Float2D, **kwargs: Any) -> Float2D:
        """
        Executes box model, stores input as self.x and output as self.y

        Args:
            x:
                prediction input, shape: (n_point, n_inp)
                shape: (n_inp,) is tolerated

        Kwargs:
            Keyword arguments

        Returns:
            prediction output, shape: (n_point, n_out)
        """
        assert self.ready, str(self.ready)
        assert self._empirical is not None

        self.x = x                # self.x is a setter ensuring 2D array
        assert self._n_inp == self.x.shape[1], \
            str((self._n_inp, self.x.shape))

        self.y = self._empirical.predict(self.x, **kwargs)
        return self.y
