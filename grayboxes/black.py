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
      2019-12-12 DWW
"""

import sys
from typing import Any, Dict, Optional, Union

from grayboxes.base import Float2D, Function
from grayboxes.boxmodel import BoxModel
from grayboxes.neural import Neural
try:
    from grayboxes.splines import Splines
except ImportError:
    print('!!! Module splines not imported')


class Black(BoxModel):
    """
    Black box model y = beta(x, w) with w = arg min ||beta(x, w) - Y||_2

        - Neural network is employed if kwargs contains 'neurons'

        - Splines are employed if kwargs contains 'splines'

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

    def __init__(self, f: Function = None, identifier: str = 'Black') -> None:
        """
        Args:
            f:
                Dummy parameter for compatibility with the other 
                children of class BoxModel where 'f' is the theoretical 
                submodel for a single data point

            identifier:
                Unique object identifier
        """
        super().__init__(f=None, identifier=identifier)
        self._empirical: Optional[Union[Neural, 
                                        """ Splines """]] = None 

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
               
            ... additional training options, see Neural.train()

        Returns:
            metrics of best training trial
                see BoxModel.train()
        """
        self.metrics = self.init_metrics()
        self.ready = False
        
        if X is not None and Y is not None:
            self.set_XY(X, Y)
    
            neurons = kwargs.get('neurons', None)
            splines = kwargs.get('splines', None) if neurons is None else None
            if neurons:
                empirical = Neural()
            elif splines and  'splines' in sys.modules:
                empirical = Splines()
            else:
                empirical = Neural(f=None)
    
            self.write('+++ train')
    
            if self._empirical is not None:
                del self._empirical
            self._empirical = empirical
            self._empirical.silent = self.silent
    
            self.metrics = self._empirical.train(self.X, self.Y, 
                                                 **self.kwargs_del(kwargs,'f'))
            
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
            or
            None if model is not trained
        """
        if not self.ready or self._empirical is None:
            self.y = None
            return self.y

        self.x = x                            # setter ensuring 2D array
        assert self._n_inp == self.x.shape[1], str((self._n_inp, self.x.shape))

        self.y = self._empirical.predict(self.x, **kwargs)
        
        return self.y
