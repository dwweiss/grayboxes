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
      2020-03-05 DWW
"""

import sys
from typing import Any, Dict, Optional, Union

from grayboxes.boxmodel import BoxModel
from grayboxes.datatype import Float2D, Function
from grayboxes.metrics import init_metrics
try:
    from grayboxes.neuraltf import Neural as NeuralTf
except ImportError:
    print('!!! Module neuraltf not imported')
try:
    from grayboxes.neuralnl import Neural as NeuralNl
except ImportError:
    print('!!! Module neuralnl not imported')
try:
    from grayboxes.neuralto import Neural as NeuralTo
except ImportError:
    print('!!! Module neuralto not imported')
try:
    from grayboxes.splines import Splines
except ImportError:
    print('!!! Module splines not imported')


class Black(BoxModel):
    """
    Black box model y = beta(x, w) with w = arg min ||beta(X, w) - Y||_2
    where Y(X) is the training data and x is arbitrary input

        - Neural network is employed if 'splines' not in kwargs, 
          Splines are employed otherwise

        - Weights of best model training trials are saved as 
          'self._empirical._weights'

    Example:
        X = np.linspace(0., 1., 200).reshape(-1, 1)      # train input 
        x = np.linspace(-1., 2., 100).reshape(-1, 1)      # test input 
        x = X * 1.9
        Y = X**2.

        # black box, neural network, compact variant:
        y = Black()(XY=(X, Y), neurons=[8, 6], x=x)

        # black box, neural network, expanded variant:
        phi = Black()                        # create instance of Black
        metrics = phi(X=X, Y=Y, neurons=[8, 6])              # training
        Y_prd = phi(x=X)               # prediction with training input
        y_prd = phi(x=x)                   # prediction with test input
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
        self._empirical: Optional[Union[NeuralTf, NeuralNl, 
                                        NeuralTo, Splines]] = None
        
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
                shape (n_point,) is tolerated

            Y:
                training target, shape: (n_point, n_out)
                shape (n_point,) is tolerated

        Kwargs:
            backend (str):
                identifier of backend:
                    'neurolab'
                    'tensorflow'
                    'torch'
                default: 'tensorflow'
                    
            neurons (int, 1D array of int, or None):
                number of neurons in hidden layer(s) of neural network

            splines (int, 1D array of float, or None):
                not specified yet
                
            trainer (str, list of str, or None):
                optimizer of network, see BruteForce.train()
               
            ... additional training options, see BruteForce.train()

        Returns:
            metrics of best training trial
                see BoxModel.train()
        """        
        self.metrics = init_metrics()
        self.ready = False
        
        if X is not None and Y is not None:
            self.set_XY(X, Y)
    
            neurons = kwargs.get('neurons', None)
            splines = kwargs.get('splines', None) if neurons is None else None
            if 'splines' not in sys.modules:
                splines = None
    
            if splines is None:            
                backend = kwargs.get('backend', 'keras').lower()
                if backend in ('tensorflow', 'tf', 'keras',):
                    backend = 'tensorflow'
                    assert 'grayboxes.neuraltf' in sys.modules
                elif backend in ('nl', 'neurolab',):
                    backend = 'neurolab'
                    assert 'grayboxes.neuralnl' in sys.modules
                elif backend in ('torch', 'pytorch',):
                    backend = 'torch'
                    assert 'grayboxes.neuralto' in sys.modules
                else:
                    backend = 'tensorflow'
                self.write('+++ backend: ' + str(backend))
            else:
                backend = None

            if splines is not None:
                empirical = Splines()
            else:                
                if backend == 'tensorflow':
                    empirical = NeuralTf(self.f)                    
                elif backend == 'torch':
                    empirical = NeuralTo(self.f)                    
                elif backend == 'neurolab':
                    empirical = NeuralNl(self.f)                    
                else:
                    assert 0, str(backend)

            self.write('+++ train')
    
            if self._empirical is not None:
                del self._empirical
            self._empirical = empirical
            self._empirical.silent = self.silent
    
            self.metrics = self._empirical.train(self.X, self.Y, 
                                                 **self.kwargs_del(kwargs,'f'))
            self.ready = self._empirical.ready
            self.metrics['ready'] = self.ready

        return self.metrics

    def predict(self, x: Float2D, **kwargs: Any) -> Float2D:
        """
        Executes box model, stores input as self.x and output as self.y

        Args:
            x:
                prediction input, shape: (n_point, n_inp)
                shape: (n_inp,) is tolerated

        Kwargs:
            Keyword arguments of emprical model

        Returns:
            prediction output, shape: (n_point, n_out)
            or
            None if self._empirical is None or not self.ready
        """
        if not self.ready or self._empirical is None:
            self.y = None
            return self.y

        self.x = x                            # setter ensuring 2D array
        assert self._n_inp == self.x.shape[1], str((self._n_inp, self.x.shape))

        self.y = self._empirical.predict(self.x, **kwargs)
        
        return self.y
