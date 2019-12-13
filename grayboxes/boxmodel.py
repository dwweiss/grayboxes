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

import inspect
from collections import OrderedDict
import numpy as np
from pandas import DataFrame
from typing import (Any, Dict, Optional, List, Sequence, Tuple, Union)

from grayboxes.array import convert_to_2d
from grayboxes.base import Base
from grayboxes.base import Float1D, Float2D, Function, Str1D
from grayboxes.parallel import communicator, predict_scatter


class BoxModel(Base):
    """
    Parent class of White, LightGray, MediumGray, DarkGray and Black

    - Inputs and outputs are 2D arrays
    - First index is data point index
    - If X, Y or x are passed as 1D array_like, they are transformed to
          self.X = np.atleast_2d(X).T and 
          self.Y = np.atleast_2d(Y).T

    - Upper case 'X' is 2D training input and 'Y' is 2D training target
    - Lower case 'x' is 2D prediction input and 'y' is 2D pred. output
    - XY = (X, Y, x_keys,y_keys) is union of X and Y, optional with keys
    """

    def __init__(self, f: Function, identifier: str = 'BoxModel') -> None:
        """
        Args:
            f:
                Theoretical submodel for single data point 
                    f(self, x: Sequence[float], *c: float, **kwargs: Any) 
                or
                    f(x: Sequence[float], *c: float, **kwargs: Any) 

            identifier:
                Unique object identifier
        """
        super().__init__(identifier=identifier)
        
        self.f: Function = f 
                                 # theoretical submodel if not black box
        self._X: Float2D = None                         # training input
        self._Y: Float2D = None                        # training target
        self._x: Union[Float1D, Float2D] = None       # prediction input
        self._y: Union[Float1D, Float2D] = None      # prediction output 
        self._x_keys: Str1D = None           # x-keys for data selection
        self._y_keys: Str1D = None           # y-keys for data selection
        self._metrics: Dict[str, Any] = self.init_metrics()    # metrics
        self._weights: Float1D = None    # weights of empirical submodel
        self._n_inp: int = -1                         # number of inputs

    def init_metrics(self, keys: Optional[Union[str, Sequence[str]]] = None,
                     values: Optional[Union[Any, Sequence[Any]]] = None) \
            -> Dict[str, Any]:
        """
        Sets default values to metrics describing model performance

        Args:
            keys:
                list of keys to be updated or added

            values:
                list of values to be updated or added

        Returns:
            default settings for 
                - metrics of best training trial 
                - model evaluation
            see description of metrics in BoxModel.train()
        """
        keys, values  = np.atleast_1d(keys), np.atleast_1d(values)
        assert len(keys) == len(values), str((keys, values))
        
        metrics = {'abs': np.inf,
                   'epochs': -1,                   
                   'evaluations': -1, 
                   'i_abs': -1,
                   'i_trial': -1, 
                   'iterations': -1, 
                   'L2': np.inf, 
                   'trainer': None
                   }
        
        if keys is not None and values is not None:
            for key, value in zip(np.atleast_1d(keys), np.atleast_1d(values)):
                metrics[key] = value
                
        return metrics

    @property
    def f(self) -> Function:
        """
        Returns:
            Theoretical submodel f(self, x, *c, **kwargs)
        """
        return self._f

    @f.setter
    def f(self, value: Union[Function, str]) -> None:
        """
        Args:
            value:
                theoretical submodel as method 
                    f(self, x, *c, **kwargs) or as external function
                    f(x, *c, **kwargs) for single data point
                or
                identifier of demo function. If value is 'demo' or 
                   'rosen', then f_demo() is assigned to self.f
        """
        if not isinstance(value, str):
            f = value
        else:
            f = self.f_demo if value.startswith(('demo', 'rosen')) else None
        if f is not None:
            first_arg = list(inspect.signature(f).parameters.keys())[0]
            if first_arg == 'self':
                f = f.__get__(self, self.__class__)
        self._f = f

    def f_demo(self, x: Optional[Sequence[float]], *c: float, 
               **kwargs: Any) -> List[float]:
        """
        Rosenbrook function as demo function f(self, x) for single data 
        point [y0] = f([x0, x1]) with two coefficients [c0, c1]

        Minimum at f(c0, c0**2) = f(1, 1) = 0 if c0=1 and c1=100
        recommended plot ranges: [(-2, 2), (-1, 3)]

        Args:
            x:
                input, shape: (n_inp,)
                or 
                None if initial tuning parameters should be returned

            c:
                tuning parameters as positional arguments, 
                shape: (n_tun, )

        Kwargs:
            key word arguments

        Returns:
            output array, shape: (n_out,)
            or
            initial tuning parameter array if x is None, shape: (n_tun,)
        """
        c0, c1 = 1., 100.
        if x is None:
            return c0, c1
        if len(c) == 2:
            c0, c1 = c
            
        y0 = (c0 - x[0])**2 + c1 * (x[1] - x[0]**2)**2
        
        return [y0]

    def set_XY(self, X: Float2D, Y: Float2D, 
               correct_xy_shape: bool = True) -> None:
        """
        Sets X and Y array. Saves X.shape[1] as self._n_inp
        
        Args:
            X:
                X array of training input, shape: (n_point, n_inp)

            Y:
                Y array of training target, shape: (n_point, n_out)

            correct_xy_shape:
                if True, correct shape of X and Y so that X.shape[0] > 1
                and Y.shape[0] > 1

        Returns:
            None
        """
        # setters ensuring valid 2D shape
        self.X = X if X is not None else self.X  
        self.Y = Y if Y is not None else self.Y
        
        if correct_xy_shape:
            if self._X.shape[0] == 1:
                self._X = self._X.T
            if self._Y.shape[0] == 1:
                self._Y = self._Y.T

        self._n_inp = self.X.shape[1]

        assert self.X.shape[0] == self.Y.shape[0], str((self.X.shape,
                                                        self.Y.shape))
        if correct_xy_shape:
            assert self.X.shape[0] >= 2, str(self.X.shape)

    @property
    def X(self) -> Float2D:
        """
        Returns:
            X array of training input, shape: (n_point, n_inp)
        """
        return self._X

    @X.setter
    def X(self, value: Float2D) -> None:
        """
        Args:
            value:
                array of training input, shape: (n_point, n_inp)
                shape: (n_inp,) is tolerated
        """
        self._X = convert_to_2d(value)

    @property
    def Y(self) -> Float2D:
        """
        Returns:
            array of training target, shape: (n_point, n_out)
        """
        return self._Y

    @Y.setter
    def Y(self, value: Float2D) -> None:
        """
        Args:
            value:
                array of training target, shape: (n_point, n_out)
                shape: (n_out,) is tolerated
        """
        self._Y = convert_to_2d(value)

    @property
    def x(self) -> Float2D:
        """
        Returns:
            array of prediction input, shape: (n_point, n_inp)
        """
        return self._x

    @x.setter
    def x(self, value: Float2D) -> None:
        """
        Args:
            value:
                array of prediction input, shape: (n_point, n_inp)
                shape: (n_inp,) is tolerated
        """
        if value is None:
            self._x = value
        else:
            self._x = np.atleast_2d(value)

    @property
    def y(self) -> Float2D:
        """
        Returns:
            y array of prediction output, shape: (n_point, n_out)
        """
        return self._y

    @y.setter
    def y(self, value: Float2D) -> None:
        """
        Args:
            value:
                y array of prediction output, shape: (n_point, n_out)
                shape: (n_out,) is tolerated
        """
        if value is None:
            self._y = value
        else:
            self._y = np.atleast_2d(value)

    @property
    def XY(self) -> Tuple[Float2D, Float2D, Str1D, Str1D]:
        """
        Returns:
            X:
                training input, shape: (n_point, n_inp)

            Y:
                training target, shape: (n_point, n_out)

            x_keys:
                list of column keys for data selection
                use self._x_keys keys if x_keys is None,
                default: ['x0', 'x1', ... ]

            y_keys:
                list of column keys for data selection
                use self._y_keys keys if y_keys is None,
                default: ['y0', 'y1', ... ]
        """
        return self._X, self._X, self._x_keys, self._y_keys

    @XY.setter
    def XY(self, value: Union[Tuple[Float2D, Float2D, Str1D, Str1D],
                              Tuple[Float2D, Float2D]]) -> None:
        """
        Args:
            value (4-tuple or 2-tuple of arrays):
                X:
                    training input, shape: (n_point, n_inp)

                Y:
                    training target, shape: (n_point, n_out)

                x_keys:
                    list of column keys for data selection
                    use self._x_keys keys if x_keys is None,
                    default: ['x0', 'x1', ... ]

                y_keys:
                    list of column keys for data selection
                    use self._y_keys keys if y_keys is None,
                    default: ['y0', 'y1', ... ]

        Side effects:
            self._X, self._Y, self._x_keys, self._y_keys will be overwritten
        """
        if len(value) == 2:
            X, Y, x_keys, y_keys = [value[0], value[1], None, None]
        elif len(value) == 4:
            X, Y, x_keys, y_keys = value
        else:
            X, Y, x_keys, y_keys = [None, None, None, None]
            self.write('??? invalid value: ' + str(value))
        
        assert X is not None and Y is not None, str(X is not None)

        self.X = X                      # setter ensuring valid 2D shape
        self.Y = Y                      # setter ensuring valid 2D shape

        assert self._X.shape[0] == self._Y.shape[0], \
            str(self._X.shape) + str(self._Y.shape)

        if x_keys is None:
            self._x_keys = ['x' + str(i) for i in range(self._X.shape[1])]
        else:
            self._x_keys = np.atleast_1d(x_keys).tolist()
        if y_keys is None:
            self._y_keys = ['y' + str(i) for i in range(self._Y.shape[1])]
        else:
            self._y_keys = np.atleast_1d(y_keys).tolist()
            
        assert not any(x in self._y_keys for x in self._x_keys), \
            str((self._x_keys, self._y_keys))

    def XY_to_frame(self, X: Float2D = None, 
                    Y: Float2D = None) -> Optional[DataFrame]:
        """
        Args:
            X:
                training input, shape: (n_point, n_inp)
                if X is None, self.X is assigned to X

            Y:
                training target, shape: (n_point, n_out)
                if Y is None, self.Y is assigned to Y

        Returns:
            Data frame created from X and Y arrays
            OR
            None if X is None or Y is None
        """
        if X is None:
            X = self._X
        if Y is None:
            Y = self._Y
        X = convert_to_2d(X)                       # sets valid 2D array
        Y = convert_to_2d(Y)                       # sets valid 2D array
        x_keys, y_keys = self._x_keys, self._y_keys
        
        if X is None or Y is None:
            return None
        
        assert X.shape[0] == Y.shape[0], str(X.shape) + ', ' + str(Y.shape)

        if x_keys is None:
            if X is not None:
                x_keys = ['x' + str(i) for i in range(X.shape[1])]
            else:
                x_keys = []
        else:
            x_keys = np.atleast_1d(self._x_keys).tolist()
        if y_keys is None:
            if Y is not None:
                y_keys = ['y' + str(i) for i in range(Y.shape[1])]
            else:
                y_keys = []
        else:
            y_keys = np.atleast_1d(self._y_keys).tolist()
        dic: Dict[str, Float2D] = OrderedDict()
        for j in range(len(x_keys)):
            dic[x_keys[j]] = X[:, j]
        for j in range(len(y_keys)):
            dic[y_keys[j]] = Y[:, j]
        return DataFrame(dic)

    def xy_to_frame(self) -> DataFrame:
        """
        Returns:
            Data frame created from self._x and self._y
        """
        return self.XY_to_frame(self._x, self._y)

    @property
    def weights(self) -> Float1D:
        """
        Returns:
            Array of weights
        """
        return self._weights

    @weights.setter
    def weights(self, value: Float1D) -> None:
        """
        Args:
            value:
                Array of weights
        """
        if value is None:
            self._weights = None
        else:
            self._weights = np.atleast_1d(value)

    @property
    def metrics(self) -> Dict[str, Any]:
        """
        Returns:
            Metrics of best training
        """
        return self._metrics

    @metrics.setter
    def metrics(self, value: Optional[Dict[str, Any]]) -> None:
        """
        Args:
            value:
                Metrics of best training
                or 
                None -> set initial metrics data
        """
        if value is None:
            self._metrics = self.init_metrics()
        else:
            self._metrics = value

    def train(self, X: Float2D, Y: Float2D, **kwargs: Any) -> Dict[str, Any]:
        """
        Trains model. This method has to be overwritten in derived classes.
        X and Y are stored as self.X and self.Y if both are not None

        Args:
            X:
                training input, shape: (n_point, n_inp)

            Y:
                training target, shape: (n_point, n_out)

        Kwargs:
            trainer (str or list of str):
                training methods

            epochs (int):
                maximum number of epochs

            goal (float):
                residuum to be met

            trials (int):
                number of repetitions of training with same method
            ...

        Returns:
            metrics of best training trial:
                'abs'      (float): max{|phi(x) - Y|} of best train
                'i_abs'      (int): index of Y where absolute error is max
                'i_trial'    (int): index of best trial
                'epochs'     (int): number of epochs of best (neural) train
                'iterations' (int): number of iterations
                'L2'       (float): sqrt{sum{(phi(x)-Y)^2}/N} of best train
                'trainer'    (str): best training method

        Note:
            self.metrics['trainer'] is set to None 
                - if X or Y is None or 
                - if training fails 
        """
        self.ready = True
        self._weights = None
        self.metrics = self.init_metrics()

        if X is not None and Y is not None:
            self.X, self.Y = X, Y
            self._n_inp = self.X.shape[1]
            #
            # ... INSERT TRAINING IN DERIVED CLASSES ...
            #
            # self.ready = ...
            # self.weights = ...
            # if self.ready:
            #    self.metrics = {'trainer': ..., 'L2': ..., 'abs': ...,
            #                    'epochs': ...}
            # else:
            #    self.metrics = self.init_metrics()

        return self.metrics

    def predict(self, x: Float2D, *c: float, **kwargs: Any) -> Float2D:
        """
        Executes model. If MPI is available, execution is distributed.
        x and y=f(x) is stored as self.x and self.y
        
        If self.ready is not True, None is returned 

        Args:
            x:
                prediction input, shape: (n_point, n_inp) or (n_inp,)
                shape: (n_inp,) is tolerated

            c:
                positional arguments to be passed to theoretical
                submodel f()

        Kwargs:
            Keyword arguments to be passed to theoretical submodel f()

        Returns:
            prediction output 
            or
            None if x is None or not self.ready
        """
        self.x = x                      # setter ensuring valid 2D array

        assert self._n_inp == -1 or self._n_inp == self.x.shape[1], \
            str((self._n_inp, self.x.shape, self.x))

        if not self.ready:
            self._y = None
        elif not communicator() or x.shape[0] <= 1:
            # self.y is a setter ensuring valid 2D array
            self.y = [np.atleast_1d(self.f(x, *c)) for x in self.x]
        else:
            # self.y is a setter ensuring valid 2D array
            self.y = predict_scatter(
                self.f, self.x, *c, **self.kwargs_del(kwargs, 'x'))
            
        return self.y

    def evaluate(self, X: Float2D, Y: Float2D, 
                 *c: float, **kwargs: Any) -> Dict[str, Any]:
        """
        Evaluates difference between prediction y(X) and reference Y(X)

        Args:
            X:
                reference input, shape: (n_point, n_inp)

            Y:
                reference output, shape: (n_point, n_out)

            c:
                tuning paremeters as positional arguments

        Kwargs:
            silent (bool):
                if True then print of norm is suppressed
                default: False

        Returns:
            metrics of evaluation
                'abs' (float): max{|net(x) - Y|} of best training
                'i_abs' (int): index of Y where absolute err is max
                'L2'  (float): sqrt{sum{(net(x)-Y)^2}/N} best train
        Note:
            maximum abs index is 1D index, eg yAbsMax=Y.ravel()[i_abs_max]
        """
        metrics = self.init_metrics()
        metrics = {key: metrics[key] for key in ('L2', 'abs', 'i_abs')}
        
        if X is None or Y is None or not self.ready:
            return metrics

        X = convert_to_2d(X)
        Y = convert_to_2d(Y)
        
        assert X.shape[0] == Y.shape[0], str((X.shape, Y.shape))

        y = self.predict(x=X, *c, **self.kwargs_del(kwargs, 'x'))

        try:
            dy = y.ravel() - Y.ravel()
        except (ValueError,AttributeError):
            print('X Y y:', X.shape, Y.shape, y.shape)
            assert 0
            
        i_abs = np.abs(dy).argmax()
        metrics = {'L2': np.sqrt(np.mean(np.square(dy))), 'i_abs': i_abs}
        metrics['abs'] = dy.ravel()[metrics['i_abs']]

        if not kwargs.get('silent', True):
            self.write('    L2: ' + str(np.round(metrics['L2'], 4)) +
                       ' max(abs(y-Y)): '+str(np.round(metrics['abs'], 5))+
                       ' [' + str(i_abs) + '] x,y:(' +
                       str(np.round(X.ravel()[i_abs], 3)) + ', ' +
                       str(np.round(Y.ravel()[i_abs], 3)) + ')')
            
        return metrics

    def pre(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Kwargs:
            XY (tuple of two 2D array of float, and optionally two list 
                of str):
                training input & training target, optional keys of X and Y
                'XY' supersedes 'X' and 'Y'

            X (2D array of float):
                training input, shape: (n_point, n_inp)
                'XY' supersede 'X' and 'Y'
                default: self.X

            Y (2D array of float):
                training target, shape: (n_point, n_out)
                'XY' supersede 'X' and 'Y'
                default: self.Y

         Returns:
            metrics of best training trial if 'XY' or ('X' and 'Y') 
            in 'kwargs', see description of metrics in BoxModel.train()

        Note:
            self.X and self.Y are overwritten by setters of XY or (X, Y)
        """
        super().pre(**kwargs)

        XY = kwargs.get('XY', None)
        if XY is not None:
            self.XY = XY                 # sets tuple of valid 2d arrays
        else:
            self.X = kwargs.get('X', None)         # sets valid 2D array
            self.Y = kwargs.get('Y', None)         # sets valid 2D array

        # trains model if self.X is not None and self.Y is not None
        if type(self).__name__ == 'White':
            self.metrics = self.init_metrics()
            self.ready = True
        
            return self.metrics
        
        if self.X is not None and self.Y is not None:
            kwargs_ = self.kwargs_del(kwargs, ('X', 'Y'))
            self.metrics = self.train(X=self.X, Y=self.Y, **kwargs_)
            self.ready = self.metrics['trainer'] is not None
        
        return self.metrics
                

    def task(self, **kwargs: Any) -> Union[Float2D, Dict[str, Any]]:
        """
        Kwargs:
            x (2D array of float):
                prediction input, shape: (n_point, n_inp) 
                shape: (n_inp,) is tolerated

        Returns:
            prediction output if x is not None and self.ready
            or
            metrics of best training of train(),
                see description of metrics in BoxModel.train()
            or 
            None if x is None and not self.ready

        Note:
            x will be saved as self.x and the output as self.y
        """
        super().task(**kwargs)

        x = kwargs.get('x', None)
        if x is None:
            return self.metrics

        self.x = x                                 # sets valid 2D array
        if self.ready:
            self.y = self.predict(x=self.x, **self.kwargs_del(kwargs, 'x'))
        else:
            self.y = None                          # sets valid 2D array
 
        return self.y
