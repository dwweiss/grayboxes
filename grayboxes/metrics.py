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
      2020-03-24 DWW
"""

__all__ = ['Metrics', 'best_metrics', 'init_metrics', 'update_errors']

import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Dict, Iterable, Optional

try:
    from grayboxes.datatype import Float2D
except ImportError:
    try:
        from datatype import Float2D
    except ImportError:
        print('!!! Module datatype not loaded')
        print('    continue with local definition of Float2D')        
        Float2D = Optional[np.ndarray]


class Metrics(dict):
    """
    Stores metrics of model performance as a dictionary
    Sets errors from reference data and prediction output (abs, mse, L2) 
    Plots target, prediction and location of maximum absolure error
    """
    
    def __init__(self, other: Optional[Dict[str, Any]] = None) -> None:
        """
        Sets default values to metrics describing model performance,
        see init_metrics() for default values 
    
        Args:
            other:
                other dictionary with initial metrics to be added to this 
                metrics
    
        Returns:
            dictionary with default settings for 
                - metrics of best training trial 
                - model evaluation
        """
        super().__init__(self)
        self.update(init_metrics())
        if other is not None:
            self.update(other)
            
    def update_errors(self, X: Float2D, Y: Float2D, y: Float2D, 
                      **kwargs) -> 'Metrics':
        """
        Evaluates difference between prediction y(X) and reference Y(X)

        Args:
            X:
                reference input, shape: (n_point, n_inp)

            Y:
                reference output, shape: (n_point, n_out)
                
            y:
                predicted output, shape: (n_point, n_out)
                
        Kwargs:
            silent (bool):
                if False, then print metrics

            plot (bool):
                if True, then plot error over training input and 
                identify location of greatest absolute error

        Returns:
            Metrics of model performance
            
        Note:
            'i_abs' is an 1D index, eg y_abs_max = Y.ravel()[i_abs_max]
        """
        self.update(update_errors(self, X, Y, y, **kwargs))
        
        return self
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Returns:
            content as dictionary
        """
        dict_ = {}
        for key, value in self.items():
            dict_[key] = value
        return dict_

    def plot(self, key: str, 
             metrices: Iterable[Dict[str, Any]]) -> None:
        plt.title("History of '" + str(key) + "'")
        has_labels = False
        for metrics in metrices:
            if key in metrics:
                has_labels = True
                y = np.atleast_1d(metrics[key])
                label = 'key: ' + str(round(y[-1], 3))
                if len(y) > 1:
                    plt.plot(y, label=label)
                else:
                    plt.scatter([0], y, label=label)
        plt.xlabel('index')
        plt.ylabel(key)
        if has_labels:
            plt.legend()
        plt.grid()
        plt.show()


def init_metrics(other: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Sets default values to metrics describing model performance

    Args:
        other:
            other dictionary with initial metrics to be added to this 
            metrics

    Returns:
        dictionary with default settings for 
            - metrics of best training trial 
            - model evaluation
    """    
    metrics = {
       'abs': np.inf,                           # maximum absolure error
       'activation': None,           # activation function of best trial
       'epochs': -1,                    # number of epochs of best trial        
       'evaluations': -1,          # number of evaluations of best trial
       'i_abs': -1,                 # 1D index of maximum absolute error
       'iterations': -1,            # number of iterations of best trial 
       'L2': np.inf,              # prediction minus true/reference data
       'mse': np.inf,                 # prediction minus validation data
       'ready': True,                     # True if ready for prediction
       'trainer': None,                          # trainer of best trial
       'trial': -1,                                # index of best trial
       'weights': None,
       }

    if other is not None:            
        for key, value in other.items():
            metrics[key] = value
            
    return metrics


def best_metrics(metrics_sequence: Iterable[Dict[str, Any]], 
                 key: str = 'mse') -> Optional[Dict[str, Any]]:
    """
    Args:
        metrics_sequence:
            sequence of metrices having 'key' as dictionary key
            
        key:
            key of value used for comparisons
            
    Returns:
        best metrics out of sequence of metrices
    """
    metrics_sequence = np.atleast_1d(metrics_sequence)
    if len(metrics_sequence) == 0:
        return None
    
    if any(key not in metrics for metrics in metrics_sequence):
        return None
    
#    i_best = 0
#    for i in range(len(metrics_sequence)):
#        if metrics_sequence[i][key] < metrics_sequence[i_best][key]:
#            i_best = i
#    return metrics_sequence[i_best]
            
    sorted_list = sorted(metrics_sequence, key=lambda dic: dic[key])
    
    return sorted_list[0]
            

def update_errors(metrics: Dict[str, Any], 
                  X: Float2D, 
                  Y: Float2D, 
                  y: Float2D, 
                  **kwargs) -> Dict[str, Any]:
    """
    Evaluates difference between model prediction y(X) and 
    reference Y(X) 
    
    If X, Y or y is None, then this metrics will stay unchanged 

    Args:
        metrics:
            metrics to be updated
            
        X:
            reference input, shape: (n_point, n_inp)

        Y: 
            reference output, shape: (n_point, n_out)
            
        y:
            predicted output for reference input, shape: (n_point,n_out)         
            
    Kwargs:
        silent (bool):
            if False, print metrics

        plot (bool):
            if True, then plot error over training input and 
            identify location of maximum absolute error

    Returns:
        Metrics of model performance
        
    Note:
        max. abs index is 1D index, y_abs_max=Y.ravel()[i_abs_max]
    """
    silent = kwargs.get('silent', True)

    if X is None or Y is None or y is None:
        if not silent:
            if X is None:
                print('??? metrics: X is None')
            if Y is None:
                print('??? metrics: Y is None')
            if y is None:
                print('??? metrics: y is None')
        return metrics

    if len(Y) != len(y):
        if not silent:
            print('??? metrics: len(Y) != len(y)')
            print('??? metrics: shapes of X Y y:', 
                  np.shape(X), np.shape(Y), np.shape(y))
        return metrics

    X = np.asfarray(X)
    Y = np.asfarray(Y)
    y = np.asfarray(y)
            
    try:
        dy = y.ravel() - Y.ravel()
    except (ValueError, AttributeError):
        print('??? shapes of X Y y:', X.shape, Y.shape, y.shape)
        assert 0
            
    i_abs = np.abs(dy).argmax()
    abs_ = dy.ravel()[i_abs]
    mse = np.mean(np.square(dy))
    L2 = np.sqrt(mse)
    
    metrics['abs'] = abs_
    metrics['i_abs'] = i_abs
    metrics['L2'] = L2
    metrics['mse'] = mse

    if not kwargs.get('silent', True):
        str_i = '[' + str(i_abs) + ']'
        print('    L2: ' + str(np.round(L2, 4)) +
              ', max(abs(y-Y)): ' + str(np.round(abs_, 5)) +
              ' at X' + str_i + '=' + str(np.round(X.ravel()[i_abs], 3)) + 
              ', Y' + str_i + '=' + str(np.round(Y.ravel()[i_abs], 3)) + 
              ', y' + str_i + '=' + str(np.round(y.ravel()[i_abs], 3)))
        
    if kwargs.get('plot', False):
        plt.title('Reference vs prediction')
        plt.xlabel('$X$')
        plt.ylabel('$y, Y$')
        plt.plot(X.reshape(-1), Y.reshape(-1), label='ref $Y$')
        plt.plot(X.reshape(-1), y.reshape(-1), label='pred $y$')
        plt.plot(X.reshape(-1)[i_abs], Y.reshape(-1)[i_abs], 'o', 
                 label='max err')
        plt.grid()
        plt.legend()
        plt.show()

        plt.title('Maximum absolute error')
        plt.xlabel('$X$')
        plt.ylabel('$y - Y$')
        plt.plot(X.reshape(-1), (y - Y).reshape(-1), label='$y-Y$')
        plt.plot(X.reshape(-1)[i_abs], (y - Y).reshape(-1)[i_abs], 
                 'o', label='max abs')
        plt.grid()
        plt.legend()
        plt.show()
        
    return metrics
