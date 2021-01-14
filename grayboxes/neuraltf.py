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
      2021-01-14 DWW
"""

__all__ = ['Neural']

import logging
logging.getLogger('tensorflow').disabled = True # disable tensorflow log

import numpy as np
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import (Callback, EarlyStopping, 
                                        ModelCheckpoint, ReduceLROnPlateau)
from tensorflow.keras.layers import Dense, Input, LeakyReLU

# TODO cecide on using load_model()
# from tensorflow.keras.models import load_model

from tensorflow.keras.optimizers import (Adam, Adamax, Nadam, SGD, RMSprop)
          # low performance in regression analysis: Adadelta, Adagrad, Ftrl

# TODO cecide on using network regularization
# from tensorflow.keras import regularizers

from grayboxes.bruteforce import BruteForce
from grayboxes.datatype import Float2D, Function


class Neural(BruteForce):
    """
    Wraps neural network implementations from Keras (Tensorflow backend)

    Major methods and attributes:
        see super().super()
        
    Literature:
        https://www.tensorflow.org/tutorials/keras/regression
        https://arxiv.org/abs/1609.04747        
    """

    def __init__(self, f: Function = None) -> None:
        """
        Args:
            f:
                theor. submodel as method f(self, x) or function f(x)

        Note: if f is not None, genetic training or training with
            derivative dE/dy and dE/dw is employed
        """
        super().__init__(f=f)
        self._backend = 'tensorflow'
        
    def n_inp(self) -> Optional[int]:
        """
        See super().n_inp()
        """        
        if not self.ready:
            return None
        
        n_inp_ = self._net.input[self._net.input.value_index].shape[0]
        assert n_inp_ == super().n_inp(), str(n_inp_)

        return n_inp_

    def _create_net(self, 
                    n_inp: int, 
                    hiddens: Iterable[int], 
                    n_out: int, 
                    activation: str, 
                    output_activation: str,
                    X_stats: Iterable[Dict[str, float]]) -> Any:
        """
        See super()._create_net()
        """        
        if activation.lower().startswith('leaky'):
            activation = LeakyReLU()

        net = Sequential()
        net.add(Input(shape=(n_inp,)))
        for hidden in np.atleast_1d(hiddens):
            net.add(Dense(units=hidden, activation=activation,))
        net.add(Dense(units=n_out, activation=output_activation,))

        return net

    def _create_callbacks(self, 
                          epochs: int, 
                          silent: bool, 
                          patience: int,
                          best_net_file: str) -> List[Any]:
        """
        See super()._create_callbacks()
        """
        adapt_learning_rate = True
        
        class _PrintDot(Callback):
            def on_epoch_end(self, epochs_, logs):
                if epochs_ == 0:
                    print('+++ epochs: ', end='')
                if epochs_ % 25 == 0:
                    print(str(epochs_) + ' ', end='')
                else:
                    if epochs_ + 1 == epochs:
                        print(str(epochs_ + 1) + ' ', end='')
        callbacks = []
        if not silent:
            callbacks.append(_PrintDot())
        if patience > 0:
            callbacks.append(EarlyStopping(monitor='val_loss', mode='auto', 
                patience=patience, min_delta=1e-4, verbose=0))
        if self._best_net_file:
            callbacks.append(ModelCheckpoint(self._best_net_file, 
                save_best_only=True, monitor='val_loss', mode='auto'))
        if adapt_learning_rate:
            callbacks.append(ReduceLROnPlateau(monitor='val_loss', 
                mode='auto', factor=0.666, patience=5, min_delta=0., 
                min_lr=5e-4, verbose=0))
            
        return callbacks
    
    def _get_weights(self) -> Float2D:
        w_tensorflow = self._net.get_weights()
        w_numpy = np.asfarray(w_tensorflow)
        
        return w_numpy
        
    def _set_weight(self, weights: Float2D) -> bool:
        if weights is None:
            return False
        w_tensorflow = weights
        self._net.set_weights(w_tensorflow)
        
        return True
    
    def _get_trainer_pool(self) -> Tuple[Dict[str, Any], List[str]]:
        """
        See super()._get_trainer_pool()
        """
        trainer_pool = {
                        # 'adadelta': Adadelta,
                        # 'adagrad': Adagrad,
                        'adam': Adam,
                        'adamax': Adamax,
                        # 'ftrl': Ftrl,
                        'nadam': Nadam,
                        'sgd': SGD,
                        'rmsprop': RMSprop,
                        'rprop': RMSprop,
                       }
        default_trainer = ['adam', 'rmsprop']
        
        return trainer_pool, default_trainer

    def _set_trainer(self, 
                     trainer: str, 
                     trainer_pool: Dict[str, Any],
                     net_metric_keys: Iterable[str],
                     **kwargs: Any) -> None:                                           
        """
        See super()._set_trainer()
        """
        opt= {}
        opt['learning_rate'] = kwargs.get('learning_rate', 0.1) 
        if trainer == 'sgd':
            opt['decay'] = kwargs.get('decay', 1e-6)
            opt['momentum'] = kwargs.get('momentum', 0.8)
            opt['nesterov'] = kwargs.get('nesterov', True)
        optimizer = trainer_pool[trainer](**opt)
                                    
        opt = {k: kwargs[k] for k in kwargs.keys() if k in \
                            ('loss_weights', 'sample_weight_mode', 
                             'target_tensors', 'weighted_metrics')
              }
        self._net.compile(loss='mean_squared_error',
            metrics=net_metric_keys, optimizer=optimizer, **opt)

    def _train_scaled(self, X: Float2D, Y: Float2D, 
                      **kwargs: Any) -> Dict[str, Any]: 
        """
        See super()._train_scaled()
        """
        hist = self._net.fit(X, Y,
            shuffle=False,  # shuffling has been done in super().train()
            **self._kwargs_get(kwargs, ('batch_size', 'callbacks', 'epochs', 
                                        'validation_data', 'verbose',)))        
        return hist.history
    
    def _predict_scaled(self, x_scaled: Float2D, **kwargs) -> Float2D:
        """
        See super()._predict_scaled()
        """
        return self._net.predict(x_scaled, 
            **self._kwargs_get(kwargs, ('batch_size', 'verbose',)))
        