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

__all__ = ['Neural']

import logging
logging.getLogger('tensorflow').disabled = True # disable tensorflow log

import numpy as np
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import (Callback, EarlyStopping, 
                                        ModelCheckpoint, ReduceLROnPlateau)
from tensorflow.keras.layers import Dense, Input, LeakyReLU
#from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import (# Adadelta, Adagrad, 
                                         Adam, Adamax,
                                         # Ftrl, 
                                         Nadam, SGD, RMSprop)
#from tensorflow.keras import regularizers
#from tensorflow.keras.utils import plot_model

try:
    from grayboxes.bruteforce import BruteForce
except:
    try:
        from bruteForce import BruteForce
    except:
        print('??? module bruteforce not imported')
        print('    ==> copy file bruteforce.py to this directory')
try:
    from grayboxes.datatype import Float2D, Function
except:
    try:
        from datatype import Float2D, Function
    except:
        print('??? module datatype not imported')
        print('    ==> copy file datatype.py to this directory')
        print('    continue with unauthorized definition of Float2D, Function')
        
        Float2D = Optional[np.ndarray]
        Function = Optional[Callable[..., List[float]]]


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
        if True:
            callbacks.append(ReduceLROnPlateau(monitor='val_loss', 
                mode='auto', factor=0.666, patience=5, min_delta=0., 
                min_lr=5e-4, verbose=0))
            
        return callbacks
    
    def _randomize_weights(self, 
                           activation: str = 'sigmoid', 
                           min_: float = -0.1, 
                           max_: float = +0.1) -> None:
        """
        See super()._randomize_weights()
        """        
        weights = self._net.get_weights() 

        if activation == 'tanh':
            weights = [np.random.normal(0., 0.05, size=w.shape) 
                       for w in weights]
        else:
#            ptp = max_ - min_
#            min_ -= ptp * 0.25
#            max_ += ptp * 0.25
#            min_ = -1 / np.sqrt(self.n_point())
            lo = -0.25
            hi = -lo

            weights = [np.random.uniform(low=lo, high=hi, size=w.shape) 
                       for w in weights]

#        weights = [np.random.permutation(w.flat).reshape(w.shape) 
#                   for w in weights]

#        print('%2 weights min max:', min(w.min() for w in weights), 
#              min(w.max() for w in weights))

        self._net.set_weights(weights)

    def _get_trainers(self) -> Tuple[Dict[str, Any], List[str]]:
        """
        See super()._get_trainers()
        """
        trainer_pool = {
#                        'adadelta': Adadelta,
#                        'adagrad': Adagrad,
                        'adam': Adam,
                        'adamax': Adamax,
#                        'ftrl': Ftrl,
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
#            opt['clip_value'] = kwargs.get('clipvalue', 0.667) 
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
            batch_size=kwargs.get('batch_size', None),
            callbacks=kwargs.get('callbacks', None),
            epochs=kwargs.get('epochs', 250), 
            shuffle=False,  # shuffling has been done in super().train()
            validation_data=kwargs.get('validation_data', None), 
            verbose=kwargs.get('verbose', 0), 
            )
        
        return hist.history
    
    def _predict_scaled(self, x_scaled: Float2D, **kwargs) -> Float2D:
        """
        See super()._predict_scaled()
        """
        return self._net.predict(x_scaled, 
            **self._kwargs_get(kwargs, ('batch_size', 'verbose')))
        
    def _plot_network(self, file: str = '') -> None:
#        if not file:
#            file = './network_structure.png'
#        try:
#            plot_model(self._net, to_file=file, show_shapes=False, 
#                       show_layer_names=True, rankdir='TB', 
#                       expand_nested=False, dpi=96)    
#            if not self.silent:
#                print('+++ plot network')
#        except:
#            if not self.silent:
#                print('!!! plot network failed in pydotprint'
        pass