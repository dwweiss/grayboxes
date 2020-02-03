"""f
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
      2020-02-03 DWW
"""

__all__ = ['Neural']

import numpy as np
from operator import itemgetter
#from time import time
from typing import Any, Callable, Dict, List, Optional

from tensorflow.keras import Sequential
#from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Input
#from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import (# Adadelta, Adagrad, 
                                         Adam, Adamax,
                                         # Ftrl, 
                                         Nadam, SGD, RMSprop)
from tensorflow.keras.callbacks import (Callback, EarlyStopping, 
                                        ModelCheckpoint, ReduceLROnPlateau)
import logging
logging.getLogger('tensorflow').disabled = True # disable tensorflow log

try:
    from grayboxes.datatypes import Float2D, Function
except:
    try:
        from datatypes import Float2D, Function
    except:
        print('??? module datatypes not imported')
        print('    ==> copy file datatypes.py to this directory')
        print('    continue with unauthorized definition of Float2D, Function')
        Float2D = Optional[np.ndarray]
        Function = Optional[Callable[..., List[float]]]
try:
    from grayboxes.metrics import init_metrics  # TODO , update_errors
except:
    try:
        from metrics import init_metrics  # TODO , update_errors
    except:
        print('??? module metrics not imported')
        print('    ==> copy file metrics.py to this directory')
try:
    from grayboxes.neural0 import Neural0
except:
    try:
        from neural0 import Neural0
    except:
        print('??? module neural0 not imported')
        print('    ==> copy file neural0.py to this directory')


class Neural(Neural0):
    """
    Wraps neural network implementations from Keras (Tensorflow backend)

    Major methods and attributes:
        see neural0.Neural0
        
    Literature:
        https://www.tensorflow.org/tutorials/keras/regression
        https://arxiv.org/abs/1609.04747
        
    Hints:
        Conclusion from regression of sin(x) + noise:        
            Proposed configuration:
                activation: 'elu', 'relu' or 'tanh' 
                    (relu is sometimes faster than elu; both elu and 
                     relu often faster than tanh), failure of: sigmoid 
                output: linear 
                trainer: Adam, Nadam - failure of: Ftrl, RMSProp, SGD
                trials: 3..5
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
        
    def n_inp(self) -> Optional[int]:
        if not self.ready:
            return None
        
        n_inp_ = self._net.input[self._net.input.value_index].shape[0]
        assert n_inp_ == super().n_inp(), str(n_inp_)

        return n_inp_

    def train(self, X: Float2D, Y: Float2D, **kwargs: Any) -> Dict[str, Any]:
        """
        Trains model, stores X and Y as self.X and self.Y, and stores 
        metrics of best training trial as self.metrics

        Args:
            X:
                training input, shape: (n_point, n_inp)
                shape: (n_point,) is tolerated
                default: self.X

            Y:
                training target, shape: (n_point, n_out)
                shape: (n_point,) is tolerated
                default: self.Y

        Kwargs:
            see Neural0.train()
            
        Returns:
            metrics of best training trial:
                see Neural0.train()
        """        
        if X is not None and Y is not None:
            self.set_XY(X, Y)
        assert self._X is not None and self._Y is not None, \
            str(self.X) + ' ' + str(self.Y)
            
        n_inp, n_out = self.X.shape[1], self.Y.shape[1]
        assert self.X.shape[0] == self.Y.shape[0], \
            str((self.X.shape, self.Y.shape))

        default_activation = 'sigmoid'
        default_activation_out = 'sigmoid'
        min_hidden_layers = 1 
        max_hidden_layers = 5 
        min_layer_size = max(3, n_inp, n_out)
        max_layer_size = min(10, min_layer_size * 4)
        default_hidden_neurons = \
            [[i] * j  for j in range(min_hidden_layers, max_hidden_layers + 1) 
                      for i in range(min_layer_size, max_layer_size + 1)]

        ################################
        # get_implementation_specific_options()
        #
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
        #
        ################################

        all_hidden_activations = np.atleast_1d(kwargs.get('activation', 
                                               default_activation)).tolist()
        if len(all_hidden_activations) == 1 and \
            all_hidden_activations[0].lower() in ('auto', ):
            all_hidden_activations = ['sigmoid']
        all_hidden_activations = [act.replace('tansig', 'tanh') 
                                  for act in all_hidden_activations]
        all_hidden_activations = [act.replace('logsig', 'sigmoid') 
                                  for act in all_hidden_activations]
        all_hidden_activations = [act.replace('lin', 'linear') 
                                  for act in all_hidden_activations]
        all_hidden_activations = [act.replace('purelin', 'linear') 
                                  for act in all_hidden_activations]
        
        all_hidden_neurons = kwargs.get('neurons', default_hidden_neurons)
        if not all_hidden_neurons:
            all_hidden_neurons = default_hidden_neurons 
        if (isinstance(all_hidden_neurons, str)
            and all_hidden_neurons.lower().startswith(('auto', 'brut', ))):
            all_hidden_neurons = default_hidden_neurons 
        if isinstance(all_hidden_neurons, (int,)):
            all_hidden_neurons = [all_hidden_neurons]
        if not any(isinstance(x, (list, tuple,)) for x in all_hidden_neurons):
            all_hidden_neurons = [all_hidden_neurons]
        if len(all_hidden_neurons) == 1 and not all_hidden_neurons[0]:
            all_hidden_neurons = default_hidden_neurons
            
        all_trainers = np.atleast_1d(kwargs.get('trainer', '')).tolist()            
        if len(all_trainers) == 1 and all_trainers[0].lower() in ('auto', ''):
            all_trainers = default_trainer
        if len(all_trainers) == 1 and all_trainers[0].lower() in ('all', ''):
            all_trainers = trainer_pool.keys()
        if any([key not in trainer_pool for key in all_trainers]):
            all_trainers = default_trainer
        epochs = kwargs.get('epochs', 250)
        learning_rate = kwargs.get('learning_rate', 0.1)
        if 'expected' in kwargs:
            self._mse_expected = kwargs.get('expected', 0.5e-3)
        else:
            self._mse_expected = kwargs.get('goal', 0.5e-3)
        self._mse_tolerated = kwargs.get('tolerated', 5e-3)
        output_activation = kwargs.get('output', 'auto')
        if output_activation.lower().startswith('lin'):
            output_activation = 'linear'
        if output_activation.lower().startswith('auto'):
            output_activation = default_activation_out
        patience = kwargs.get('patience', 10)
        plot = kwargs.get('plot', 0)
        regularization = kwargs.get('regularization', None)
        if regularization is None:
            regularization = kwargs.get('rr', None)
        self.silent = kwargs.get('silent', self.silent)
        trials = kwargs.get('trials', 5)
        validation_split = kwargs.get('validation_split', 0.20)
        verbose = 0 if self.silent else kwargs.get('verbose', 0)
         
        if not self.silent:
            print('+++ hidden_neurons:')
            for lay in all_hidden_neurons:
                print('    layers: ' + str(len(lay)), end='')
                for nrn in lay:
                    print('  ' + nrn * '=', end='')
                print()
            print()

        #################
        #            
        # get_callbacks(**kwargs: Any) -> List[Any]
        #     Kwargs: 
        #         epochs, patience, best_net_file
        #
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
        if not self.silent:
            callbacks.append(_PrintDot())
        if patience > 0:
            callbacks.append(EarlyStopping(monitor='val_loss', mode='auto', 
                patience=patience, min_delta=1e-4, verbose=0))
        if self._best_net_file:
            callbacks.append(ModelCheckpoint(self._best_net_file, 
                save_best_only=True, monitor='val_loss', mode='auto'))
        if True:
            callbacks.append(ReduceLROnPlateau(monitor='val_loss', mode='auto',
                factor=0.666, patience=5, min_delta=1e-4, min_lr=5e-4,
                verbose=0))
        #
        ##################

        net_metric_keys = ['mse', ]
        assert 'mse' in net_metric_keys
        
        best_trial: Optional[Dict[str, Any]] = None
        all_trials: List[Dict[str, Any]] = []
        stop_early = False
        
        # scale training data
        X_scaled = self._scale(self.X, self._X_stats,
                               min_max_scale=self._min_max_scale)
        Y_scaled = self._scale(self.Y, self._Y_stats, 
                               min_max_scale=self._min_max_scale)
        
        # tf.keras.fit() would shuffle after splitting. 
        # Therefore shuffling is done here AND the shuffle 
        # argument is not used when calling tf.keras.fit()
        if kwargs.get('shuffle', True):
            X_scaled, Y_scaled = self._shuffle(X_scaled, Y_scaled)
        if 0. < validation_split <= 1.:
            n_trn = int(X_scaled.shape[0] * (1. - validation_split))
            X_trn, Y_trn = X_scaled[:n_trn], Y_scaled[:n_trn]
            X_val, Y_val = X_scaled[n_trn:], Y_scaled[n_trn:]
        else:
            X_trn, Y_trn = X_scaled, Y_scaled
            X_val, Y_val = X_trn, Y_trn
        validation_data = (X_val, Y_val)

        for neurons in all_hidden_neurons:        
            if not self.silent:
                print('+++ neurons:', neurons)
                
            for activation in all_hidden_activations:    
                if not self.silent:
                    print('+++ activation:', activation)
                                    
#                def create_keras_net(n_inp: int, neurons: Iterable[float], 
#                                     activation: str, output: str, 
#                                     regularization: Optional[float] = None) \
#                                     -> Any:
#                    net = Sequential()
#                    net.add(Input(shape=(n_inp,)))
#                    if regularization is None:
#                        for nrn in np.atleast_1d(neurons):       
#                            net.add(Dense(units=nrn, activation=activation))
#                        net.add(Dense(units=n_out, 
#                                      activation=output_activation))
##                    else:
#                        for nrn in np.atleast_1d(neurons):       
#                            net.add(Dense(units=nrn, activation=activation,
#                                kernel_regularizer=regularizers.l2(
#                                    regularization)))
#                        net.add(Dense(units=n_out, 
#                            activation=output_activation,
#                            kernel_regularizer=regularizers.l2(regularization)
#                            ))
                        
#                    return net        
                
#                self._net = create_keras_net(n_inp, n_out, neurons, activation, 
#                                             output_activation)
                self._net = Sequential()
                self._net.add(Input(shape=(n_inp,)))
                for nrn in np.atleast_1d(neurons):       
                    self._net.add(Dense(units=nrn, activation=activation,
#                       kernel_regularizer=regularizers.l2(regularization)
                        ))
                self._net.add(Dense(units=n_out, activation=output_activation,
#                   kernel_regularizer=regularizers.l2(regularization))
                    ))
                #
                ################

                for trainer in all_trainers:
                    if not self.silent:
                        print('+++ trainer:', trainer)
                                                                                        
                    kwargs_ = {'clipvalue': 0.667, 
                               'learning_rate': learning_rate} 
                    if trainer == 'sgd':
                        kwargs_.update({'decay': 1e-6, 
                                        'momentum': 0.8, 
                                        'nesterov': True})
                    optimizer = trainer_pool[trainer](**kwargs_)
                                                
                    # compile() only sets these four arguments
                    valid_keys = ('loss_weights', 'sample_weight_mode', 
                        'target_tensors', 'weighted_metrics')
                    kwargs_ = {k: kwargs[k] for k in kwargs.keys()
                                                if k in valid_keys}
                    for trial in range(trials):    
                        if not self.silent:
                            print('+++ trial:', trial)
    
                        # random re-initialization of weights
                        weights = self._net.get_weights()
                        weights = [np.random.permutation(w.flat).reshape(
                                w.shape) for w in weights]
                        self._net.set_weights(weights)

                        self._net.compile(loss='mse', 
                            metrics=net_metric_keys, verbose=0,
                            optimizer=optimizer, **kwargs_)
                        
                        # training with early stopping, see callbacks
                        valid_keys = ('batch_size', ) 
                        kwargs_ = {k: kwargs[k] for k in kwargs.keys()
                                                    if k in valid_keys}
                        hist = self._net.fit(x=X_trn, y=Y_trn,
                            callbacks=callbacks, 
                            epochs=epochs, 
                            shuffle=False,  # shuffling was done above
                            validation_data=validation_data, 
                            verbose=verbose, **kwargs_)

                        # metrics of actual trial
                        actual_trial = {
                            'activation': activation,
                            'epochs': len(hist.history['mse']),
                            'i_history': len(all_trials),
                            'L2': np.sqrt(hist.history['mse'][-1]),
                            'mse_final': hist.history['mse'][-1],
                            'net': self._net,
                            'neurons': neurons,
                            'trainer': trainer, 
                            'trial': trial, 
                            }                        
                        for key in hist.history.keys():
                            actual_trial.update({key: hist.history[key]})
                                                
                        # actual history of all erors/losses
                        for key in net_metric_keys:
                            actual_trial[key] = hist.history[key]

                        # add to all histories
                        all_trials.append(actual_trial)
        
                        # update best history
                        if best_trial is None or \
                         (best_trial['mse_final'] > actual_trial['mse_final']):
                            best_trial = actual_trial
                            
                        # plot all histories and indicate best one
                        if plot: 
                            self._plot_all_trials(best_trial, all_trials)
                        print()
                        
                        if True:
                            if best_trial is not None and \
                                  best_trial['mse_final'] < self._mse_expected:
                                print('==> early stop of multiple trials,',
                                      'mse:', best_trial['mse_final'])
                                stop_early = True

                        if stop_early: break
                    if stop_early: break
            if stop_early: break
        
        if plot:
            self._plot_best(best_trial)
        
        self._net = best_trial['net']
        
        mse_final = best_trial['mse'][-1]

        self._ready = (mse_final <= self._mse_tolerated)

        self._metrics = init_metrics({
            'activation': activation, 
            'epochs': len(best_trial['mse']), 
            'L2': np.sqrt(mse_final), 
            'mse_final': mse_final,
            'ready': self.ready,
            'trainer': trainer
            })
        
        if not self.silent:
            print('+++ mse:', self._metrics['mse_final'], 
                  'L2:', self._metrics['L2'])

        if plot:
            n_best = 5

            print('*** history of all trials')
            self._plot_all_trials(best_trial, all_trials)        
            sorted_histories = sorted(all_trials, reverse=False, 
                                      key=itemgetter('mse_final'))
            if len(all_trials) > n_best:
                print('*** history of best', n_best, 'trials')
                self._plot_all_trials(best_trial, sorted_histories[:n_best])
                
                self._plot_err_bars(best_trial, sorted_histories[:n_best])
                self._plot_err_bars(best_trial, sorted_histories)

        return self.metrics
    
    def _predict_scaled(self, x_scaled: Float2D, **kwargs) -> Float2D:
        """
        Executes the network with scaled input
        
        Args:
            x_scaled:
                scaled input, shape: (n_point, n_inp)
                n_inp must equal self.X.shape[1]

        Kwargs:
            batch_size (int or None):
                see keras.model.predict()
                default: None

            verbose (int):
                see keras.model.predict()

        Returns:
            scaled output, shape: (n_point, self.Y.shape[1])
        """
        valid_keys = ('batch_size', 'verbose') 
        
        kwargs_ = {k: kwargs[k] for k in kwargs.keys() if k in valid_keys}
        
        return self._net.predict(x_scaled, **kwargs_)

