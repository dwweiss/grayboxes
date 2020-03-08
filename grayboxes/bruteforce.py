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

__all__ = ['BruteForce']

import inspect
#try: 
#    from keyboard import is_pressed
#except:
#    pass
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
#import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

try:
    from grayboxes.datatype import Float1D, Float2D, Function
except:
    try:
        from datatype import Float1D, Float2D, Function
    except:
        print('??? module datatype not imported')
        print('    ==> copy file datatype.py to this directory')
        print('    continue with unauthorized definition of Float1D, ' \
              'Float2D, Function')

        Float1D = Optional[np.ndarray]
        Float2D = Optional[np.ndarray]
        Function = Optional[Callable[..., List[float]]]
try:
    from grayboxes.metrics import init_metrics, update_errors
except:
    try:
        from metrics import init_metrics, update_errors
    except:
        print('??? module metrics not imported')
        print('    ==> copy file metrics.py to this directory')

    
class BruteForce(object):
    """
    - Encapsulation of implementation of empirical models
    - Brute force configuration of empirical model
    - Graphic pesentation of training history

    Example of training and prediction of  neural network derived
    from this class
    
        - compact form:
              y = Neural()(X=X, Y=Y, x=x, neurons=[6,4], trainer='adam')
        - expanded form:
              phi = Neural()
              metrics = phi(X=X, Y=Y, neurons=[6,4], trainer='adam')
              y = phi(x=x)
              L2_norm = metrics['L2']            # or: phi.metrics['L2']

    Major methods and attributes (return type in the comment):
        - y = Neural()(X=[[..]], Y=[[..]], x=[[..]], **kwargs) 
                                             # y.shape: (n_point, n_out)
        - metrics = self.train(X, Y, **kwargs)        # see self.metrics
        - y = self.predict(x, **kwargs)      # y.shape: (n_point, n_out)
        - self.ready                                              # bool
        - self.metrics          # dict{str: float/str/int} + init/update
        - self.plot()

    Note:
        This class has not the connectivity functionality of the 
        box type models derived from class BoxModel.
        
    Literature:
        Activation:
            https://ml-cheatsheet.readthedocs.io/en/latest/
                activation_functions.html
        Regularization:             
            https://www.analyticsvidhya.com/blog/2018/04/
                fundamentals-deep-learning-regularization-techniques/   
            https://machinelearningmastery.com/how-to-stop-training-deep-
                neural-networks-at-the-right-time-using-early-stopping/ 
    """

    def __init__(self, f: Function = None) -> None:
        """
        Args:
            f:
                theoret. submodel as method f(self, x) or function f(x)

        Note: if f is not None, genetic training or training with
            derivative dE/dy and dE/dw is employed
        """
        self._backend : str = ''             # info about implementation
        self._best_net_file: str = ''                # './best_net.hdf5'
        
        self.f: Function = f     # theor. submodel for single data point
        self.identifier = 'BruteForce'
        self._net: Any = None                           # neural network
        self._metrics: Dict[str, Any] = init_metrics()        
        
        self._min_max_scale: bool = True   # False if normal distributed
        
        self._mse_expected: float = 0.5e-3 # if fulfilled,no more trials
        self._mse_tolerated: float = 5e-3 # if exceeded,self.ready=False
        
        self._ready: bool = False              # if True, net is trained
        self._scale_margin = 0.1     # scaled margin for min-max-scaling
        self._silent: bool = False         # True if no print to console

        self._X: Float2D = None    # train input, shape: (n_point,n_inp)
        self._Y: Float2D = None        # target, shape: (n_point, n_out)
        self._x: Float2D = None   # pred. input, shape: (n_point, n_inp)
        self._y: Float2D = None               # prediction y = net(x, w)        
        self._X_stats: List[Dict[str, float]] = []     # shape: (n_inp,)
                 # dictionary with mean, std, min and max of all columns
        self._Y_stats: List[Dict[str, float]] = []     # shape: (n_out,) 
                 # dictionary with mean, std, min and max of all columns
        self._x_keys: List[str] = []                          # x labels
        self._y_keys: List[str] = []                          # y labels
        
        plt.rcParams.update({'font.size': 10})              # axis fonts
        plt.rcParams['legend.fontsize'] = 12           # fonts in legend

    def _key_pressed(self, hot_key: str = 'q') -> bool:
#       TODO return 'keyboard' in sys.modules and is_pressed(hot_key)
        return False
            
    def _kwargs_del(self, kwargs: Dict[str, Any],
                    exclude: Union[str, Iterable[str]]) -> Dict[str, Any]:
        """
        Creates a copy of kwargs exclusive the members with keys given 
        in 'excluded'
        
        Returns:
            copy of kwargs excluding values with keys given by 'exclude'
        """
        if isinstance(exclude, str):
            exclude = [exclude]
            
        return {k: kwargs[k] for k in kwargs.keys() if k not in exclude}

    def _kwargs_get(self, kwargs: Dict[str, Any],
                    include: Union[str, Iterable[str]]) -> Dict[str, Any]:
        """
        Creates a copy of members of kwargs which keys are in 'include'
        
        Returns:
            new dictionary with the values of kwargs which have keys 
            contained in 'include')
        """
        if isinstance(include, str):
            include = [include]
            
        return {k: kwargs[k] for k in kwargs.keys() if k in include}

    def _create_net(self, 
                    n_inp: int, 
                    hiddens: Iterable[int], 
                    n_out: int, 
                    activation: str, 
                    output_activation: str,
                    X_stats: Iterable[Dict[str, float]]) -> Optional[Any]:
        """
        Args:
            n_inp:
                number of input neurons

            hiddens:
                list of number of neurons of hidden layers 

            n_out:
                number of output neurons
                
            activation:
                identifier of activation function of hidden layers

            output_activation:
                identifier of activation function of hidden layers
                
            X_stats:
                list of dictionary of 'min' & 'max' value of every input
                
        Returns:
            multi-layer perceptron with defined activation functions        
        """
        print('??? neural0._create_net() does create a net')           
        
        return None

    def _create_callbacks(self, 
                          epochs: int, 
                          silent: bool, 
                          patience: int,
                          best_net_file: str) -> List[Any]:
        """
        Args:
            epochs:
                maximum number of epochs
                
            silent:
                If True, then no print to console 
                
            patience:
                number of epochs before pplying early stopping
                
            best_net_file:
                name of file for storing best network during training 
        
        Returns:
            List of a relevant callbacks for network training
        """
        return []
    
    def _get_trainers(self) -> Tuple[Dict[str, Any], List[str]]:
        """
        Returns:
            2-tuple of 
                dictionary of a all avaible trainers for actual 
                implementation 
                and 
                List of keys of default trainers
        """
        trainer_pool = {}
        default_trainer = []
        
        return trainer_pool, default_trainer

    def _randomize_weights(self, activation: str = 'sigmoid') -> Optional[Any]:
        """
        initializes randomly the weights of the neural network
        
        Args:
            net:
                neural network
        
        Returns:
            network with randomly initialzed weights
        """
        print('??? super()._randomize_weights() does not modify weights')
        return None        
        

    def _create_hiddens(self, n_inp: int, n_out: int) -> List[List[int]]:
        """
        initializes randomly the weights of the neural network
        
        Args:
            n_inp:
                number of inputs

            n_out:
                number of outputs
        
        Returns:
            Sequence of arrays of number of hidden neurons per layer
              hiddens[j][k], j=0..n_variation-1, k=1..n_hidden_layer
        """
        min_hidden_layers = 1 
        max_hidden_layers = 5 
        min_layer_size = max(3, n_inp, n_out)
        max_layer_size = min(8, min_layer_size * 2)

        hiddens = \
            [[i] * j  for j in range(min_hidden_layers, max_hidden_layers + 1) 
                      for i in range(min_layer_size, max_layer_size + 1)]
        return hiddens

    def _check_activation(self, activations: Union[Iterable[str], 
                                                   str]) -> List[str]:
        """
        Checks validity of activations and corrects if invalid
        
        Args:
            activations:
                single activation or list of activation functions

        Returns:
                corrected list of activation functions
        """
        activations = np.atleast_1d(activations)
        for (alternative, default) in [('tansig', 'tanh'), 
                                       ('logsig', 'sigmoid'),
#                                       ('lin', 'linear'), 
                                       ('purelin', 'linear')
                                      ]:
            activations = [act.replace(alternative, default) 
                           for act in activations]
        return activations

    def train(self, X: Float2D, Y: Float2D, **kwargs: Any) -> Dict[str, Any]:
        """
        Trains model, stores X and Y as self.X and self.Y, and stores 
        result of best training trial as self.metrics
        
        [Keras only] see 'patience' for deactivation of early stopping

        Args:
            X:
                training input (real-world), shape: (n_point, n_inp)
                shape: (n_point,) is tolerated
                default: self.X

            Y:
                training target (real-world), shape: (n_point, n_out)
                shape: (n_point,) is tolerated
                default: self.Y

        Kwargs:
            activation (str or list of str):
                activation function of hidden layers
                  'elu': alpha*(exp(x)-1) if x <= 0 else x  [Keras only]
                  'leakyrelu': max(0, x) - alpha*max(0, -x) [Keras only]
                  'linear': x (unmodified input)            
                  'relu': max(0, x)                         [Keras only]
                  'sigmoid': (LogSig): 1 / (1 + exp(-z))
                  'tanh': (TanSig): tanh(x) or 
                default: 'sigmoid'

            batch_size (int or None) [Keras only] :
                see keras.model.predict()
                default: None

            epochs (int):
                max number of epochs of single trial
                default: 250

            expected (float):
                limit of error for stop of training (0. < goal < 1.)
                default: 5e-4
                [identical with 'goal', 'expected' superseeds 'goal']

            goal (float) [Neurolab only]:
                limit of error for stop of training (0. < goal < 1.)
                default: 5e-4
                [note: MSE of 1e-4 corresponds to L2-norm of 1e-2]
                [identical with 'expected', 'expected' superseeds 'goal']

            learning_rate (float or list of float) [Keras only]:
                learning rate of optimizer
                default: None

            momentum (float) [Keras only]:
                momentum of optimizer
                default: None

            neurons (list of int, or list of list of int):
                array of number of neurons in hidden layers

            output (str or None):
                activation function of output layer, see 'activation'
                default: 'linear'

            patience (int) [Keras only]:
                controls early stopping of training
                if patience is <= 0, then early stopping is deactivated
                default: 30

            plot (int):
                controls frequency of plotting progress of training
                default: 0 (no plot)

            regularization (float) [Neurolab only]:
                regularization rate (sum of all weights is added to
                cost function of training, 0. <= regularization <= 1.
                default: 0. (no effect of sum of all weights)
                [same as 'rr']
                [note: neurolab trainer 'bfgs' ignores 'rr' argument]

            rr (float) [Neurolab only]:
                [same as 'regularization']

            show (int) [Neurolab only]:
                control of information about training, if show>0: print
                default: epochs // 10
                [argument 'show' superseeds 'silent' if show > 0]

            silent (bool):
                if True then no information is sent to console
                default: self.silent
                [Neurolab] argument 'show' superseds 'silent' if show > 0
                
            trials (int):
                maximum number of training trials
                default: 3
                
            trainer (str or list of str):
                if 'all' or None, then all training methods are assigned
                default: 'auto' ==> ['adam', 'rprop']

            tolerated (float):
                limit of error for declaring network as ready
                default: 5e-3
                [note: MSE of 1e-4 corresponds to L2-norm of 1e-2]

            trials (int):
                maximum number of training trials
                default: 3

            validation_split (float) [Keras only] :
                share of training data excluded from training
                default: 0.25

            verbose (int):
                controls print of progress of training and prediction
                default: 0 (no prints)
                ['silent' supersedes level of 'verbose']

        Returns:
            metrics of best training trial:
                'abs'      (float): max{|net(x) - Y|} of best trial
                'activation' (str): max{|net(x) - Y|} of best trial
                'epochs'     (int): number of epochs of best trial
                'i_abs'      (int): index of Y where abs. error is max.
                'L2'       (float): sqrt{sum{(net(x)-Y)^2}/N},best trial
                'mse_trn'(float): sum{(net(x)-Y)^2}/N of best trial
                'mse_val'(float): sum{(net(x)-Y_val)^2}/N of best trial
                'trainer'    (str): best training method

        Note:
            - Reference to optional theor. submodel is stored as self.f
            - References to real-world training data are stored as 
              self._X and self._Y
            - Reference to real-world prediction input is stored as 
              self._x
            - The best network is stored as self._net
            - If training fails, then self.ready is False
        """    
        if X is not None and Y is not None:
            self.set_XY(X, Y)
        assert self._X is not None and self._Y is not None, \
            str(self.X) + ' ' + str(self.Y)
            
        n_inp, n_out = self.X.shape[1], self.Y.shape[1]
        assert self.X.shape[0] == self.Y.shape[0], \
            str((self.X.shape, self.Y.shape))

        default_act_hid = 'sigmoid'
        default_act_out = 'linear'
        default_hiddens = self._create_hiddens(n_inp, n_out)
        
        trainer_pool, default_trainer = self._get_trainers()

        all_act_hid = np.atleast_1d(kwargs.get('activation', 
            default_act_hid)).tolist()
        if len(all_act_hid) == 1 \
                and all_act_hid[0].lower() in ('auto', ):
            all_act_hid = ('sigmoid', 'tanh', 'elu',)
                    
        all_hiddens = kwargs.get('neurons', default_hiddens)
        if not all_hiddens:
            all_hiddens = default_hiddens 
        if (isinstance(all_hiddens, str)
            and all_hiddens.lower().startswith(('auto',))):
            all_hiddens = default_hiddens 
        if isinstance(all_hiddens, (int,)):
            all_hiddens = [all_hiddens]
        if not any(isinstance(x, (list, tuple,)) for x in all_hiddens):
            all_hiddens = [all_hiddens]
        if len(all_hiddens) == 1 and not all_hiddens[0]:
            all_hiddens = default_hiddens
            
        all_trainers = np.atleast_1d(kwargs.get('trainer', '')).tolist()            
        if len(all_trainers) == 1 and all_trainers[0].lower() in ('auto', ''):
            all_trainers = default_trainer
        if len(all_trainers) == 1 and all_trainers[0].lower() in ('all', ''):
            all_trainers = trainer_pool.keys()
        if any([key not in trainer_pool for key in all_trainers]):
            all_trainers = default_trainer
        epochs = kwargs.get('epochs', 250)
#        learning_rate = kwargs.get('learning_rate', 0.1)
        if 'expected' in kwargs:
            self._mse_expected = kwargs.get('expected', 0.5e-3)
        else:
            self._mse_expected = kwargs.get('goal', 0.5e-3)
        self._mse_tolerated = kwargs.get('tolerated', 5e-3)
        activation_output = kwargs.get('output', None)

        if activation_output is None:
            activation_output = default_act_out
        if activation_output is not None:
            if activation_output.lower().startswith('lin'):
                activation_output = 'linear'
            if activation_output.lower().startswith('auto'):
                activation_output = default_act_out

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
            for layer in all_hiddens:
                print('    ' + str(len(layer)) + ' layers ', end='')
                for n in layer:
                    print('  ' + n * '=', end='')
                print()

        all_act_hid = self._check_activation(all_act_hid)
        if activation_output is not None:
            activation_output = self._check_activation(activation_output)[0]
        
        callbacks = self._create_callbacks(epochs, self.silent, patience, 
                                           self._best_net_file)
        net_metric_keys = ['mse', ]
        assert 'mse' in net_metric_keys
        
        best_trial: Optional[Dict[str, Any]] = None
        all_trials: List[Dict[str, Any]] = []
        stop_early = False
        
        # scale training data
#        TODO .
#        self._min_max_scale = not (activation_output == 'tanh' \
#                              and all(a == 'tanh' for a in all_act_hid))
        self._min_max_scale = True
        X_scaled = self._scale(self.X, self._X_stats,
                               min_max_scale=self._min_max_scale)
        Y_scaled = self._scale(self.Y, self._Y_stats, 
                               min_max_scale=self._min_max_scale)
        
        # tf.keras.fit() would shuffle after splitting. Therefore 
        # shuffling is done here AND the shuffle argument is False 
        # when calling tf.keras.fit()
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

        for hiddens in all_hiddens:
            if not self.silent:
                print('+++ hiddens:', hiddens)
                
            for act_hid in all_act_hid:
                act_out = activation_output
                if act_out is None:
                    act_out = act_hid
                if not self.silent:
                    print('+++ activation:', act_hid, '&', act_out)
                                        
                for trainer in all_trainers:
                    if not self.silent:
                        print('+++ trainer:', trainer)
                                             
                    for trial in range(trials):    
                        if not self.silent:
                            print('+++ trial:', trial)
                                                
                        if self._net:
                            del self._net
                        self._net = self._create_net(n_inp, hiddens, n_out,
                            act_hid, act_out, self._X_stats)
                        self._set_trainer(trainer, trainer_pool, 
                            net_metric_keys, **self._kwargs_del(kwargs, 
                                                                'trainer'))
                        
                        # training with early stopping, see callbacks
                        history = self._train_scaled(X=X_trn, Y=Y_trn,
                            batch_size=kwargs.get('batch_size', None),
                            callbacks=callbacks, 
                            epochs=epochs, 
                            shuffle=False,    # shuffling was done above
                            validation_data=validation_data, 
                            verbose=verbose, 
                        )

                        # metrics of actual trial
                        actual_trial = {
                            'activation': act_hid,
                            'backend': self._backend,
                            'epochs': len(history['mse']),
                            'i_history': len(all_trials),
                            'L2': np.sqrt(history['mse'][-1]),
                            'mse_trn': history['mse'][-1],
                            'mse_val': history['val_mse'][-1],
                            'net': self._net,
                            'neurons': hiddens,
                            'trainer': trainer, 
                            'trial': trial, 
                            }                        
                        for key in history.keys():
                            actual_trial.update({key: history[key]})
                                                
                        # actual history of all erors/losses
                        for key in net_metric_keys:
                            actual_trial[key] = history[key]

                        # adds actual history to all histories
                        all_trials.append(actual_trial)
        
                        # updates best history
                        if best_trial is None or \
                         (best_trial['mse_trn'] > actual_trial['mse_trn']):
                            best_trial = actual_trial
                            
                        # plots all histories and indicate best one
                        if plot: 
                            self._plot_all_trials(best_trial, all_trials)
                        print()

                        # stop training manually or when goal achieved                        
                        if self._key_pressed():
                            stop_early = True
                        if best_trial is not None and \
                              best_trial['mse_trn'] < self._mse_expected:
                            stop_early = True
                            print('==> early stop of multiple trials,',
                                  'mse (trn/val):', best_trial['mse_trn'],
                                  best_trial['mse_val'])

                        if stop_early: break
                    if stop_early: break
            if stop_early: break
        
        if plot:
            self._plot_best(best_trial)
        
        self._net = best_trial['net']
        
        mse_trn_final = best_trial['mse'][-1]
        mse_val_final = best_trial['val_mse'][-1]

        self._ready = (mse_trn_final <= self._mse_tolerated)

        self._metrics = init_metrics({
            'activation': act_hid, 
            'epochs': len(best_trial['mse']), 
            'L2': np.sqrt(mse_trn_final), 
            'mse_trn': mse_trn_final,
            'mse_val': mse_val_final,
            'ready': self.ready,
            'trainer': trainer
            })
        
        if not self.silent:
            print('+++ mse_trn/val:', (self._metrics['mse_trn'], 
                  self._metrics['mse_val']), 'L2:', self._metrics['L2'])

        if plot:
            n_best = 5

            print('*** history of all trials')
            self._plot_all_trials(best_trial, all_trials)        
            sorted_histories = sorted(all_trials, reverse=False, 
                                      key=itemgetter('mse_trn'))
            if len(all_trials) > n_best:
                print('*** history of best', n_best, 'trials')
                self._plot_all_trials(best_trial, sorted_histories[:n_best])
                
                self._plot_err_bars(best_trial, sorted_histories[:n_best])
                self._plot_err_bars(best_trial, sorted_histories)
                
                
        if plot:
            self._plot_network()
            
        return self.metrics

    def _predict_scaled(self, x_scaled: Float2D, **kwargs) -> Float2D:
        """
        Args:
            x_scaled:
                scaled prediction input, shape: (n_point, n_inp)
                shape: (n_inp,) is tolerated

        Kwargs:                
            additional keyword arguments of actual backend

        Returns:
            scaled prediction y = net(x_scaled), shape: (n_point, n_out)
        """        
        raise NotImplementedError

    @property
    def X(self) -> Float2D:
        return self._X

    @property
    def Y(self) -> Float2D:
        return self._Y

    @property
    def x(self) -> Float2D:
        return self._x

    @property
    def y(self) -> Float2D:
        return self._y

    @property
    def f(self) -> Function:
        return self._f

    @f.setter
    def f(self, value: Function) -> None:
        if value is not None:
            first_arg = list(inspect.signature(value).parameters.keys())[0]
            if first_arg == 'self':
                value = value.__get__(self, self.__class__)
        self._f = value
        
    @property
    def silent(self) -> bool:
        return self._silent

    @silent.setter
    def silent(self, value: bool) -> None:
        self._silent = value
        
    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def metrics(self) -> Dict[str, Any]:
        """
        Returns:
            metrics of best training trial
            [see self.train()]
        """
        return self._metrics
    
    def __call__(self, 
                 X: Float2D = None, 
                 Y: Float2D = None, 
                 x: Float2D = None, 
                 **kwargs: Any) -> Union[Dict[str, Any], Float2D]:
        """
        - Trains neural network if X is not None and Y is not None
        - Sets self.ready to True if training is successful
        - Predicts y for input x if x is not None and self.ready is True
        - Plots history of training and comparison of train and test data
        - Stores best result 
        - After training, class keeps the best training result

        Args:
            X:
                training input, shape: (n_point, n_inp) 
                shape: (n_point,) is tolerated
                default: self.X
                
            Y:
                training target, shape: (n_point, n_out)
                shape: (n_point,) is tolerated
                default: self.Y

            x:
                prediction input, shape: (n_point, n_inp)
                shape: (n_inp,) is tolerated
                default: self.x

        Kwargs:
            keyword arguments, see: train() and predict()

        Returns:
            metrics of best training trial if X and Y are not None,
                see self.train()
            OR
            prediction of net(x) if x is not None and self.ready,
                2D array with shape: (n_point, n_out)
            OR
            empty metrics 
                if x is None and ((X, Y are None) or (not self.ready))

        Note:
            - X.shape[1] must equal x.shape[1]
            - Shape of X, Y and x is corrected to (n_point, n_inp/n_out)
            - References to X, Y, x and y are stored as self.X, self.Y,
              self.x, self.y, see self.train() and self.predict()
        """
        if X is not None and Y is not None:
            self._metrics = self.train(X=X, Y=Y, **kwargs)
            
        if x is not None and self.ready:
            return self.predict(x=x, **kwargs)

        return self.metrics

    def _shuffle(self, 
                 X: Float2D, 
                 Y: Float2D) -> Tuple[Float2D, Float2D]:
        """
        shuffles unisonely two 2D arrays
        
        Args:
            X:
                scaled 2D array of shape (n_point, n_inp)

            Y:
                scaled 2D array of shape (n_point, n_out)
                                
        Returns: 
            shuffled 2D arrays, shapes: (n_point, n_inp), 
                                        (n_point, n_out)
        """
        p = np.random.permutation(X.shape[0])
        
        return X[p, :], Y[p, :]

    def _scale(self, 
               X: Float2D, 
               X_stats: Iterable[Dict[str, float]],
               min_max_scale: bool) -> Float2D:
        """
        Scales array X:
          1. X_scaled = (X - min(X) / (max(X) - min(X)) if min_max_scale
          2. X_scaled = (X - mean(X) / std(X)                  otherwise
        
        Args:
            X:
                real-world 2D array of shape (n_point, n_inp/n_out)
                
            X_stats:
                list of dictionary of statistics of every column of X
                (keys: 'min', 'max', 'mean', 'std')
                
            min_max_scale:
                if True, X has been scaled in [0, 1] range, else
                X has been normalized with mean and standard deviation                
                
        Returns: 
            scaled 2D array, shape: (n_point, n_inp/n_out)
            or 
            None if X is None
            
        Note:
            see super().set_XY for the lower und upper bound of X_stat 
        """
        if X is None:
            return None

        X_scaled = np.zeros(X.shape)
        
        if min_max_scale:
            for j in range(X.shape[1]):
                X_scaled[:,j] = (X[:,j] - X_stats[j]['min']) / \
                    (X_stats[j]['max'] - X_stats[j]['min'])
        else:
            for j in range(X.shape[1]):
                X_scaled[:,j] = (X[:,j] - X_stats[j]['mean']) / \
                    (X_stats[j]['std'])
                    
        return X_scaled 

    def _descale(self, 
                 X: Float2D, 
                 X_stats: Iterable[Dict[str, float]],
                 min_max_scale: bool) -> Float2D:
        """
        Descales array:
          1. X_real_world = min(X) + X_scaled * (max(X) - min(X))  
                                                        if min_max_scale
          2. X_real_world = X_scaled * std(X) + mean(X)        otherwise
          
        Args:
            X:
                scaled 2D array of shape (n_point, n_inp/n_out)
                
            X_stats:
                list of dictionary of statistics of every column of X
                (keys: 'min', 'max', 'mean', 'std')

            min_max_scale:
                if True, X has been scaled in [0, 1] range, else
                X has been normalized with mean and standard deviation                

        Returns: 
            real-world 2D array, shape: (n_point, n_inp/n_out)
            or 
            None if X is None
        """
        if X is None:
            return None
        
        X_real_world = np.zeros(X.shape)

        if min_max_scale:
            for j in range(X.shape[1]):
                X_real_world[:,j] = X_stats[j]['min'] + \
                    X[:,j] * (X_stats[j]['max'] - X_stats[j]['min'])
        else:
            for j in range(X.shape[1]):
                X_real_world[:,j] = X[:,j] * (X_stats[j]['std']) + \
                    X_stats[j]['mean']
        
        return X_real_world 

    def set_XY(self, X: Union[Float2D, Float1D], 
               Y: Union[Float2D, Float1D],
               x_keys: Optional[Iterable[str]] = None, 
               y_keys: Optional[Iterable[str]] = None) -> None:
        """
        - Stores training input X and training target Y as self.X and 
          self.Y
        - converts self.X and self.Y to 2D arrays
        - transposes self.X and self.Y if n_point < n_inp/n_out

        Args:
            X:
                training input, shape: (n_point, n_inp)
                shape: (n_point,) is tolerated

            Y:
                training target, shape: (n_point, n_out)
                shape: (n_point,) is tolerated

            x_keys:
                list of column keys for data selection
                use self._x_keys keys if x_keys is None
                default: ['x0', 'x1', ... ]

            y_keys:
                list of column keys for data selection
                use self._y_keys keys if y_keys is None
                default: ['y0', 'y1', ... ]
        """
        self._X = np.atleast_2d(X)
        self._Y = np.atleast_2d(Y)

        if self._X.shape[0] < self._X.shape[1]:
            self._X = self._X.transpose()
        if self._Y.shape[0] < self._Y.shape[1]:
            self._Y = self._Y.transpose()
            
        assert self._X.shape[0] == self._Y.shape[0], \
            'input arrays incompatible [' + str(self._X.shape[0]) + \
            ']  vs. [' + str(self._Y.shape[0]) + ']\n' + \
            'self._X: ' + str(self._X) + '\nself._Y: ' + str(self._Y)

        # min, max, mean and standard deviation of all columns of X and Y
        self._X_stats = [{'mean': c.mean(), 'std': c.std(), 
                          'min': c.min(), 'max': c.max()} for c in self._X.T]
        self._Y_stats = [{'mean': c.mean(), 'std': c.std(),
                          'min': c.min(), 'max': c.max()} for c in self._Y.T]

        # 10% safety margin in distance between lower and upper bound
        for array in (self._X_stats, self._Y_stats):
            for column in array:
                margin = self._scale_margin * (column['max'] - column['min'])
                column['min'] -= margin  
                column['max'] += margin  
    
        # avoid zero division in normalization of X and Y 
        for stats in (self._X_stats, self._Y_stats):
            for col in stats:
                if np.isclose(col['std'], 0.0):
                    col['std'] = 1e-10
                    
        # set default keys 'xi' and 'yj' if not x_keys or y_keys
        if not x_keys:
            self._x_keys = ['x' + str(i) for i in range(self._X.shape[1])]
        else:
            self._x_keys = x_keys
        if not y_keys:
            self._y_keys = ['y' + str(i) for i in range(self._Y.shape[1])]
        else:
            self._y_keys = y_keys

    def predict(self, x: Float2D, **kwargs: Any) -> Float2D:
        """
        Executes the network if it is ready
        
        - Reshapes and scales real-world input x, 
        - Executes network, 
        - Rescales the scaled prediction y, 
        - Stores x as self.x

        Args:
            x:
                real-world prediction input, shape: (n_point, n_inp)
                shape: (n_inp,) is tolerated

        Kwargs:                
            plot (bool):
                if true, plots comparison of train data with prediction
                default: False

            additional keyword arguments, see: self._predict_scaled()

        Returns:
            real-world prediction y = net(x)
            or
            None if x is None or not self.ready or 
                x shape is incompatible

        Note:
            - Shape of x is corrected to: (n_point, n_inp) if x is 1D
            - Input x and output net(x) are stored as self.x and self.y
        """
        if x is None or not self.ready or self.n_inp() is None:
            self._x, self._y = None, None
            return None
    
        x = np.asfarray(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.shape[1] != self.n_inp():
            x = np.transpose(x)
        if x.shape[1] != self.n_inp():
            print('??? incompatible input shape:', x.shape)
            self._x, self._y = None, None
            return None
        
        self._x = x
        x_scaled = self._scale(self.x, self._X_stats, 
                               min_max_scale=self._min_max_scale)
        
        y_scaled = self._predict_scaled(x_scaled, **kwargs)

        if y_scaled is None:
            if not self.silent:
                print('??? predict: y_scaled is None')
            return None
        
        self._y = self._descale(y_scaled, self._Y_stats, 
                                min_max_scale=self._min_max_scale)

        if kwargs.get('plot', False):             
            self._plot_train_vs_pred()
                
        return self._y
        
    def evaluate(self, 
                 X_ref: Float2D, 
                 Y_ref: Float2D, 
                 **kwargs: Any) -> Dict[str, Any]:
        """
        Evaluates difference between prediction y(X_ref) and 
        given reference Y_ref(X_ref)

        Args:
            X_ref:
                real-world reference input, shape: (n_point, n_inp)

            Y_ref:
                real-world reference output, shape: (n_point, n_out)

        Kwargs:
            silent (bool):
                if True then print of norm is suppressed
                default: self.silent
                
            plot (bool):
                if true, plots comparison of train data with prediction
                default: False

            additional keyword arguments, see: self._predict_scaled()

        Returns:
            metrics of evaluation, see metrics.init_metrics()
            or 
            default_metrics, see metrics.init_metrics()

        Note:
            max. abs index is 1D index, y_abs_max=Y.ravel()[i_abs_max]
        """
        metrics = init_metrics()
        
        if X_ref is None or Y_ref is None or not self.ready:
            return metrics

        y = self.predict(x=X_ref, **kwargs)

        metrics = update_errors(metrics, X_ref, Y_ref, y, 
                                silent=kwargs.get('silent', self.silent))
        return metrics

    def n_point(self) -> Optional[int]:
        if self._X is None:
            return None

        return self._X.shape[0]

    def n_inp(self) -> Optional[int]:
        if not self._X_stats:
            return None

        return len(self._X_stats)

    def n_out(self) -> Optional[int]:
        if not self._Y_stats:
            return None

        return len(self._Y_stats)

    def plot(self) -> None:
        self.plot_network()
        self._plot_train_vs_pred()

    def _plot_network(self, file: str = '') -> None:
        pass

    def _plot_all_trials(self, 
                         best: Dict[str, Any],
                         histories: Iterable[Dict[str, Any]],
                         metrics_keys: Union[str, Iterable[str]] = 'mse',
                         ) -> None:
        
        metrics_keys = np.atleast_1d(metrics_keys).tolist()
        assert len(metrics_keys) > 0
        
        n_act = len(np.unique([hist['activation'] for hist in histories]))
        n_bck = len(np.unique([hist['backend'   ] for hist in histories]))
        n_nrn = len(np.unique([hist['neurons'   ] for hist in histories]))
        n_trl = len(np.unique([hist['trial'     ] for hist in histories]))
        n_trn = len(np.unique([hist['trainer'   ] for hist in histories]))
                                
        nrn = best.get('neurons', '?')
        if len(nrn) == 1:
            nrn = str(nrn)
        else:
            nrn = str(nrn)[1:-1].replace(', ', ':')

        plt.title(str(np.round(1e3 * best.get(metrics_keys[0])[-1], 2)) + \
                  'e-3 ' + best.get('trainer', '?')[:8] + \
                  ':' + str(best.get('activation', '?')[:10]) + \
                  ' #' + str(best.get('trial', '?')) + \
                  ' ' + nrn)
        for history in histories:
            for key in list(metrics_keys):
                if key in history:
                    s = ''
                    if n_trn > 1:
                        s += str(history.get('trainer', '?')[:8]) + ':' 
                    if n_act > 1:
                        s += history.get('activation', '?')
                    if n_trl > 1:
                        s += '#' + str(history.get('trial', '?'))
                    if n_nrn > 1:
                        hn = '[' + str(history.get('neurons', '?'))
                        s += hn[1:-1].replace(', ', ':') + ']'
                    if n_bck > 0:
                        hn = ' $' + str(history.get('backend', '?')[:1])
                        s += hn
                    if key == 'mse':
                        plt.plot(history[key], '-', label=s)
                    else:
                        plt.plot(history[key], ':')
                
        if n_nrn * n_act * n_trn * n_trl > 1:
            plt.scatter([len(best[metrics_keys[0]])-1], 
                        [best[metrics_keys[0]][-1]], 
                        color='b', marker='o', label='best')
        plt.yscale('log')
        plt.ylim([1e-6, 1.])
        plt.xlabel('epoch')
        plt.ylabel('mean square error')
        if n_trn * n_act * n_trl * n_nrn > 1:
            plt.legend(bbox_to_anchor=(1.1, 1.05), loc='upper left')
        plt.grid()
        plt.show()

    def _plot_best(self, best: Dict[str, Any]) -> None:
        has_curves = False        
        keys = ['mse', 'val_mse']
        if not all(key in best for key in keys):
            keys = ['loss', 'val_loss']
        if 'lr' in best:
            keys.append('lr')
        for key in keys:
            if key in best:
                has_curves = True
                if np.isnan(best[key][-2]):
                    plt.plot(best[key], '.', label=key)
                else:
                    plt.plot(best[key], label=key)
        if has_curves:
            plt.title('Best trial')
            plt.xlabel('epochs')
            plt.ylabel('metrics')
            plt.yscale('log')
            plt.grid()
            plt.legend()
            plt.show()

    def _plot_err_bars(self, 
                       best: Dict[str, Any], 
                       histories: Iterable[Dict[str, Any]],
                       metrics_key: str = 'mse_trn') -> None:
        """
        Args:
            best:
                metrics of best trial
            
            histories:
                sequence of metrices 
                
            metrics_key:
                key of metric defining primaery measure of error
        """
        nrn = best.get('neurons', '?')
        if len(nrn) == 1:
            nrn_str = ':' + str(nrn)
        else:
            nrn_str = str(nrn)[1:-1].replace(', ', ':')
        n_act = len(np.unique([hist['activation'] for hist in histories]))
        n_bck = len(np.unique([hist['backend'   ] for hist in histories]))
        n_nrn = len(np.unique([hist['neurons'   ] for hist in histories]))
        n_trl = len(np.unique([hist['trial'     ] for hist in histories]))
        n_trn = len(np.unique([hist['trainer'   ] for hist in histories]))

        x_, y_ = [], []
        for history in histories:
            s = ''
            if n_trn > 1:
                s += str(history['trainer']) + ':' 
            if n_act > 1:
                s += history['activation']
            if n_trl > 1:
                s += '#' + str(history['trial'])
            if n_nrn > 1:
                s += '(' + str(history['neurons'])[1:-1].replace(', ', ':')+')'
            if n_bck > 0:
                s += ' $' + str(history['backend'])[:1]

            x_.append(s)
            y_.append(history['mse_trn'])

        fig_size_y = max(3, len(histories) * 0.25)
        plt.figure(figsize=(6, fig_size_y))
        plt.title(str(round(1e3 * best.get(metrics_key, '?'), 2)) + 'e-3 ' + \
                  best.get('trainer', '?')[:] + \
                  ':' + str(best.get('activation', '?')[:]) + \
                  ' #' + str(best.get('trial', '?')) + \
                  ' ' + nrn_str)
        plt.xlabel('mean square error')
        plt.ylabel('variants')
        plt.xscale('log')
        plt.barh(x_[::-1], np.array(y_[::-1]).clip(1e-6, 1))
        plt.vlines(self._mse_expected, -0.5, len(x_) - 0.5, 
                   color='g', linestyles='-', label='expected')
        plt.vlines(self._mse_tolerated, -0.5, len(x_) - 0.5, 
                   color='r', linestyles='-', label='tolerated')
        plt.grid()
        plt.legend()
        plt.show()

    def _plot_train_vs_pred(self, **kwargs) -> None:
        """
        Kwargs:
            see self._predict_scaled()
        
        Note:
            This method is called by self.predict(). Thus, this method 
            calls self._predict_scaled() instead of self.predict()
        """

        X_scaled = self._scale(self.X, self._X_stats, self._min_max_scale)   
        
        Y_prd_scaled = self._predict_scaled(X_scaled, **kwargs)
        
        Y_prd = self._descale(Y_prd_scaled, self._Y_stats, self._min_max_scale)

        if Y_prd is None:
            if not self.silent:
                print('??? plot train vs pred: predict() returned None')
            return 
            
        dY = Y_prd - self.Y

        X_, Y_ = self.X[:,0], self.Y[:,0]
        Y_prd_, dY_ = Y_prd[:,0], dY[:,0]

        plt.title('Train data versus prediction')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.plot(X_, Y_, '.', c='r', label='train')
        plt.plot(X_, Y_prd_, '.', c='b', label='pred')
        plt.legend()
        DY = self.Y.max() - self.Y.min()
        plt.ylim([self.Y.min() - 0.5 * DY, 2 * self.Y.max() + 0.5 * DY])
        plt.grid()
        plt.show()
            
        plt.title('Prediction minus target')
        plt.xlabel('$x$')
        plt.ylabel(r'$\Delta y = \phi(X) - Y$')
        plt.plot(X_, dY_, '.')
        plt.grid()
        plt.show()

        plt.title('Target versus prediction')
        plt.xlabel('target $Y$')
        plt.ylabel('prediction $y$')
        plt.plot(Y_, Y_, '-', label='$Y(X)$')
        plt.plot(Y_, Y_prd_, '.', label='$y(X)$')
        plt.legend()
        plt.grid()
        plt.show()
