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
      2020-02-03 DWW

  Acknowledgements:
      Neurolab is a contribution by E. Zuev (pypi.python.org/pypi/neurolab)
"""

__all__ = ['NeuralBase', 'Neural', 'propose_hidden_neurons']

from collections import OrderedDict
import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

try:
    import neurolab as nl
    _has_neurolab = True
except ImportError:
    print('??? Package neurolab not imported')
    _has_neurolab = False
    
try:
    from grayboxes.datatypes import Float1D, Float2D, Float3D, Function, Str1D
except ImportError:
    try:
        from datatypes import Float1D, Float2D, Float3D, Function, Str1D
    except ImportError:
        print('    continue with unauthorized definition of Float1D, ' +
              'Float2D, Float3D, Function, Str1D')        
        Float1D = Optional[np.ndarray]
        Float2D = Optional[np.ndarray]
        Float3D = Optional[np.ndarray]
        Function = Optional[Callable[..., List[float]]]
        Str1D = Optional[np.ndarray]


def propose_hidden_neurons(X: Float2D, Y: Float2D, alpha: float = 2.,
                           silent: bool = False) -> List[int]:
    """
    Proposes optimal number of hidden neurons for given training data set

                   n_point
    hidden = -----------------------        with: 2 <= alpha <= 10
             alpha * (n_inp + n_out)

    Args:
        X (2D array of float):
            training input, shape: (n_point, n_inp)

        Y (2D array of float):
            training target, shape: (n_point, n_out)

        alpha:
            tuning parameter, alpha = 2..10 (2 reduces over-fitting)
            default: 2

        silent:
            if True then suppress printed messages
            default: False

    Returns:
        Estimate of optimal number of neurons of a single hidden layer
    """
    x_shape, y_shape = np.atleast_2d(X).shape, np.atleast_2d(Y).shape

    assert x_shape[0] > x_shape[1], str(x_shape)
    assert x_shape[0] > 2, str(x_shape)
    assert x_shape[0] == y_shape[0], str(x_shape) + str(y_shape)

    n_point, n_inp, n_out = x_shape[0], x_shape[1], y_shape[1]
    try:
        n_hidden = max(1, round(n_point / (alpha * (n_inp + n_out))))
    except ZeroDivisionError:
        n_hidden = max(n_inp, n_out) + 2
    if not silent:
        print("+++ auto definition of 'n_hidden': " + str(n_hidden))

    return [n_hidden]


class NeuralBase(object):
    """
    Multi-layer perceptron 
    """

    def __init__(self, f: Function = None) -> None:
        """
        Args:
            f:
                theor. submodel as method f(self, x) or function f(x)

        Note: if f is not None, genetic training or training with
            derivative dE/dy and dE/dw is employed
        """
        self.f = f                 # theor. submodel, single data point

        self._net = None           # model
        self._X: Float2D = None    # input of training
        self._Y: Float2D = None    # target
        self._x: Float2D = None    # input of prediction
        self._y: Float2D = None    # prediction y = model(x)
        self._norm_y = None        # data from normalization of target
        self._x_keys: Str1D = None # x-keys for import from data frame
        self._y_keys: Str1D = None # y-keys for import from data frame
        self._trainers = ''        # list of trainers
        self._final_errors = []    # error (SSE, MSE) of best trial of 
                                   #   each training method
        self._final_L2_norms = []  # L2-norm of best trial of each train
        self._best_epochs = []     # epochs of best trial of each method
        self._ready: bool = False  # flag indicating successful training

        self._silent: bool = False
        
        plt.rcParams.update({'font.size': 14})
        plt.rcParams['legend.fontsize'] = 14            # fonts in plots

        self._metrics: Dict[str, Any] = {  # see metrics.init_metrics()
            'abs': np.inf, 
            'activation': None, 
            'i_abs': -1, 
            'epochs': -1,
            'L2': np.inf,  # L2-norm || phi(x_vld) - y_vld || 
            'mse': np.inf, 
            'neurons': None,
            'trainer': None, 
            }

    def __call__(self, X: Float2D = None, Y: Float2D = None, x: Float2D = None, 
                 **kwargs: Any) -> Union[Dict[str, Any], Float2D]:
        """
        - Trains neural network if X is not None and Y is not None
        - Sets self.ready to True if training is successful
        - Predicts y for input x if x is not None and self.ready is True

        Args:
            X (2D array_like of float, optional, default: self.X):
                training input, shape: (n_point, n_inp) 
                shape: (n_point,) is tolerated
                
            Y (2D array_like of float, optional, default: self.Y):
                training target, shape: (n_point, n_out)
                shape: (n_point,) is tolerated

            x (2D array_like of float, optional, default: self.x):
                prediction input, shape: (n_point, n_inp)
                shape: (n_inp,) is tolerated

        Kwargs:
            keyword arguments, see: train() and predict()

        Returns:
            prediction of net(x) if x is not None and self.ready,
                shape: (n_point, n_out)
            or
            metrics of best training trial if X and Y are not None
                'trainer' (str): best training method
                'L2'    (float): sqrt{mean{(net(x)-Y)^2}} best train
                'abs'   (float): max{|net(x) - Y|} of best training
                'i_abs'   (int): index of Y where abs. error is max.
                'epochs'  (int): number of epochs of best training
            or
            empty metrix 
                if X, Y and x are None 
                if x is not None and not self.ready

        Note:
            - Shape of X, Y and x is corrected to (n_point, n_inp/n_out)
            - References to X, Y, x and y are stored as self.X, self.Y,
              self.x, self.y, see self.train() and self.predict()
        """
        if X is not None and Y is not None:
            self.metrics = self.train(X=X, Y=Y, **kwargs)
            
        if x is not None and self.ready:
            return self.predict(x=x, **kwargs)

        return self.metrics

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

    @metrics.setter
    def metrics(self, value):
        self._metrics = value

    def import_dataframe(self, df: pd.DataFrame, x_keys: Sequence[str],
                         y_keys: Sequence[str]) -> None:
        """
        Imports training input X and training target Y. self.Y is the 
        normalized target after import, but 'df' stays unchanged

        Args:
            df:
                data object

            x_keys:
                input  keys for data selection

            y_keys:
                output keys for data selection
        """
        self._x_keys = np.atleast_1d(x_keys)
        self._y_keys = np.atleast_1d(y_keys)
        assert all(k in df for k in x_keys), "unknown x-keys: '" + \
            str(x_keys) + "', valid keys: '" + df.columns + "'"
        assert all(k in df for k in y_keys), "unknown y-keys: '" + \
            str(y_keys) + "', valid keys: '" + df.columns + "'"
        self._X = np.asfarray(df.loc[:, x_keys])
        self._Y = np.asfarray(df.loc[:, y_keys])

        self._norm_y = nl.tool.Norm(self._Y)
        self._Y = self._norm_y(self._Y)

    def set_XY(self, X: Float2D, Y: Float2D,
                   x_keys: Str1D = None, y_keys: Str1D = None) -> None:
        """
        - Imports training input X and training target Y
        - converts X and Y to 2D arrays
        - normalizes training target (self.Y is then normalized target,
          but argument 'Y' stays unchanged)

        Args:
            X (2D array_like of float):
                training input, shape: (n_point, n_inp)
                shape: (n_point,) is tolerated

            Y (2D array_like of float):
                training target, shape: (n_point, n_out)
                shape: (n_point,) is tolerated

            x_keys:
                list of column keys for data selection
                use self._x_keys keys if xKeys is None
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
            
#        print('neu297', self._X, self._Y,self._Y.min(), self._Y.max())
# TODO .            
#        assert not np.isclose(self._Y.max(), self._Y).min(), str(self._Y)

        if x_keys is None:
            self._x_keys = ['x' + str(i) for i in range(self._X.shape[1])]
        else:
            self._x_keys = x_keys
        if y_keys is None:
            self._y_keys = ['y' + str(i) for i in range(self._Y.shape[1])]
        else:
            self._y_keys = y_keys

        self._norm_y = nl.tool.Norm(self._Y)
        self._Y = self._norm_y(self._Y)

    def plot(self) -> None:
        self._plot_test_with_train_data()

    def _plot_test_with_train_data(self) -> None:
        for trainer, error, epochs in zip(self._trainers, self._final_errors,
                                          self._best_epochs):
            y = self.predict(x=self._X)                  # y: prediction
            Y = self._norm_y.renorm(self._Y)                 # Y: target

            title = 'Train (' + trainer + ') mse: ' + \
                str(round(error * 1e3, 2)) + 'e-3 [' + str(epochs) + ']'

            plt.title(title)
            for j, y_train_sub in enumerate(Y.T):
                dy = np.subtract(y.T[j], y_train_sub)
                for i, x_train_sub in enumerate(self._X.T):
                    label = self._x_keys[i] + ' & ' + self._y_keys[j]
                    plt.plot(x_train_sub, dy, label=label)
            plt.xlabel('$x$')
            plt.ylabel('$y_{pred} - y_{train}$')
            plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
            plt.show()

            plt.title(title)
            for j, y_train_sub in enumerate(Y.T):
                for i, x_train_sub in enumerate(self._X.T):
                    label = self._x_keys[i] + ' & ' + self._y_keys[j]
                    plt.plot(x_train_sub, y.T[j], label=label)
                    plt.plot(x_train_sub, y_train_sub, label=label +
                             ' (target)', linestyle='', marker='*')
            plt.xlabel('$x$')
            plt.ylabel('$y$')
            plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
            plt.show()

        x = range(len(self._final_errors))
        y = self._final_errors
        y2 = np.asfarray(self._best_epochs) * 1e-5
        f = plt.figure()
        ax = f.add_axes([.1, .1, .8, .8])
        ax.plot(np.asfarray(x)+0.01, y2, color='b', label='epochs*1e-5')
        ax.bar(x, y, align='center', color='r', label='MSE')
        ax.set_xticks(x)
        ax.set_xticklabels(self._trainers)
        ax.set_yticks(np.add(y, y2))
        plt.title('Final training errors')
        plt.xlabel('trainer')
        plt.ylabel('error')
        plt.yscale('log', nonposy='clip')
        plt.grid()
        plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
        plt.show()


class Neural(NeuralBase):
    """
    - Wraps neural network implementations from
        - Neurolab: trains exclusively with backpropagation
        - NeuralGenetic: trains exclusively with genetic algorithm

    - Compares training algorithms and regularisation settings

    - Presents graphically history of norms for each trial of training

    Example of training and prediction of neural network in
        - compact form:
              y = Neural()(X=X, Y=Y, x=x, neurons=[6])
        - expanded form:
              phi = Neural()
              metrics = phi(X=X, Y=Y, neurons=[6])
              y = phi(x=x)
              L2_norm = metrics['L2']  # or: submodel.metrics['L2']

    Major methods and attributes (return type in the comment):
        - y = Neural()(X=None, Y=None, x=None, **kwargs) 
                                             # y.shape: (n_point, n_out)
        - metrics = self.train(X, Y,**kwargs)         # see self.metrics
        - y = self.predict(x, **kwargs)      # y.shape: (n_point, n_out)
        - self.ready                                              # bool
        - self.metrics                        # dict{str: float/str/int}
        - self.plot()

    References:
        - Recommended training algorithms:
              'rprop': resilient backpropagation (NO REGULARIZATION)
                       wikipedia: 'Rprop'
              'bfgs':  Broyden–Fletcher–Goldfarb–Shanno algorithm,
                       see: scipy.optimize.fmin_bfgs() and wikipedia: 
                           'Broyden-Fletcher-Goldfarb-Shanno_algorithm'
        - http://neupy.com/docs/tutorials.html#tutorials
    """

    def __init__(self, f: Function = None) -> None:
        """
        Args:
            f:
                theor. submodel as method f(self, x) or function f(x)

        Note: if f is not None, genetic training or training with
            derivative dE/dy and dE/dw is employed
        """
        super().__init__(f)

    def train(self, X: Float2D = None, Y: Float2D = None, 
              **kwargs: Any) -> Dict[str, Any]:
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
            activation (str or function):
                activation function of hidden layers, 
                usually 'tansig': TanSig(): tanh(x) or 
                        'logsig': LogSig(): 1 / (1 + exp(-z))
                default: 'tansig'

            alpha (float):
                factor for autodefinition of number of hidden neurons,
                see: propose_hidden_neurons()
                default: 2.0

            epochs (int):
                max number of iterations of single trial
                default: 1000

            errorf (function or str)
                error function: (nl.error.MSE() nl.error.MAE(), )
                default: 'mse'

            goal (float):
                limit of 'errorf' for stop of training (0. < goal < 1.)
                default: 1e-3
                [note: MSE=1e-6 corresponds to L2-norm=1e-3]

            trainer (str or list of str):
                if string, then space sep. string is converted to list
                if 'all' or None, then all training methods are assigned
                default: 'auto' ==> ['rprop', 'bfgs']

            neurons (int or array_like of int):
                array of number of neurons in hidden layers
                default: [] ==> use estimate of propose_hidden_neurons()

            output (str or function):
                activation function of output layer
                usually 'tanh' or 'tansig': TanSig(): tanh(x) or 
                        'logsig': LogSig(): 1 / (1 + exp(-z))
                default: 'tanh'

            plot (int):
                controls frequency of plotting progress of training
                default: 0 (no plot)

            regularization (float):
                regularization rate (sum of all weights is added to
                cost function of training, 0. <= regularization <= 1.
                default: 0. (no effect of sum of all weights)
                [same as 'rr']
                [note: neurolab trainer 'bfgs' ignores 'rr' argument]

            rr (float):
                [same as 'regularization']

            show (int):
                control of information about training, if show>0: print
                default: epochs // 10
                [argument 'show' superseds 'silent' if show > 0]

            silent (bool):
                if True then no information is sent to console
                default: self.silent
                [argument 'show' superseds 'silent' if show > 0]

            smart_trials (bool):
                if False, perform all trials even if goal was reached
                default: True

            trials (int):
                maximum number of training trials
                default: 3

        Returns:
            metrics of best training trial:
            'trainer' (str): best training method
            'L2'    (float): sqrt{sum{(net(x)-Y)^2}/N} of best train
            'abs'   (float): max{|net(x) - Y|} of best training
            'i_abs'   (int): index of Y where abs. error is maximum
            'epochs'  (int): number of epochs of best training

        Note:
            - If training fails, then self.metrics['trainer'] is None
            - Reference to optional theor. submodel is stored as self.f
            - Reference to training data is stored as self.X and self.Y
            - The best network is assigned to 'self._net'
        """
        if X is not None and Y is not None:
            self.set_XY(X, Y)
            
        assert self._X is not None and self._Y is not None, \
            str(self.X) + ' ' + str(self.Y)

        alpha = kwargs.get('alpha', 2.)
        epochs = kwargs.get('epochs', 1000)
        errorf = kwargs.get('errorf', nl.error.MSE())
        if isinstance(errorf, str):
            if errorf.lower() == 'mse':
                errorf = nl.error.MSE()
            elif errorf.lower() == 'sse':
                errorf = nl.error.SSE()
            elif errorf.lower() == 'sae':
                errorf = nl.error.SAE()
            elif errorf.lower() == 'mae':
                errorf = nl.error.MAE()
            else:
                errorf = nl.error.MSE()
        goal = kwargs.get('goal', 1e-3)
        trainer = kwargs.get('trainer', ['bfgs', 'rprop'])
        neurons = kwargs.get('neurons', None)
        outputf = kwargs.get('output', 'tanh')
        if isinstance(outputf, str):
            if outputf.lower() in ('tansig', 'tanh', ):
                outputf = nl.trans.TanSig
            elif outputf.lower() in ('lin', 'linear', 'purelin', ):
                outputf = nl.trans.PureLin
            elif outputf.lower() in ('logsig', 'sigmoid', ):
                outputf = nl.trans.LogSig
            else:
                outputf = nl.trans.TanSig
        assert outputf in (nl.trans.TanSig, nl.trans.LogSig, nl.trans.PureLin)
        plot         = kwargs.get('plot', 1)
        rr           = kwargs.get('regularization', None)
        if rr is None:
            rr       = kwargs.get('rr', None)
        if rr is None:
            rr = 1.
        show         = kwargs.get('show', 0)
        self.silent  = kwargs.get('silent', self.silent)
        if show is not None and show > 0:
            self.silent = False
        smart_trials = kwargs.get('smart_trials', True)
        transf       = kwargs.get('activation', 'tanh')
        if isinstance(transf, str):
            if transf.lower() in ('tansig', 'tanh', ):
                transf = nl.trans.TanSig
            elif transf.lower() in ('lin', 'linear', 'purelin', ):
                transf = nl.trans.PureLin
            else:
                transf = nl.trans.LogSig
        assert transf in (nl.trans.TanSig, nl.trans.LogSig, nl.trans.PureLin)
        trials         = kwargs.get('trials', 3)

        if self.silent:
            show = 0
            plot = 0


        if errorf is None:
            errorf = nl.error.MSE()
        if show is None:
            show = epochs // 10
        if self.silent:
            plot = False

        if neurons is None:
            neurons = [1]
        if isinstance(neurons, (int, float)):
            neurons = list([int(neurons)])

        print('neurons', neurons)

        if not neurons or len(neurons) == 0 or not all(neurons):
            neurons = propose_hidden_neurons(X=self._X, Y=self._Y, 
                                             alpha=alpha,
                                             silent=self.silent)
        print('2neurons', neurons)

        if not isinstance(neurons, list):
            neurons = list(neurons)
            
        print('neurons', neurons)
            
        assert all(nrn > 0 for nrn in np.array(neurons).reshape(-1))

        size = neurons.copy()
        size.append(self._Y.shape[1])
        assert size[-1] == self._Y.shape[1]

        trainer_pool = {'genetic':    nl.train.train_bfgs,      # TODO .
                        'derivative': nl.train.train_bfgs,      # TODO .
                        'bfgs':       nl.train.train_bfgs,
                        'cg':         nl.train.train_cg,
                        'gd':         nl.train.train_gd,
                        'gda':        nl.train.train_gda,
                        'gdm':        nl.train.train_gdm,
                        'gdx':        nl.train.train_gdx,
                        'rprop':      nl.train.train_rprop
                        }

        default_trainers = ['rprop', 'bfgs']
        assert all([tr in trainer_pool for tr in default_trainers])

        if self.f is not None:
            # alternative training if theoretical submodel 'f'
            trainer = [x for x in trainer if x in ('genetic', 'derivative')]
            if not trainer:
                trainer = 'genetic'
        else:
            if not trainer:
                trainer = 'all'
            if isinstance(trainer, str):
                if trainer == 'all':
                    trainer = ['cg', 'gd', 'gdx', 'gdm', 'gda', 'rprop', 
                               'bfgs', 'genetic']
                if trainer == 'auto':
                    trainer = default_trainers
            trainer = list(OrderedDict.fromkeys(trainer))   # redundancy
        trainer = [trn.lower() for trn in trainer if trn in trainer_pool]
        self._trainers = trainer if trainer else default_trainers 
        
        if not self.silent:
            print('+++ trainers:', self._trainers)


        self._ready = True # predict() returns None if self._ready is False

        self._net = None
        sequence_error = np.inf
        best_trainer = self._trainers[0]
        self._final_errors, self._final_L2norms, self._best_epochs = [], [], []

        for trainer in self._trainers:
            trainf = trainer_pool[trainer]
            trainer_err = np.inf
            trainer_epochs = None
            trainer_l2norm = np.inf

            margin = 0.0
            minmax = [[x[0] - margin*(x[1] - x[0]), 
                       x[1] + margin*(x[1] - x[0])]
                                              for x in nl.tool.minmax(self._X)]
            net = nl.net.newff(minmax, size)
            net.transf = transf
            net.trainf = trainf
            net.errorf = errorf
            net.f = self.f
            
            if self.f is not None:
                net.outputf = nl.trans.PureLin
            else:
                net.outputf = outputf

            for j_trial in range(trials):
                if trainer in ('genetic', ):
                    net.init()
                    trial_errors = net.train(self._X, self._Y, f=self.f,
                                             epochs=epochs, goal=goal, rr=rr,
                                             show=show)
                elif trainer == 'rprop':
                    net.init()
                    trial_errors = net.train(self._X, self._Y, epochs=epochs,
                                             show=show, goal=goal)
                else:
                    for i_repeat in range(3):
                        del net
                        net = nl.net.newff(nl.tool.minmax(self._X), size)
                        net.init()
                        trial_errors = net.train(self._X, self._Y, 
                                                 epochs=epochs,
                                                 show=show, goal=goal, rr=rr)
                        if i_repeat > 0:
                            print('!!! Neural, L697, i_repeat:', i_repeat,
                                  'trainer:', trainer,'trial:', j_trial)
                        if len(trial_errors) >= 1:
                            break
                if len(trial_errors) < 1:
                    trial_errors.append(np.inf)
                
                if sequence_error > trial_errors[-1]:
                    sequence_error = trial_errors[-1]
                    del self._net
                    self._net = net.copy()
                if (trainer_err < goal and trainer_epochs > len(trial_errors))\
                   or (trainer_err >= goal and trainer_err > trial_errors[-1]):
                    trainer_err = trial_errors[-1]
                    trainer_epochs = len(trial_errors)
                    trainer_l2norm = np.sqrt(np.mean(np.square(
                      self.predict(x=self._X) - self._norm_y.renorm(self._Y))))
                if plot:
                    plt.plot(range(len(trial_errors)), trial_errors,
                             label='trial: ' + str(j_trial))
                if smart_trials:
                    if trial_errors[-1] < goal:
                        break

            self._final_errors.append(trainer_err)
            self._final_L2norms.append(trainer_l2norm)
            self._best_epochs.append(trainer_epochs)
            i_best = self._trainers.index(best_trainer)
            if trainer_err < self._final_errors[i_best]:
                best_trainer = trainer

            if plot:
                plt.title("'" + trainer + "' mse:" +
                          str(round(trainer_err*1e3, 2)) + 'e-3 L2:' +
                          str(round(trainer_l2norm, 3)) +
                          ' [' + str(trainer_epochs) + ']')
                plt.xlabel('epochs')
                plt.ylabel('error')
                plt.yscale('log', nonposy='clip')
                plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
                plt.grid()
                plt.show()
            if not self.silent:
                print('    ' + trainer + ':' + str(round(trainer_err, 5)) +
                      '[' + str(trainer_epochs) + '], ')

        if self._net is None:
            if not self.silent:
                print('??? All training trials failed')
            self._ready = False
            return self.metrics

        if plot:
            self._plot_test_with_train_data()

        i_best = self._trainers.index(best_trainer)
        if not self.silent:
            if len(self._trainers) > 1:
                print("    best trainer: '" + self._trainers[i_best] +
                      "' out of: [" + ' '.join(self._trainers) +
                      '], error:', round(self._final_errors[i_best], 5))
                if len(self._final_errors) > 1:
                    print("    (trainer:err): [", end='')
                    s = ''
                    for trn, err in zip(self._trainers, self._final_errors):
                        s += trn + ':' + str(round(err, 5)) + ' '
                    print(s[:-2] + ']')

        Y = self._norm_y.renorm(self._Y)
        y = self.predict(self._X)
        dy = y - Y
        i_abs_max = np.abs(dy).argmax()
        self._metrics = {
                         'abs': Y.ravel()[i_abs_max],
                         'i_abs': i_abs_max,
                         'epochs': self._best_epochs[i_best],
                         'L2': np.sqrt(np.mean(np.square(dy))),
                         'ready': self.ready,
                         'trainer': self._trainers[i_best],
                         }

        return self.metrics

    def predict(self, x: Float2D, **kwargs: Any) -> Float2D:
        """
        Executes network, stores x as self.x

        Args:
            x:
                prediction input, shape: (n_point, n_inp)
                shape: (n_inp,) is tolerated

        Kwargs:
            silent (bool):
                if True, then no printing
                default: self.silent

        Returns:
            prediction y = net(x)
            or
            None if x is None or not self.ready or not self._net

        Note:
            - Shape of x is corrected to: (n_point, n_inp)
            - Input x and output net(x) are stored as self.x and self.y
        """
        self.silent = kwargs.get('silent', self.silent)

        if not self._net or not self.ready:
            if not self.silent:
                print('!!! Neural model.ready is False ==> returned y is None')
            self._y = None
        elif x is None:
            if not self.silent:
                print('!!! Neural x is None ==> returned y is None')
            self._y = None
        else:
            x = np.asfarray(x)
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            if x.shape[1] != self._net.ci:
                x = np.transpose(x)
            self._x = x    
            self._y = self._net.sim(x)
            self._y = self._norm_y.renorm(self._y)

        return self._y
