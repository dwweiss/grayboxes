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
      2019-03-20 DWW

  Acknowledgements:
      Neurolab is a contribution by E. Zuev (pypi.python.org/pypi/neurolab)
"""

__all__ = ['Neural', 'propose_hidden_neurons']

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


def propose_hidden_neurons(X: np.ndarray, Y: np.ndarray, alpha: float=2,
                           silent: bool=False) -> List[int]:
    """
    Proposes optimal number of hidden neurons for given training data set

                   n_point
    hidden = ---------------------        with: 2 <= alpha <= 10
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
    hidden = [n_hidden]

    return hidden


class Neural(object):
    """
    a) Wraps different neural network implementations from
        - Neurolab: trains exclusively with backpropagation
        - NeuralGenetic: trains exclusively with genetic algorithm

    b) Compares training algorithms and regularisation settings

    c) Presents graphically history of norms for each trial of training

    Example of training and prediction of neural network in
        - compact form:
              y = Neural()(X=X, Y=Y, x=x, neurons=[6])
        - expanded form:
              net = Neural()
              best = net(X=X, Y=Y, neurons=[6])
              y = net(x=x)
              L2_norm = best['L2']  # or: net.best['L2']

    Major methods and attributes (return type in the comment):
        - y = Neural()(X=None, Y=None, x=None, **kwargs) 
                                               # y.shape:(n_point,n_out)
        - best = self.train(X, Y,**kwargs)               # see self.best
        - y = self.predict(x, **kwargs)      # y.shape: (n_point, n_out)
        - self.ready                                              # bool
        - self.best                           # dict{str: float/str/int}
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

    def __init__(self, f: Optional[Callable]=None) -> None:
        """
        Args:
            f:
                theor. submodel as method f(self, x) or function f(x)

        Note: if f is not None, genetic training or training with
            derivative dE/dy and dE/dw is employed
        """
        self.f = f               # theor. submodel for single data point

        self._net = None         # network
        self._X = None           # input of training
        self._Y = None           # target
        self._x = None           # input of prediction
        self._y = None           # prediction y = net(x)
        self._norm_y = None      # data from normalization of target
        self._x_keys = None      # xKeys for import from data frame
        self._y_keys = None      # yKeys for import from data frame
        self._methods = ''       # list of training algorithms
        self._final_errors = []  # error (SSE, MSE) of best trial of 
                                 # each method
        self._finalL2norms = []  # L2-norm of best trial of each method
        self._best_epochs = []   # epochs of best trial of each method
        self._ready = False      # flag indicating successful training

        self._silent = False
        plt.rcParams.update({'font.size': 14})
        plt.rcParams['legend.fontsize'] = 14            # fonts in plots

        self._best = {'method': None, 'L2': np.inf, 'abs': np.inf,
                      'iAbs': -1, 'epochs': -1}   # result of best trial

    def __call__(self, X: Optional[np.ndarray]=None, 
                 Y: Optional[np.ndarray]=None,
                 x: Optional[np.ndarray]=None, 
                 **kwargs: Any) -> Union[np.ndarray, Dict[str, Any], None]:
        """
        - Trains neural network if X is not None and Y is not None
        - Sets self.ready to True if training is successful
        - Predicts y for input x if x is not None and self.ready is True

        Args:
            X (2D or 1D array_like of float, optional, default: self.X):
                training input, shape: (n_point, n_inp) or (n_point,)
                
            Y (2D or 1D array_like of float, optional, default: self.Y):
                training target, shape: (n_point, n_out) or (n_point,)

            x (2D or 1D array_like of float, optional, default: self.x):
                prediction input, shape: (n_point, n_inp) or (n_inp,)

        Kwargs:
            keyword arguments, see: train() and predict()

        Returns:
            (2D array of float):
                prediction of net(x) if x is not None and self.ready
            or
            (dictionary):
                result of best training trial if X and Y are not None
                    'method' (str): best method
                    'L2'   (float): sqrt{sum{(net(x)-Y)^2}/N} best train
                    'abs'  (float): max{|net(x) - Y|} of best training
                    'iAbs'   (int): index of Y where abs. error is max.
                    'epochs' (int): number of epochs of best training
            or
            (None):
                if (X, Y and x are None) or not self.ready

        Note:
            - Shape of X, Y and x is corrected to: (n_point,n_inp/n_out)
            - References to X, Y, x and y are stored as self.X, self.Y,
              self.x, self.y, see self.train() and self.predict()
        """
        if X is not None and Y is not None:
            best = self.train(X=X, Y=Y, **kwargs)
        else:
            best = None
        if x is not None:
            return self.predict(x=x, **kwargs)
        return best

    @property
    def f(self) -> Callable:
        return self._f

    @f.setter
    def f(self, value: Callable) -> None:
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
    def X(self) -> np.ndarray:
        return self._X

    @property
    def Y(self) -> np.ndarray:
        return self._Y

    @property
    def x(self) -> np.ndarray:
        return self._x

    @property
    def y(self) -> np.ndarray:
        return self._y

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def best(self) -> Dict[str, Any]:
        """
        Returns:
            results for best training trial
            [see self.train()]
        """
        return self._best

    def import_dataframe(self, df: pd.DataFrame, x_keys: Sequence[str],
                         y_keys: Sequence[str]) -> None:
        """
        Imports training input X and training target Y.  self.Y is the 
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

    def set_arrays(self, X: np.ndarray, Y: np.ndarray,
                   x_keys: Optional[Sequence[str]]=None,
                   y_keys: Optional[Sequence[str]]=None) -> None:
        """
        - Imports training input X and training target Y
        - converts X and Y to 2D arrays
        - normalizes training target (self.Y is then normalized target,
          but argument 'Y' stays unchanged)

        Args:
            X (2D or 1D array_like of float):
                training input, shape: (n_point, n_inp) or (n_point,)

            Y (2D or 1D array_like of float):
                training target, shape: (n_point, n_out) or (n_point,)

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
        assert not np.isclose(self._Y.max(), self._Y.min()), str(self._Y)

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

    def train(self, X: Optional[np.ndarray]=None, Y: Optional[np.ndarray]=None,
              **kwargs: Any) -> Dict[str, Any]:
        """
        Trains model, stores X and Y as self.X and self.Y, and stores 
        result of best training trial as self.best

        Args:
            X (2D or 1D array of float):
                training input, shape: (n_point, n_inp) or (n_point,)
                default: self.X

            Y (2D or 1D array of float):
                training target, shape: (n_point, n_out) or (n_point,)
                default: self.Y

        Kwargs:
            alpha (float):
                factor for autodefinition of number of hidden neurons,
                see: propose_hidden_neurons()
                default: 2.0

            epochs (int):
                max number of iterations of single trial
                default: 1000

            errorf (function)
                error function: (nl.error.MSE() or nl.error.SSE())
                default: MSE

            goal (float):
                limit of 'errorf' for stop of training (0. < goal < 1.)
                default: 1e-3
                [note: MSE of 1e-3 corresponds to L2-norm of 1e-6]

            methods (str or list of str):
                if string, then space sep. string is converted to list
                if 'all' or None, then all training methods are assigned
                default: 'bfgs'

            method (str or list of str):
                [same as 'methods']

            neurons (int or array_like of int):
                array of number of neurons in hidden layers
                default: [] ==> use estimate of propose_hidden_neurons()

            outputf (function):
                activation function of output layer
                default: TanSig()

            plot (int):
                controls frequency of plotting progress of training
                default: 0 (no plot)

            regularization (float):
                regularization rate (sum of all weights is added to
                cost function of training, 0. <= regularization <= 1.
                default: 0. (no effect of sum of all weights)
                [same as 'rr']

            rr (float):
                [same as 'regularization']

            show (int):
                control of information about training, if show=0: no print
                default: epochs // 10
                [argument 'show' superseds 'silent' if show > 0]

            silent (bool):
                if True then no information is sent to console
                default: self.silent
                [argument 'show' superseds 'silent' if show > 0]

            smart_trials (bool):
                if False, perform all trials even if goal has been reached
                default: True

            transf (function):
                activation function of hidden layers
                default: TanSig()

            trials (int):
                maximum number of training trials
                default: 3

        Returns:
            (dictionary)
                result of best training trial:
                'method' (str): best method
                'L2'   (float): sqrt{sum{(net(x)-Y)^2}/N} of best train
                'abs'  (float): max{|net(x) - Y|} of best training
                'iAbs'   (int): index of Y where abs. error is maximum
                'epochs' (int): number of epochs of best training

        Note:
            - If training fails then self.best['method']=None
            - Reference to optional theoretical submodel is stored as self.f
            - Reference to training data is stored as self.X and self.Y
            - The best network is assigned to 'self._net'
        """
        if X is not None and Y is not None:
            self.set_arrays(X, Y)
        assert self._X is not None and self._Y is not None, \
            str(self.X) + ' ' + str(self.Y)

        alpha        = kwargs.get('alpha',          2.)
        epochs       = kwargs.get('epochs',         1000)
        errorf       = kwargs.get('errorf',         nl.error.MSE())
        goal         = kwargs.get('goal',           1e-3)
        methods      = kwargs.get('methods',        None)
        if methods is None:
            methods  = kwargs.get('method',         'bfgs rprop')
        neurons      = kwargs.get('neurons',        None)
        outputf      = kwargs.get('outputf',        nl.trans.TanSig())
        plot         = kwargs.get('plot',           1)
        rr           = kwargs.get('regularization', None)
        if rr is None:
            rr       = kwargs.get('rr',             None)
        if rr is None:
            rr = 1.
        show         = kwargs.get('show',           0)
        self.silent  = kwargs.get('silent',         self.silent)
        if show is not None and show > 0:
            self.silent = False
        smart_trials = kwargs.get('smart_trials',   True)
        transf       = kwargs.get('transf',         nl.trans.TanSig())
        trials       = kwargs.get('trials',         3)

        if self.silent:
            show = 0
            plot = 0

        self._ready = False

        # alternative training if theoretical submodel 'f' is provided
        if self.f is not None:
            methods = [x for x in methods if x in ('genetic', 'derivative')]
            if not methods:
                methods = 'genetic'
        else:
            if not methods:
                methods = 'all'
            if isinstance(methods, str):
                if methods == 'all':
                    methods = 'cg gd gdx gdm gda rprop bfgs genetic'
                methods = methods.split()
            methods = list(OrderedDict.fromkeys(methods))   # redundancy
        self._methods = [x.lower() for x in methods]

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
        if not neurons or len(neurons) == 0 or not all(neurons):
            neurons = propose_hidden_neurons(X=self._X, Y=self._Y, alpha=alpha,
                                             silent=self.silent)
        if not isinstance(neurons, list):
            neurons = list(neurons)
        assert all(x > 0 for x in neurons), str(neurons)

        size = neurons.copy()
        size.append(self._Y.shape[1])
        assert size[-1] == self._Y.shape[1]

        trainf_dict = {'genetic':    nl.train.train_bfgs,   # TODO .
                       'derivative': nl.train.train_bfgs,   # TODO .
                       'bfgs':       nl.train.train_bfgs,
                       'cg':         nl.train.train_cg,
                       'gd':         nl.train.train_gd,
                       'gda':        nl.train.train_gda,
                       'gdm':        nl.train.train_gdm,
                       'gdx':        nl.train.train_gdx,
                       'rprop':      nl.train.train_rprop
                       }

        assert all([x in trainf_dict for x in self._methods]), \
            str(self._methods)
        if not self.silent:
            print('+++ methods:', self._methods)

        sequence_error = np.inf
        best_method = self._methods[0]
        self._final_errors, self._finalL2norms, self._best_epochs = [], [], []

        for method in self._methods:
            trainf = trainf_dict[method]
            method_err = np.inf
            method_epochs = None
            method_l2norm = None

            net = nl.net.newff(nl.tool.minmax(self._X), size)
            net.transf = transf
            net.outputf = outputf
            net.trainf = trainf
            net.errorf = errorf

            net.f = self.f
            if self.f is not None:
                net.outputf = nl.trans.PureLin

            for j_trial in range(trials):
                if method in ('genetic'):
                    net.init()
                    trial_errors = net.train(self._X, self._Y, f=self.f,
                                             epochs=epochs, goal=goal, rr=rr,
                                             show=show)
                elif method == 'rprop':
                    net.init()
                    trial_errors = net.train(self._X, self._Y, epochs=epochs,
                                             show=show, goal=goal)
                else:
                    net.init()
                    trial_errors = net.train(self._X, self._Y, epochs=epochs,
                                             show=show, goal=goal, rr=rr)
                assert len(trial_errors) >= 1, '\nte:'+str(trial_errors)+'\n'+\
                    str(self._X) + '\n' + str(self._Y) + '\n' + str(self.f)
                if sequence_error > trial_errors[-1]:
                    sequence_error = trial_errors[-1]
                    del self._net
                    self._net = net.copy()
                if (method_err < goal and method_epochs > len(trial_errors)) \
                   or (method_err >= goal and method_err > trial_errors[-1]):
                    method_err = trial_errors[-1]
                    method_epochs = len(trial_errors)
                    method_l2norm = np.sqrt(np.mean(np.square(
                      self.predict(x=self._X) - self._norm_y.renorm(self._Y))))
                if plot:
                    plt.plot(range(len(trial_errors)), trial_errors,
                             label='trial: ' + str(j_trial))
                if smart_trials:
                    if trial_errors[-1] < goal:
                        break

            self._final_errors.append(method_err)
            self._finalL2norms.append(method_l2norm)
            self._best_epochs.append(method_epochs)
            i_best = self._methods.index(best_method)
            if method_err < self._final_errors[i_best]:
                best_method = method

            if plot:
                plt.title("'" + method + "' mse:" +
                          str(round(method_err*1e3, 2)) + 'e-3 L2:' +
                          str(round(method_l2norm, 3)) +
                          ' [' + str(method_epochs) + ']')
                plt.xlabel('epochs')
                plt.ylabel('error')
                plt.yscale('log', nonposy='clip')
                plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
                plt.grid()
                plt.show()
            if not self.silent:
                print('    ' + method + ':' + str(round(method_err, 5)) +
                      '[' + str(method_epochs) + '], ')
        if plot:
            self._plot_test_with_train_data()

        i_best = self._methods.index(best_method)
        if not self.silent:
            if len(self._methods) > 1:
                print("    best method: '" + self._methods[i_best] +
                      "' out of: [" + ' '.join(self._methods) +
                      '], error:', round(self._final_errors[i_best], 5))
                if len(self._final_errors) > 1:
                    print("    (method:err): [", end='')
                    s = ''
                    for method, err in zip(self._methods, self._final_errors):
                        s += method + ':' + str(round(err, 5)) + ' '
                    print(s[:-2] + ']')

        self._ready = True

        # assign results of best trial to return value
        Y = self._norm_y.renorm(self._Y)
        dy = self.predict(self._X) - Y
        i_abs_max = np.abs(dy).argmax()
        self._best = {'method': self._methods[i_best],
                      'L2': np.sqrt(np.mean(np.square(dy))),
                      'abs': Y.ravel()[i_abs_max],
                      'iAbs': i_abs_max,
                      'epochs': self._best_epochs[i_best]}
        return self.best

    def predict(self, x: np.ndarray, **kwargs: Any) -> Optional[np.ndarray]:
        """
        Executes network, stores x as self.x

        Args:
            x (2D or 1D array_like of float):
                prediction input, shape: (n_point, n_inp) or (n_inp,)

        Kwargs:
            silent (bool):
                if True then no printing
                default: self.silent

        Returns:
            (2D array of float):
                prediction y = net(x) if x is not None
            or
            (None):
                if x is None

        Note:
            - Shape of x is corrected to: (n_point, n_inp)
            - Input x and output net(x) are stored as self.x and self.y
        """
        if x is None:
            return None

        self.silent = kwargs.get('silent', self.silent)

        x = np.asfarray(x)
        if x.ndim == 1:
            x = x.reshape(x.size, 1)
        if x.shape[1] != self._net.ci:
            x = np.transpose(x)
        self._x = x

        self._y = self._net.sim(x)
        self._y = self._norm_y.renorm(self._y)

        return self._y

    def plot(self) -> None:
        self._plot_test_with_train_data()

    def _plot_test_with_train_data(self) -> None:
        for method, error, epochs in zip(self._methods, self._final_errors,
                                         self._best_epochs):
            y = self.predict(x=self._X)        # prediction
            Y = self._norm_y.renorm(self._Y)   # target

            title = 'Train (' + method + ') mse: ' + \
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
        ax.set_xticklabels(self._methods)
        ax.set_yticks(np.add(y, y2))
        plt.title('Final training errors')
        plt.xlabel('method')
        plt.ylabel('error')
        plt.yscale('log', nonposy='clip')
        plt.grid()
        plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
        plt.show()
