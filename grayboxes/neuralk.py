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
      2020-01-03 DWW
"""

__all__ = ['NeuralK']

from collections import OrderedDict
import inspect
import matplotlib.pyplot as plt
import numpy as np
import sys
try:
    dummy = tf
except:
    print("!!! Module 'tensorflow' not imported")
    print('    Set in Spyder: tools->Preferences->IPython console->')
    print("        Startup->'import tensorflow'")
    print('    or restart kernel and enter in IPython console:')
    print('        $ import tensorflow as tf')
    sys.exit()    

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adadelta, Adam, SGD, RMSprop
from tensorflow.keras.callbacks import Callback, EarlyStopping

from typing import Any, Callable, Dict, Iterable, List, Optional, Union

Float2D = Optional[np.ndarray]  # 2D array of float
Function = Optional[Callable]


class NeuralK(object):
    """
    - Wraps neural network implementations from Keras
    - Presents graphically history of training

    Example of training and prediction of neural network in
        - compact form:
              y = Neural()(X=X, Y=Y, x=x, neurons=[6, 4])
        - expanded form:
              phi = Neural()
              metrics = phi(X=X, Y=Y, neurons=[6, 4])
              y = phi(x=x)
              L2_norm = metrics['L2']            # or: phi.metrics['L2']

    Major methods and attributes (return type in the comment):
        - y = Neural()(X=None, Y=None, x=[[x0, x1, ...]], **kwargs) 
                                             # y.shape: (n_point, n_out)
        - metrics = self.train(X, Y,**kwargs)         # see self.metrics
        - y = self.predict(x, **kwargs)      # y.shape: (n_point, n_out)
        - self.ready                                              # bool
        - self.metrics                        # dict{str: float/str/int}
        - self.plot()

    Note:
        This class has not the connectivity functionality of the 
        box type models derived from class BoxModel.
        
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
        self.f = f                   # theor.submodel, single data point

        self._model = None           # model (neural net)

        self._metrics: Dict[str, Any] = self.init_metrics()
        
        self._X: Float2D = None      # train input,shape:(n_point,n_inp)
        self._Y: Float2D = None      # target, shape:(n_point, n_out)
        self._x: Float2D = None      # input of prediction
        self._y: Float2D = None      # prediction y = net(x, w)

        self._X_stats: List[Dict[str, float]] = []  # mean and std of 
                                     # every X column, shape: (n_inp,)
        self._Y_stats: List[Dict[str, float]] = []  # shape: (n_out,) 
        self._x_keys: List[str] = [] # x labels
        self._y_keys: List[str] = [] # y labels
        
        self._ready: bool = False    # if True, net is trained
        self._silent: bool = False
    
        plt.rcParams.update({'font.size': 12})
        plt.rcParams['legend.fontsize'] = 12            # fonts in plots

    def init_metrics(self) -> Dict[str, Any]:
        return OrderedDict({
                  'abs': np.inf, 
                  'i_abs': -1, 
                  'epochs': -1,
                  'L2': np.inf, 
                  'trainer': None, 
               })

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
    def model(self):
        return self._model
        
    @property
    def silent(self) -> bool:
        return self._silent

    @silent.setter
    def silent(self, value: bool) -> None:
        self._silent = value
        
    @property
    def metrics(self) -> Dict[str, Any]:
        """
        Returns:
            metrics of best training trial
            [see self.train()]
        """
        return self._metrics
    
    def __call__(self, X: Float2D = None, Y: Float2D = None, x: Float2D = None, 
                 **kwargs: Any) -> Union[Dict[str, Any], Float2D]:
        """
        - Trains neural network if X is not None and Y is not None
        - Sets self.ready to True if training is successful
        - Predicts y for input x if x is not None and self.ready is True
        - Plots history of training and comparison of train and test data

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
                'abs'   (float): max{|net(x) - Y|} of best training
                'i_abs'   (int): index of Y where abs. error is max.
                'epochs'  (int): number of epochs of best training
                'L2'    (float): sqrt{mean{(net(x)-Y)^2}} best train
                'trainer' (str): best training method
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
            self._metrics = self.train(X=X, Y=Y, **kwargs)
                        
        if x is not None and self.ready:
            return self.predict(x=x, **kwargs)

        return self.metrics

    def _normalize(self, X: Float2D, 
                   X_stats: List[Dict[str, float]]) -> Float2D:
        """
        Args:
            X:
                real-world 2D array of shape (n_point, n_inp/n_out)
                
            X_stats:
                list of dictionary of statistics of every column of X
                
        Returns: 
            normalized 2D array, shape: (n_point, n_inp/n_out)
        """
        X_normed = np.zeros(X.shape)
        for j in range(X.shape[1]):
            X_normed[:,j] = (X[:,j] - X_stats[j]['mean']) / X_stats[j]['std']
        return X_normed 

    def _denormalize(self, X: Float2D, 
                     X_stats: List[Dict[str, float]]) -> Float2D:
        """
        Args:
            X:
                normalized 2D array of shape (n_point, n_inp/n_out)
                
            X_stats:
                list of dictionary of statistics of every column of X
                
        Returns: 
            real-world 2D array, shape: (n_point, n_inp/n_out)
        """
        X_real_world = np.zeros(X.shape)
        for j in range(X.shape[1]):
            X_real_world[:,j] = X[:,j] * X_stats[j]['std'] + X_stats[j]['mean']
        return X_real_world 

    def set_XY(self, X: Float2D, Y: Float2D,
               x_keys: Optional[Iterable[str]] = None, 
               y_keys: Optional[Iterable[str]] = None) -> None:
        """
        - Imports training input X and training target Y
        - converts X and Y to 2D arrays
        - transposes X and Y if n_point < n_inp/n_out
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

        # mean and standard deviation of every column of X and Y
        self._X_stats = [{'mean': c.mean(), 'std': c.std()} for c in self._X.T]
        self._Y_stats = [{'mean': c.mean(), 'std': c.std()} for c in self._Y.T]

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

    def n_inp(self) -> Optional[int]:
        if not self.ready:
            return None

        n_inp_ = self.model.input[self.model.input.value_index].shape[0] 
        assert n_inp_ == len(self._X_stats), str((n_inp_, len(self._X_stats)))

        return n_inp_

    def train(self, X: Float2D, Y: Float2D, **kwargs: Any) -> Dict[str, Any]:
        """
        Trains model, stores X and Y as self.X and self.Y, and stores 
        result of best training trial as self.metrics

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
            activation (str):
                activation function of hidden layers, 
                    'elu': 
                    'tanh': TanSig(): tanh(x) or 
                    'sigmoid': LogSig(): 1 / (1 + exp(-z))
                default: 'tanh'

            epochs (int):
                max number of iterations of single trial
                default: 1000

            goal (float):
                limit of 'errorf' for stop of training (0. < goal < 1.)
                default: 1e-3
                [note: MSE of 1e-3 corresponds to L2-norm of 1e-6]

            neurons (array_like of int):
                array of number of neurons in hidden layers

            output (str):
                activation function of output layer
                default: 'linear'

            plot (int):
                controls frequency of plotting progress of training
                default: 0 (no plot)

            silent (bool):
                if True then no information is sent to console
                default: self.silent
                [argument 'show' superseds 'silent' if show > 0]

            trainer (str or list of str):
                if 'all' or None, then all training methods are assigned
                default: 'auto' ==> ['rmsprop', 'sgd']

            trials (int):
                maximum number of training trials
                default: 3

        Returns:
            metrics of best training trial:
            'abs'   (float): max{|net(x) - Y|} of best training
            'epochs'  (int): number of epochs of best training
            'i_abs'   (int): index of Y where abs. error is maximum
            'L2'    (float): sqrt{sum{(net(x)-Y)^2}/N} of best train
            'trainer' (str): best training method

        Note:
            - If training fails, then self.metrics['trainer'] is None
            - Reference to optional theor. submodel is stored as self.f
            - Reference to training data is stored as self.X and self.Y
            - The best network is assigned to 'self._model'
        """
        self.ready = False
                
        if X is not None and Y is not None:
            self.set_XY(X, Y)
        assert self._X is not None and self._Y is not None, \
            str(self.X) + ' ' + str(self.Y)
        n_inp, n_out = self.X.shape[1], self.Y.shape[1]
        assert self.X.shape[0] == self.Y.shape[0], \
            str((self.X.shape, self.Y.shape))
        X_normed = self._normalize(self.X, self._X_stats)
        Y_normed = self._normalize(self.Y, self._Y_stats)

        activation_hidden = np.atleast_1d(kwargs.get('activation', 
                                                     'elu')).tolist()
        activation_output = kwargs.get('output', 'linear')
        all_trainers = np.atleast_1d(kwargs.get('trainer', '')).tolist()
        if len(all_trainers) == 1 and all_trainers[0].lower() in ('auto', ''):
            all_trainers = ('rmsprop', 'sgd')
        epochs = kwargs.get('epochs', 1)
        goal = kwargs.get('goal', 1e-3)
        learning_rate = kwargs.get('learning_rate', None)
        momentum = kwargs.get('momentum', 0.)
        neurons = kwargs.get('neurons', [10, 8, 6])
        plot = kwargs.get('plot', False)
        patience = kwargs.get('patience', 30)
        trials = kwargs.get('trials', 1)
        verbose = kwargs.get('verbose', 0)
        
        # early stop
        class _PrintDot(Callback):
            def on_epoch_end(self, epochs, logs):
                if epochs == 0:
                    print('    ', end='')
                if epochs % 50 == 0:
                    print(str(epochs) + ' ', end='')

        early_stop = EarlyStopping(monitor='val_loss', patience=patience, 
                                   verbose=0, )  # baseline=goal)
        metrics_keys = ['mse', 'mae']
        best_net = {'model': None, 'mse': np.inf}
        all_histories: List[Dict[str, Any]] = []
        
        for trainer in all_trainers:
            trainer = trainer.lower()
                
            if trainer.startswith(('adad',)):
                optimizer = Adadelta()
            if trainer.startswith(('adam',)):
                optimizer = Adam(learning_rate=learning_rate) 
            elif trainer.startswith(('sgd',)):
                optimizer = SGD(learning_rate=learning_rate, 
                                momentum=momentum, nesterov=False)
            elif trainer.startswith(('rms',)):
                optimizer = RMSprop()
            else:
                print("!!! correct activation: '" + str(trainer) + "' ==> '" +\
                      "Adam'")
                optimizer = Adam(learning_rate=learning_rate)
            
            for activation in activation_hidden:
                self._model = Sequential()
                self.model.add(Input(shape=(n_inp,)))
                for nrn in np.atleast_1d(neurons):        
                    self.model.add(Dense(units=nrn, activation=activation))
                self.model.add(Dense(units=n_out, 
                                     activation=activation_output))

                for trial in range(trials):    
                    if not self.silent:
                        print('+++ compile model, trial:', trial)                    
                    self.model.compile(loss='mse', optimizer=optimizer,
                                       metrics=metrics_keys, verbose=verbose)

                    if not self.silent:
                        print('+++ fit, trainer:', trainer)
                    hist = self.model.fit(x=X_normed, y=Y_normed,
                        callbacks=[early_stop, _PrintDot()],
                        epochs=epochs, 
                        validation_split=0.2,
                        batch_size=50,
                        use_multiprocessing=True, 
                        verbose=verbose)
                    actual_history = {'trainer': trainer, 'trial': trial,
                                      'activation': activation}
                    for key in metrics_keys:
                        actual_history[key] = hist.history[key]
                    all_histories.append(actual_history)
    
                    mse = hist.history['mse'][-1]
                    if best_net['mse'] > mse:
                        best_net['mse'] =  mse
                        best_net['model'] = self.model
                        best_net['i_history'] = len(all_histories) - 1
                        best_net['trainer'] = trainer
                        best_net['trial'] = trial
                        best_net['activation'] = activation
    
                    self.plot_all_histories(best_net, all_histories, 
                                            metrics_keys, trials)
                                        
        self._model = best_net['model']
        loss_and_metrics = self.model.evaluate(X_normed, Y_normed, 
            batch_size=100, verbose=verbose)
        self._metrics['L2'] = loss_and_metrics
        self.ready = True
        
        print('+++ metrics, L2:', self._metrics['L2'])

        if plot:
            self.plot_all_histories(best_net, all_histories, 
                                    metrics_keys, trials)        
        return self.metrics
            
    def predict(self, x: Float2D, **kwargs: Any) -> Float2D:
        """
        reshapes real-world input x, executes network, 
        denormalizes dimensionless prediction y, stores x as self.x

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
            None if x is None or not self.ready or 
                x shape is incompatible

        Note:
            - Shape of x is corrected to: (n_point, n_inp)
            - Input x and output net(x) are stored as self.x and self.y
        """
        if x is None or not self.ready or self.n_inp() is None:
            self._x = None    
            self._y = None
        else:
            x = np.asfarray(x)
            if x.ndim == 1:
                x = x.reshape(x.size, 1)
            if x.shape[1] != self.n_inp():
                x = np.transpose(x)
            if x.shape[1] != self.n_inp():
                print('??? incompatible input shape:', x.shape)
                self._y = None
            else:
                self._x = x 
                x_normed = self._normalize(self.x, self._X_stats)
                y_normed = self.model.predict(x_normed, batch_size=1, 
                                              verbose=kwargs.get('verbose', 0))
                self._y = self._denormalize(y_normed, self._Y_stats)
                                       
            if kwargs.get('plot', False):            
                self.plot_train_vs_test()
                
        return self._y

    def plot(self) -> None:
        self.plot_train_vs_test()

    def plot_all_histories(self, best_net: Dict[str, Any], 
                           all_histories: List[Dict[str, Any]],
                           metrics_keys: List[str],
                           trials: int) -> None:
        plt.title('mse ' + str(round(1e3 * best_net.get('mse'), 3)) + \
                  'e-3, train ' + best_net.get('trainer', '?') + \
                  ', trial ' + str(best_net.get('trial', '?')) + \
                  ', act ' + str(best_net.get('activation', '?')[:3])
                  )
        for history in all_histories:
            for key in metrics_keys:
                s = str(history['trainer'])
                s += '/' + history['activation']
                if trials > 1:
                    s += '[' + str(history['trial']) + ']'
                if key == 'mse':
                    plt.plot(history[key], '-', label=s)
                else:
#                    plt.plot(history[key], ':')                        
                    pass
        plt.yscale('log')
#            plt.ylim([0, 0.5])
        plt.legend(bbox_to_anchor=(1.1, 1.05), loc='upper left')
        plt.grid()
        plt.show()
        
    def plot_train_vs_test(self) -> None:
        plt.title('Train data versus prediction')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        x = self.X.reshape(-1)
        y = self.Y.reshape(-1)
        if y is not None:
            plt.plot(x, y, '.', c='r', label='trn')
        x = self.x.reshape(-1)
        y = self.predict(self.x).reshape(-1)
        if y is not None:
            plt.plot(x, y, '-', c='b', label='tst')
        plt.legend()
        plt.grid()
        plt.show()

        plt.title('Target versus prediction')
        plt.xlabel('$Y$')
        plt.ylabel('$y$')
        x = self.Y
        y = self.predict(self.X)
        if y is not None:
            plt.plot(x.reshape(-1), y.reshape(-1), '.', label='trn')
        x = self.y
        y = self.predict(self.x)
        if y is not None:
            plt.plot(x.reshape(-1), y.reshape(-1), '-', label='tst')
        plt.legend()
        plt.grid()
        plt.show()


########################################################################

if __name__ == '__main__': 
    
    X = np.random.rand(100) * 4 * np.pi
    X = X.reshape(len(X), 1)
    Y = (np.sin(X) - 1) * 3 + 4.
#    Y = X**4 + X**3 - X

    
    dX = 0.5 * (X.max() - X.min())
    x = np.linspace(X.min() - dX, X.max() + dX, 101)
    x = x.reshape(len(x), 1)

    foo = NeuralK()
    foo(X=X, Y=Y, 
        activation=('tanh', 'elu',), # 'sigmoid', 'linear', ),
        goal=5e-4,
        epochs=1000,
        learning_rate=0.01,
        momentum=0.,
        patience=30,
        plot=1,
        trainer=('adam', ),  # 'rmsrprop', 'sgd',),
        trials=5,
        verbose=0,
      )
    y = foo(x=x, verbose=0, plot=1)

