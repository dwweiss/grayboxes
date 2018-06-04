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
      2018-06-03 DWW

  Acknowledgements:
      Neurolab is a contribution by E. Zuev (pypi.python.org/pypi/neurolab)
"""

from collections import OrderedDict
import inspect
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
try:
    import neurolab as nl
    _hasNeurolab = True
except ImportError:
    print('??? Module neurolab not imported')
    _hasNeurolab = False


def proposeHiddenNeurons(X, Y, alpha=2, silent=False):
    """
    Proposes number of hidden neurons for given training data set

    Args:
        X (2D array_like of float):
            training input, shape: (nPoint, nInp)

        Y (2D array_like of float):
            training target, shape: (nPoint, nOut)

        alpha (float, optional):
            tuning parameter, alpha = 2..10 (2 reduces over-fitting)
            default: 2

        silent (bool, optional):
            if True then suppress printed messages
            deafult: False

    Returns:
        (list of int):
            Estimate for optimal number of neurons of a single hidden layer
    """
    alpha = 2
    xShape, yShape = np.atleast_2d(X).shape, np.atleast_2d(Y).shape

    assert xShape[0] > xShape[1], str(xShape)
    assert xShape[0] > 2, str(xShape)
    assert xShape[0] == yShape[0], str(xShape) + str(yShape)

    nPoint, nInp, nOut = xShape[0], xShape[1], yShape[1]
    try:
        nHidden = max(1, round(nPoint / (alpha * (nInp + nOut))))
    except ZeroDivisionError:
        nHidden = max(nInp, nOut) + 2
    if not silent:
        print("+++ auto definition of 'nHidden': " + str(nHidden))
    hidden = [nHidden]

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
        - y = Neural()(X=None, Y=None, x=None, **kwargs) #y.shape:(nPoint,nOut)
        - best = self.train(X, Y,**kwargs)                      # see self.best
        - y = self.predict(x, **kwargs)               # y.shape: (nPoint, nOut)
        - self.ready                                                     # bool
        - self.best                            # dict{str: float or str or int}
        - self.plot()

    References:
        - Recommended training algorithms:
              'rprop': resilient backpropagation (NO REGULARIZATION)
                       wikipedia: 'Rprop'
              'bfgs':  Broyden–Fletcher–Goldfarb–Shanno algorithm,
                       see: scipy.optimize.fmin_bfgs()
                       wikipedia: 'Broyden-Fletcher-Goldfarb-Shanno_algorithm'
        - http://neupy.com/docs/tutorials.html#tutorials
    """

    def __init__(self, f=None):
        """
        Args:
            f (method or function, optional):
                theoretical submodel as method f(self, x) or function f(x)
                default: None

                if f is not None, genetic training or training with derivative
                dE/dy and dE/dw is employed
        """
        self.f = f               # theoretical submodel for single data point

        self._net = None         # network
        self._X = None           # input of training
        self._Y = None           # target
        self._x = None           # input of prediction
        self._y = None           # prediction y = f(x)
        self._norm_y = None      # data from normalization of target
        self._xKeys = None       # xKeys for import from data frame
        self._yKeys = None       # yKeys for import from data frame
        self._methods = ''       # list of training algorithms
        self._finalErrors = []   # error(SSE,MSE) of best trial of each method
        self._finalL2norms = []  # L2-norm of best trial of each method
        self._bestEpochs = []    # epochs of best trial of each method
        self._ready = False      # flag indicating successful training

        self._silent = False
        plt.rcParams.update({'font.size': 14})
        plt.rcParams['legend.fontsize'] = 14                   # fonts in plots

        self._best = {'method': None, 'L2': np.inf, 'abs': np.inf,
                      'iAbs': -1, 'epochs': -1}   # results of best train trial

    def __call__(self, X=None, Y=None, x=None, **kwargs):
        """
        - Trains neural network if X is not None and Y is not None
        - Sets self.ready to True if training is successful
        - Predicts y for input x if x is not None and self.ready is True

        Args:
            X (2D or 1D array_like of float, optional):
                training input, shape: (nPoint, nInp) or shape: (nPoint)
                default: self.X

            Y (2D or 1D array_like of float, optional):
                training target, shape: (nPoint, nOut) or shape: (nPoint)
                default: self.Y

            x (2D or 1D array_like of float, optional):
                prediction input, shape: (nPoint, nInp) or shape: (nInp)
                default: self.x

            kwargs (dict, optional):
                keyword arguments, see: train() and predict()

        Returns:
            (2D array of float):
                prediction of network net(x) if x is not None and self.ready
            or
            (dict {str: float or str or int}):
                result of best training trial if X and Y are not None
                    'method'  (str): best method
                    'L2'    (float): sqrt{sum{(net(x)-Y)^2}/N} of best training
                    'abs'   (float): max{|net(x) - Y|} of best training
                    'iAbs'    (int): index of Y where absolute error is maximum
                    'epochs'  (int): number of epochs of best training
            or
            (None):
                if (X, Y and x are None) or not self.ready

        Note:
            - The shape of X, Y and x is corrected to: (nPoint, nInp/nOut)
            - The references to X, Y, x and y are stored as self.X, self.Y,
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
    def f(self):
        return self._f

    @f.setter
    def f(self, value):
        if value is not None:
            firstArg = list(inspect.signature(value).parameters.keys())[0]
            if firstArg == 'self':
                value = value.__get__(self, self.__class__)
        self._f = value

    @property
    def silent(self):
        return self._silent

    @silent.setter
    def silent(self, value):
        self._silent = value

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def ready(self):
        return self._ready

    @property
    def best(self):
        """
        Returns:
            (dict {str : float or str or int}):
                results for best training trial
                [see self.train()]
        """
        return self._best

    def importDataFrame(self, df, xKeys, yKeys):
        """
        Imports training input X and training target Y
        self.Y is the normalized target after import, but 'df' stays unchanged

        Args:
            df (pandas.DataFrame):
                data object

            xKeys (list of str):
                input  keys for data selection

            yKeys (list of str):
                output keys for data selection
        """
        self._xKeys = np.atleast_1d(xKeys)
        self._yKeys = np.atleast_1d(yKeys)
        assert all(k in df for k in xKeys), "unknown x-keys: '" + str(xKeys) +\
            "', valid keys: '" + df.columns + "'"
        assert all(k in df for k in yKeys), "unknown y-keys: '" + str(yKeys) +\
            "', valid keys: '" + df.columns + "'"
        self._X = np.asfarray(df.loc[:, xKeys])
        self._Y = np.asfarray(df.loc[:, yKeys])

        self._norm_y = nl.tool.Norm(self._Y)
        self._Y = self._norm_y(self._Y)

    def setArrays(self, X, Y, xKeys=None, yKeys=None):
        """
        - Imports training input X and training target Y
        - converts X and Y to 2D arrays
        - normalizes training target (self.Y is then the normalized target,
          but argument 'Y' stays unchanged)

        Args:
           X (2D or 1D array_like of float, optional):
                training input, shape: (nPoint, nInp) or shape: (nPoint)

            Y (2D or 1D array_like of float, optional):
                training target, shape: (nPoint, nOut) or shape: (nPoint)

            xKeys (1D array_like of str, optional):
                list of column keys for data selection
                use self._xKeys keys if xKeys is None,
                default: ['x0', 'x1', ... ]

            yKeys (1D array_like of str, optional):
                list of column keys for data selection
                use self._yKeys keys if yKeys is None,
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

        if xKeys is None:
            self._xKeys = ['x' + str(i) for i in range(self._X.shape[1])]
        else:
            self._xKeys = xKeys
        if yKeys is None:
            self._yKeys = ['y' + str(i) for i in range(self._Y.shape[1])]
        else:
            self._yKeys = yKeys

        self._norm_y = nl.tool.Norm(self._Y)
        self._Y = self._norm_y(self._Y)

    def train(self, X=None, Y=None, **kwargs):
        """
        Trains model, stores X and Y as self.X and self.Y, and stores result of
        best training trial as self.best

        Args:
            X (2D or 1D array_like of float, optional):
                training input, shape: (nPoint, nInp) or shape: (nPoint)
                default: self.X

            Y (2D or 1D array_like of float, optional):
                training target, shape: (nPoint, nOut) or shape: (nPoint)
                default: self.Y

            kwargs (dict, optional):
                keyword arguments:

                alpha (float):
                    factor for autodefinition of number of hidden neurons,
                    see: proposeHiddenNeurons()
                    default: 2.0

                epochs (int):
                    max number of iterations of single trial
                    default: 1000

                errorf (function)
                    error function: (nl.error.MSE() or nl.error.SSE())
                    default: MSE

                goal (float):
                    limit of 'errorf' for stop of training (0. < goal < 1.)
                    default: 1e-5
                    [note: L2-norm of 1e-6 corresponds to MSE of 1e-3]

                methods (str or list of str):
                    if no list then space separated string is converted to list
                    if 'all' or None, then all training methods are assigned
                    default: 'bfgs' if self.f is None else 'genetic'

                method (str or list of str):
                    [same as 'methods']

                neurons (int or array_like of int):
                    array of number of neurons in hidden layers
                    default: [] ==> use estimate of proposeHiddenNeurons()

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

                smartTrials (bool):
                    if False, perform all trials even if goal has been reached
                    default: True

                transf (function):
                    activation function of hidden layers
                    default: TanSig()

                trials (int):
                    maximum number of training trials
                    default: 3

        Returns:
            result of best training trial:
                'method'  (str): best method
                'L2'    (float): sqrt{sum{(net(x)-Y)^2}/N} of best training
                'abs'   (float): max{|net(x) - Y|} of best training
                'iAbs'    (int): index of Y where absolute error is maximum
                'epochs'  (int): number of epochs of best training

        Note:
            - If training fails then self.best['method']=None
            - Reference to optional theoretical submodel is stored as self.f
            - Reference to training data is stored as self.X and self.Y
            - The best network is assigned to 'self._net'
        """
        if X is not None and Y is not None:
            self.setArrays(X, Y)
        assert self._X is not None and self._Y is not None

        alpha          = kwargs.get('alpha',          2.0)
        epochs         = kwargs.get('epochs',         1000)
        errorf         = kwargs.get('errorf',         nl.error.MSE())
        goal           = kwargs.get('goal',           1e-5)
        methods        = kwargs.get('methods', None)
        if methods is None:
            methods    = kwargs.get('method', 'bfgs rprop')
        neurons        = kwargs.get('neurons',        None)
        outputf        = kwargs.get('outputf',        nl.trans.TanSig())
        plot           = kwargs.get('plot',           1)
        rr             = kwargs.get('regularization', None)
        if rr is None:
            rr         = kwargs.get('rr',             None)
        if rr is None:
            rr = 1.0
        show           = kwargs.get('show',           0)
        self.silent    = kwargs.get('silent',         self.silent)
        if show is not None and show > 0:
            self.silent = False
        smartTrials    = kwargs.get('smartTrials',    True)
        transf         = kwargs.get('transf',         nl.trans.TanSig())
        trials         = kwargs.get('trials',         3)

        if self.silent:
            show = 0
            plot = 0

        self._ready = False

        # if theoretical submodel 'f' is provided, alternative training is used
        if self.f is not None:
            methods = [x for x in methods if x in ('genetic', 'derivative')]
            if not methods:
                methods = 'genetic'
        else:
            if not methods:
                methods == 'all'
            if isinstance(methods, str):
                if methods == 'all':
                    methods = 'cg gd gdx gdm gda rprop bfgs genetic'
                methods = methods.split()
            methods = list(OrderedDict.fromkeys(methods))        # redundancy
        self._methods = methods

        if errorf is None:
            errorf = nl.error.MSE()
        if show is None:
            show = epochs // 10
        if self.silent:
            plot = False

        if isinstance(neurons, (int, float)):
            neurons = list([int(neurons)])
        if not neurons or len(neurons) == 0 or not all(neurons):
            neurons = proposeHiddenNeurons(X=self._X, Y=self._Y, alpha=alpha,
                                           silent=self.silent)
        if not isinstance(neurons, list):
            neurons = list(neurons)
        assert all(x > 0 for x in neurons), str(neurons)

        size = neurons.copy()
        size.append(self._Y.shape[1])
        assert size[-1] == self._Y.shape[1]

        trainfDict = {'genetic':    trainGenetic,
                      'derivative': nl.train.train_bfgs,
                      'bfgs':       nl.train.train_bfgs,
                      'cg':         nl.train.train_cg,
                      'gd':         nl.train.train_gd,
                      'gda':        nl.train.train_gda,
                      'gdm':        nl.train.train_gdm,
                      'gdx':        nl.train.train_gdx,
                      'rprop':      nl.train.train_rprop
                      }

        assert all([x in trainfDict for x in self._methods]), \
            str(self._methods)
        if not self.silent:
            print('+++ methods:', self._methods)

        sequenceError = np.inf
        bestmethod = self._methods[0]
        self._finalErrors, self._finalL2norms, self._bestEpochs = [], [], []

        for method in self._methods:
            trainf = trainfDict[method]
            methodErr = np.inf
            methodEpochs = None
            methodL2norm = None

            if method in ('genetic', 'derivative'):
                assert 0
                # f ...
                net = nl.net.newff(nl.tool.minmax(self._X), size)
                net.transf, net.outputf = transf, outputf
                net.errorf = errorf    # TODO MSE with f ?
                net.outputf = outputf  # TODO f
            else:
                net = nl.net.newff(nl.tool.minmax(self._X), size)
                net.transf, net.outputf = transf, outputf
                net.trainf, net.errorf = trainf, errorf

            for jTrial in range(trials):
                if method.startswith(('gen', 'deriv')):
                    net.init()
                    trialErrors = net.train(self._X, self._Y, f=self.f,
                                            epochs=epochs, goal=goal, rr=rr,
                                            show=show, )
                elif method == 'rprop':
                    net.init()
                    trialErrors = net.train(self._X, self._Y, epochs=epochs,
                                            show=show, goal=goal)
                else:
                    net.init()
                    trialErrors = net.train(self._X, self._Y, epochs=epochs,
                                            show=show, goal=goal, rr=rr)
                assert len(trialErrors) >= 1, str(trialErrors)
                if sequenceError > trialErrors[-1]:
                    sequenceError = trialErrors[-1]
                    del self._net
                    self._net = net.copy()
                if (methodErr < goal and methodEpochs > len(trialErrors)) or\
                   (methodErr >= goal and methodErr > trialErrors[-1]):
                    methodErr = trialErrors[-1]
                    methodEpochs = len(trialErrors)
                    methodL2norm = np.sqrt(np.mean(np.square(
                      self.predict(x=self._X) - self._norm_y.renorm(self._Y))))
                if plot:
                    plt.plot(range(len(trialErrors)), trialErrors,
                             label='trial: ' + str(jTrial))
                if smartTrials:
                    if trialErrors[-1] < goal:
                        break

            self._finalErrors.append(methodErr)
            self._finalL2norms.append(methodL2norm)
            self._bestEpochs.append(methodEpochs)
            iBest = self._methods.index(bestmethod)
            if methodErr < self._finalErrors[iBest]:
                bestmethod = method

            if plot:
                plt.title("'" + method + "' mse:" +
                          str(round(methodErr*1e3, 2)) + 'e-3 L2:' +
                          str(round(methodL2norm, 3)) +
                          ' [' + str(methodEpochs) + ']')
                plt.xlabel('epochs')
                plt.ylabel('error')
                plt.yscale('log', nonposy='clip')
                plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
                plt.grid()
                plt.show()
            if not self.silent:
                print('    ' + method + ':' + str(round(methodErr, 5)) +
                      '[' + str(methodEpochs) + '], ')
        if plot:
            self.plotTestWithTrainData()

        iBest = self._methods.index(bestmethod)
        if not self.silent:
            if len(self._methods) > 1:
                print("    best method: '" + self._methods[iBest] +
                      "' out of: [" + ' '.join(self._methods) +
                      '], error:', round(self._finalErrors[iBest], 5))
                if len(self._finalErrors) > 1:
                    print("    (method:err): [", end='')
                    s = ''
                    for method, err in zip(self._methods, self._finalErrors):
                        s += method + ':' + str(round(err, 5)) + ' '
                    print(s[:-2] + ']')

        self._ready = True

        # assign results of best trial to return value
        Y = self._norm_y.renorm(self._Y)
        dy = self.predict(self._X) - Y
        iAbsMax = np.abs(dy).argmax()
        self._best = {'method': self._methods[iBest],
                      'L2': np.sqrt(np.mean(np.square(dy))),
                      'abs': Y.ravel()[iAbsMax],
                      'iAbs': iAbsMax,
                      'epochs': self._bestEpochs[iBest]}
        return self.best

    def predict(self, x, **kwargs):
        """
        Executes network, stores x as self.x

        Args:
            x (2D or 1D array_like of float):
                prediction input, shape: (nPoint, nInp) or shape: (nInp)

            kwargs (dict, optional):
                keyword arguments

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
            - Shape of x is corrected to: (nPoint, nInp)
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

    def plot(self):
        self.plotTestWithTrainData()

    def plotTestWithTrainData(self):
        for method, error, epochs in zip(self._methods, self._finalErrors,
                                          self._bestEpochs):
            y = self.predict(x=self._X)        # prediction
            Y = self._norm_y.renorm(self._Y)   # target

            title = 'Train (' + method + ') mse: ' + \
                str(round(error * 1e3, 2)) + 'e-3 [' + str(epochs) + ']'

            plt.title(title)
            for j, yTrainSub in enumerate(Y.T):
                dy = np.subtract(y.T[j], yTrainSub)
                for i, xTrainSub in enumerate(self._X.T):
                    label = self._xKeys[i] + ' & ' + self._yKeys[j]
                    plt.plot(xTrainSub, dy, label=label)
            plt.xlabel('$x$')
            plt.ylabel('$y_{pred} - y_{train}$')
            plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
            plt.show()

            plt.title(title)
            for j, yTrainSub in enumerate(Y.T):
                for i, xTrainSub in enumerate(self._X.T):
                    label = self._xKeys[i] + ' & ' + self._yKeys[j]
                    plt.plot(xTrainSub, y.T[j], label=label)
                    plt.plot(xTrainSub, yTrainSub, label=label +
                             ' (target)', linestyle='', marker='*')
            plt.xlabel('$x$')
            plt.ylabel('$y$')
            plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
            plt.show()

        x = range(len(self._finalErrors))
        y = self._finalErrors
        y2 = np.asfarray(self._bestEpochs) * 1e-5
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


def trainGenetic(net, X, Y, **kwargs):
    """
    Trains model, stores X and Y as self.X and self.Y, and stores result of
    best training trial as self.best

    Args:
        net (network object):
            neural network object

        X (2D or 1D array_like of float):
            training input, shape: (nPoint, nInp) or shape: (nPoint)

        Y (2D or 1D array_like of float):
            training target, shape: (nPoint, nOut) or shape: (nPoint)

        kwargs (dict, optional):
            keyword arguments:

            f (function, optional):
                theoretical submodel
                default: None

            ... additional keyword arguments as in self.train()

    Returns:
        see self.train()
    """
    f = kwargs('f', None)

    best = {'method': None, 'L2': np.inf}

    return best


# Examples ####################################################################

if __name__ == '__main__':
    ALL = 1

    if 0 or ALL:
        s = 'Example 1 __call__()'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        def f(x):
            return np.sin(x) * 10 + 0

        X = np.linspace(-1.75 * np.pi, 1.75 * np.pi, 50)
        dx = 0.25 * (X.max() - X.min())
        x = np.linspace(X.min() - dx, X.max() + dx)
        X[0] = x[0]
        X[-1] = x[-1]
        Y = f(X)
        net = Neural()
        net(X=X, Y=Y, neurons=[6], epochs=500, goal=1e-4, show=None,
            trials=5, methods='rprop bfgs', regularization=0.0, plot=1)
        y = net(x=x)

        L2_norm = np.sqrt(np.mean(np.square(y - f(x))))
        plt.title('Test (' + net.best['method'] + ') L2: ' +
                  str(round(net.best['L2'], 2)))
        plt.plot(x, y, '-')
        plt.plot(X, Y, '.')
        plt.plot(x, f(x), ':')
        plt.legend(['pred', 'targ', 'true'])
        plt.xlabel('x')
        plt.ylabel('y(x)')
        plt.show()

    if 0 or ALL:
        s = 'Example 1 compact form'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        def f(x):
            return np.sin(x) * 10 + 0

        X = np.linspace(-1.75 * np.pi, 1.75 * np.pi, 50)
        Y = f(X)
        dx = 0.5 * (X.max() - X.min())
        x = np.linspace(X.min() - dx, X.max() + dx)

        net = Neural()
        y = net(X=X, Y=Y, x=x, neurons=[6], plot=1, epochs=500, goal=1e-5,
                trials=5, methods='cg gdx rprop bfgs',
                regularization=0.0, show=None)

        plt.title('Test, L2:' + str(round(net.best['L2'], 5)))
        plt.plot(x, y, '-')
        plt.plot(X, Y, '.')
        plt.legend(['pred', 'targ', ])
        plt.xlabel('x')
        plt.ylabel('y(x)')
        plt.show()

    if 0 or ALL:
        s = 'Example 2'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        df = DataFrame({'p0': [10, 20, 30, 40], 'p1': [11, 21, 31, 41],
                        'p2': [12, 22, 32, 42], 'r0': [31, 41, 51, 52],
                        'r1': [32, 42, 52, 55]})
        xKeys = ['p0', 'p2']
        yKeys = ['r0', 'r1']
        net = Neural()
        net.importDataFrame(df, xKeys, yKeys)
        best = net.train(goal=1e-6, neurons=[10, 3], plot=1, epochs=2000,
                         methods='cg gdx rprop bfgs', trials=10,
                         regularization=0.01, smartTrials=False)

    if 0 or ALL:
        s = 'Example 3'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        try:
            from plotArrays import plotSurface, plotIsolines, plotIsoMap, \
                 plotWireframe
        except ImportError:
            print("??? import from 'plotArrays' failed")

        X = [[10, 11], [11, 33], [33, 14], [37, 39], [20, 20]]
        Y = [[10, 11], [12, 13], [35, 40], [58, 68], [22, 28]]
        x = X.copy()

        net = Neural()
        y = net(X, Y, x, neurons=6, plot=1, epochs=1000, goal=1e-6,
                methods='cg gdx rprop bfgs', trials=5)
        dy = y - Y
        X, Y, x = net.X, net.Y, net.x
        if X.shape[1] == 2:
            plotWireframe(X[:, 0], X[:, 1],  y[:, 0],  title='$y_{prd}$',
                          labels=['x', 'y', r'$Y_{trg}$'])
            plotWireframe(X[:, 0], X[:, 1],  Y[:, 0],  title='$Y_{trg}$',
                          labels=['x', 'y', r'$Y_{trg}$'])
            plotWireframe(X[:, 0], X[:, 1], dy[:, 0], title=r'$\Delta y$',
                          labels=['x', 'y', r'$\Delta y$'])
            plotIsolines(X[:, 0], X[:, 1],  y[:, 0], title='$y_{prd}$')
            plotIsoMap(X[:,   0], X[:, 1],  y[:, 0], title='$y_{prd}$')
            plotIsoMap(X[:,   0], X[:, 1],  Y[:, 0], title='$Y_{trg}$')
            plotIsolines(X[:, 0], X[:, 1],  Y[:, 0], title='$Y_{trg}$')
            plotIsoMap(X[:,   0], X[:, 1], dy[:, 0], title=r'$\Delta y$')
            plotSurface(X[:,  0], X[:, 1], dy[:, 0], title=r'$\Delta y$')
            plotSurface(X[:,  0], X[:, 1],  y[:, 0], title='$y_{prd}$')

    if 0 or ALL:
        s = 'Example 4: newff and train without class Neural'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        X = np.atleast_2d(np.linspace(-7, 7, 20)).T
        Y = np.sin(X) * 10

        norm_y = nl.tool.Norm(Y)
        YY = norm_y(Y)
        net = nl.net.newff(nl.tool.minmax(X), [5, YY.shape[1]])
        # net.trainf = nl.train.train_rprop  # or:
        net.trainf = nl.train.train_bfgs

        err = net.train(X, YY, epochs=10000, show=100, goal=1e-6)
        yTrain = norm_y.renorm(net.sim(X))

        print(err[-1])
        plt.subplot(211)
        plt.plot(err)
        plt.legend(['L2 error'])
        plt.xlabel('Epoch number')
        plt.ylabel('error (default SSE)')

        xTest = np.atleast_2d(np.linspace(-5, 8, 150)).T
        yTest = norm_y.renorm(net.sim(xTest)).ravel()

        plt.subplot(212)
        plt.plot(xTest, yTest, '-', X, Y, '.')
        plt.legend(['pred', 'targ'])
        plt.xlabel('x')
        plt.ylabel('y(x)')
        plt.show()

    if 1 or ALL:
        s = 'Example 5'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        try:
            from plotArrays import plotSurface, plotIsolines, plotIsoMap
        except ImportError:
            print("??? import of 'plotArrays' failed")

        X = np.atleast_2d(np.linspace(-2 * np.pi, 2 * np.pi, 50)).T
        Y = np.sin(X) * 5
        x = X
        y = Neural()(X=X, Y=Y, x=x, neurons=[8, 2], plot=1, epochs=2000,
                     goal=1e-5, methods='rprop bfgs', trials=8)

        if X.shape[1] == 1:
            plt.plot(X, y, label='pred')
            plt.plot(X, Y, label='targ')
            plt.legend()
            plt.show()
        elif X.shape[1] == 2:
            plotSurface(X[:,  0], X[:, 1], y[:, 0], title='$y_{prd}$')
            plotIsolines(X[:, 0], X[:, 1], y[:, 0], title='$y_{prd}$')
            plotIsolines(X[:, 0], X[:, 1], Y[:, 0], title='$y_{trg}$')
        dy = y - Y
        if X.shape[1] == 2:
            plotIsoMap(X[:, 0], X[:, 1], dy[:, 0],
                       title='$y_{prd} - y_{trg}$')
