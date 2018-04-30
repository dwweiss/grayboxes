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
      2018-04-30 DWW
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
    _hasNeurolab = False
    print("??? Import from 'neurolab' failed")


def proposeHiddenNeurons(X, Y, alpha=2, silent=False):
    """
    Proposes number of hidden neurons for given training data set

    Args:
        X, Y (2D array_like of float):
            training data

        alpha (float, optional):
            tuning parameter: alpha = 2..10 (2 supresses usually over-fitting)
    """
    alpha = 2
    xShape, yShape = np.atleast_2d(X).shape, np.atleast_2d(Y).shape

    assert xShape[0] > 2, str(xShape)
    assert xShape[0] == yShape[0], str(xShape) + ' ' + str(yShape)

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
    Trains and executes neural network

    Methods and attributes:
        - (error, trainer, epochs) = train(X, Y, **kwargs)
        - bool ready
        - string bestTrainer
        - y = predict(x, **kwargs)
        - plot()
        - y = __call__(X=None, Y=None, x=None, **kwargs)

    a) Wraps different neural network implementations from
        1) Neurolab: trains exclusively with backpropagation
        2) NeuralGenetic: trains exclusively with genetic algorithm

    b) Compares training algorithms and regularisation settings

    c) Presents graphically history of norms for each trial of training

    References:
        - Recommended training algorithms:
            'bfgs':  Broyden–Fletcher–Goldfarb–Shanno algorithm,
                     see: scipy.optimize.fmin_bfgs()
                     ref: wikipedia: Broyden-Fletcher-Goldfarb-Shanno_algorithm
            'rprop': resilient backpropagation (NO REGULARIZATION)
                     ref: wikipedia: Rprop
        - http://neupy.com/docs/tutorials.html#tutorials
    """

    def __init__(self, f=None):
        """
        Args:
            f (method or function, optional):
                theoretical model f(self, x) or f(x),
                default is None.
                if f is not None, genetic training or training with derivative
                dE/dy and dE/dw is employed
        """
        self._net = None         # network
        self._X = None           # input of training
        self._Y = None           # target
        self._x = None           # input of prediction
        self._y = None           # prediction y = f(x)
        self._norm_y = None      # data from normalization of target
        self._xKeys = None       # xKeys for import from data frame
        self._yKeys = None       # yKeys for import from data frame
        self._trainers = ''      # list of training algorithms
        self._finalErrors = []   # final errors of best trial for
        #                          each trainer in 'self._trainers'
        self._finalL2norms = []  # final L2-norm of best trial for each trainer
        self._bestEpochs = []    # epochs of best trial for each trainer
        self._ready = False      # flag indicating successful training
        self._best = None        # (error, trainer, epochs) of best train.trial

        if f is not None:
            firstArg = list(inspect.signature(f).parameters.keys())[0]
            if firstArg == 'self':
                f = f.__get__(self, self.__class__)
        self.f = f

        plt.rcParams.update({'font.size': 14})
        plt.rcParams['legend.fontsize'] = 14

    def __call__(self, X=None, Y=None, x=None, **kwargs):
        """
        - Trains neural network if both X is not None and Y is not None and
          sets self.ready to True if training is successful
        - Predicts y for input x if x is not None and self.ready is True

        Args:
            X, Y (2D array_like of float, optional):
                input and target for training

            x (1D or 2D array_like of float, optional):
                input for prediction

        Returns:
            (2D array of float):
                Prediction of neural network if x is not None
            or (3-tuple of float):
                (error, trainer, epochs) for best training trial if x is None
                    and both X and Y are not None
            or None:
                if X, Y and x are None

        Note:
            - The shape of X, Y and x is corrected to: (nPoint, nInp / nOut)
            - The references to X, Y, x and y are stored as self._X, self._Y,
              self._x, self._y and be accessed via the coresponding decorators
        """
        if X is not None and Y is not None:
            err = self.train(X, Y, **kwargs)
        else:
            err = None
        if x is not None:
            assert self.ready
            return self.predict(x=x)
        else:
            return err

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
            (3-tuple of float):
                (error, trainer, epochs) for best training trial
        """
        return self._best

    def importDataFrame(self, df, xKeys, yKeys):
        """
        Args:
            df (DataFrame):
                data  object

            xKeys (list of string):
                input  keys for data selection

            yKeys (list of string):
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

    def importArrays(self, X, Y, xKeys=None, yKeys=None):
        """
        Args:
            X (1D or 2D array_like of float):
                X will be converted to 2D-array
                (first index is data point index)

            Y (1D or 2D array_like of float):
                Y will be converted to 2D-array
                (first index is data point index)

            xKeys (1D array_like of string, optional):
                list of column keys for data selection
                use self._xKeys keys if xKeys is None,
                default: ['x0', 'x1', ... ]

            yKeys (1D array_like of string, optional):
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
        Args:
            X (1D or 2D array_like of float, optional):
                training input, X.shape: (nPoint, nInp)

            Y (1D or 2D array_like of float, optional):
                training target, Y.shape: (nPoint, nOut)

            kwargs (dict, optional):
                keywork arguments:

                epochs (int):
                    max number of iterations of single trial,
                    default is 1000

                errorf (function)
                    error function: { nl.error.MSE() | nl.error.SSE() },
                    default is MSE

                f (method or function, optional):
                    theoretical model y = f(x) for single data point

                goal (float):
                    limit of 'errorf' for stop of training (0 < goal < 1),
                    default is 1e-5

                hidden (int or array_like of int):
                    array of number of neurons in hidden layers,
                    default is max(1, round(nPoint / (alpha * (nInp + nOut))))

                outputf (function):
                    activation function of output layer,
                    default is TanSig()

                plot (int):
                    control of plots showing training progress,
                    default is 0 (no plot)

                regularization (float):
                    control of regularization (sum of all weights is added to
                    cost function of training, 0. <= regularization <= 1,
                    default is 0 (no effect of sum of all weights)

                show (int):
                    control of information about training, if show=0: no print,
                    default is epochs // 10

                silent (bool):
                    if True, no information is sent to console,
                    default is False

                smartTrials (bool):
                    if False, perform all trials even if goal has been reached,
                    default is True

                trainers (string or list of string):
                    if no list, space separated string is converted to list
                    if 'all' or None, all training algorithms will be applied,
                    default is 'bfgs'

                transf (function):
                    activation function of hidden layers,
                    default is TanSig()

                trials (int):
                    maximum number of training trials,
                    default is 3

        Returns:
            (3-tuple of float):
                (error, trainer, epochs) for best training trial

        Note:
            - Reference to training data is stored as self._X and self._Y
            - The best network has been assigned to 'self._net' before return
            - (error, trainer, epochs) for best training trial is stored as
              self._best
        """
        if X is not None and Y is not None:
            self.importArrays(X, Y)
        assert self._X is not None and self._Y is not None

        epochs         = kwargs.get('epochs',         1000)
        errorf         = kwargs.get('errorf',         nl.error.MSE())
        f              = kwargs.get('f',              None)
        goal           = kwargs.get('goal',           1e-5)
        hidden         = kwargs.get('hidden',         None)
        outputf        = kwargs.get('outputf',        nl.trans.TanSig())
        plot           = kwargs.get('plot',           1)
        regularization = kwargs.get('regularization', 1.0)
        show           = kwargs.get('show',           None)
        silent         = kwargs.get('silent',         False)
        smartTrials    = kwargs.get('smartTrials',    True)
        trainers       = kwargs.get('trainers',       'bfgs')
        transf         = kwargs.get('transf',         nl.trans.TanSig())
        trials         = kwargs.get('trials',         3)

        self._ready = False

        # if theoretical model 'f' is provided, alternative training is used
        if f is not None:
            trainers = [x for x in trainers if x in ('genetic', 'derivative')]
            if not trainers:
                trainers = 'genetic'
        else:
            if not trainers:
                trainers == 'all'
            if isinstance(trainers, str):
                if trainers == 'all':
                    trainers = 'cg gd gdx gdm gda rprop bfgs genetic'
                trainers = trainers.split()
            trainers = list(OrderedDict.fromkeys(trainers))  # redundancy
        self._trainers = trainers

        if errorf is None:
            errorf = nl.error.MSE()
        if show is None:
            show = epochs // 10
        if silent:
            plot = False

        if isinstance(hidden, (int, float)):
            hidden = list([int(hidden)])
        if not hidden or len(hidden) == 0:
            hidden = proposeHiddenNeurons(self._X, self._Y, alpha=2,
                                          silent=silent)
        if not isinstance(hidden, list):
            hidden = list(hidden)
        size = hidden.copy()
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

        assert all([x in trainfDict for x in self._trainers]), \
            str(self._trainers)
        if not silent:
            print('+++ trainers:', self._trainers)

        sequenceError = float('inf')
        bestTrainer = self._trainers[0]
        self._finalErrors = []
        self._finalL2norms = []
        self._bestEpochs = []
        for trainer in self._trainers:
            trainf = trainfDict[trainer]
            trainerErr = float('inf')
            trainerEpochs = None
            trainerL2norm = None

            net = nl.net.newff(nl.tool.minmax(self._X), size)
            net.transf = transf
            net.trainf = trainf
            if trainer in ('genetic', 'derivative'):
                # f ...
                net.errorf = errorf  # TODO mse with f ?
                net.outputf = outputf  # TODO f
            else:
                net.errorf = errorf
                net.outputf = outputf

            for jTrial in range(trials):
                net.init()
                if trainer == 'rprop':
                    e = net.train(self._X, self._Y, epochs=epochs, show=show,
                                  goal=goal)
                elif trainer.startswith(('gen', 'deriv')):
                    e = net.train(self._X, self._Y, f=self.f, epochs=epochs,
                                  show=show, goal=goal, rr=regularization)
                else:
                    e = net.train(self._X, self._Y, epochs=epochs, show=show,
                                  goal=goal, rr=regularization)
                trialErrors = e
                if sequenceError > trialErrors[-1]:
                    sequenceError = trialErrors[-1]
                    del self._net
                    self._net = net.copy()
                if (trainerErr < goal and trainerEpochs > len(trialErrors)) or\
                   (trainerErr >= goal and trainerErr > trialErrors[-1]):
                    trainerErr = trialErrors[-1]
                    trainerEpochs = len(trialErrors)
                    trainerL2norm = np.sqrt(np.mean(np.square(
                      self.predict(x=self._X) - self._norm_y.renorm(self._Y))))
                if plot:
                    plt.plot(range(len(trialErrors)), trialErrors,
                             label='trial: ' + str(jTrial))
                if smartTrials:
                    if trialErrors[-1] < goal:
                        break

            self._finalErrors.append(trainerErr)
            self._finalL2norms.append(trainerL2norm)
            self._bestEpochs.append(trainerEpochs)
            iBest = self._trainers.index(bestTrainer)
            if trainerErr < self._finalErrors[iBest]:
                bestTrainer = trainer

            if plot:
                plt.title("'" + trainer + "' mse:" +
                          str(round(trainerErr*1e3, 2)) + 'e-3 L2:' +
                          str(round(trainerL2norm, 3)) +
                          ' [' + str(trainerEpochs) + ']')
                plt.xlabel('epochs')
                plt.ylabel('error')
                plt.yscale('log', nonposy='clip')
                plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
                plt.grid()
                plt.show()
            if not silent:
                print('    ' + trainer + ':' + str(round(trainerErr, 5)) +
                      '[' + str(trainerEpochs) + '], ')
        if plot:
            self.plotTestWithTrainData()

        iBest = self._trainers.index(bestTrainer)
        if not silent:
            if len(self._trainers) > 1:
                print("    best trainer: '" + self._trainers[iBest] +
                      "' out of: [" + ' '.join(self._trainers) +
                      '], error:', round(self._finalErrors[iBest], 5))
                if len(self._finalErrors) > 1:
                    print("    (trainer:err): [", end='')
                    s = ''
                    for trainer, err in zip(self._trainers, self._finalErrors):
                        s += trainer + ':' + str(round(err, 5)) + ' '
                    print(s[:-2] + ']')

        self._ready = True

        self._best = self._finalErrors[iBest], self._trainers[iBest], \
            self._bestEpochs[iBest]
        return self._best

    def predict(self, x, **kwargs):
        if x is None:
            return None

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
        for trainer, error, epochs in zip(self._trainers, self._finalErrors,
                                          self._bestEpochs):
            y = self.predict(x=self._X)        # prediction
            Y = self._norm_y.renorm(self._Y)   # target

            title = 'Train (' + trainer + ') mse: ' + \
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
        ax.set_xticklabels(self._trainers)
        ax.set_yticks(np.add(y, y2))
        plt.title('Final training errors')
        plt.xlabel('trainer')
        plt.ylabel('error')
        plt.yscale('log', nonposy='clip')
        plt.grid()
        plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
        plt.show()


def trainGenetic(self, X, Y, **kwargs):
    epochs         = kwargs.get('epochs',   1000)
    errorf         = kwargs.get('errorf',   nl.error.MSE())
    f              = kwargs.get('f',        None)
    goal           = kwargs.get('goal',     1.0001e-5)
    hidden         = kwargs.get('hidden',   None)
    outputf        = kwargs.get('outputf',  None)
    regularization = kwargs.get('rr',       1.0)
    show           = kwargs.get('show',     None)
    transf         = kwargs.get('transf',   None)

    return [0]


# Examples ####################################################################

if __name__ == '__main__':
    ALL = 0

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
        best = net(X=X, Y=Y, hidden=[6], plot=1, epochs=500, goal=1e-4,
                   trials=5, trainers='rprop bfgs', regularization=0.0,
                   show=None)

        y = net(x=x)

        L2_norm = np.sqrt(np.mean(np.square(y - f(x))))
        plt.title('Test (' + best[1] + ') L2: ' +
                  str(round(L2_norm, 2)) + ' best: ' + str(round(best[0], 5)))
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

        y = Neural()(X=X, Y=Y, x=x, hidden=[6], plot=1, epochs=500, goal=1e-5,
                     trials=5, trainers='cg gdx rprop bfgs',
                     regularization=0.0, show=None)

        L2_norm = np.sqrt(np.mean(np.square(y - f(x))))
        plt.title('Test (' + best[1] + ') L2: ' +
                  str(round(L2_norm, 2)) + ' best: ' + str(round(best[0], 5)))
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
        best = net.train(goal=1e-6, hidden=[10, 3], plot=1, epochs=2000,
                         trainers='cg gdx rprop bfgs', trials=10,
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
        x = X

        net = Neural()
        y = net(X, Y, x, hidden=6, plot=1, epochs=1000, goal=1e-6,
                trainers='cg gdx rprop bfgs', trials=5)
        dy = np.subtract(y, Y)
        X, Y, x = net.X, net.Y, net.x
        if X.shape[1] == 2:
            plotWireframe(X[:, 0], X[:, 1], y[:, 0],  title='$y_{prd}$',
                          labels=['x', 'y', r'$Y_{targ}$'])
            plotWireframe(X[:, 0], X[:, 1], Y[:, 0],  title='$Y_{trg}$',
                          labels=['x', 'y', r'$Y_{targ}$'])
            plotWireframe(X[:, 0], X[:, 1], dy[:, 0], title=r'$\Delta y$',
                          labels=['x', 'y', r'$\Delta y$'])
            plotIsolines(X[:, 0], X[:, 1], y[:, 0],  title='$y_{prd}$')
            plotIsoMap(X[:,   0], X[:, 1], y[:, 0],  title='$y_{prd}$')
            plotIsoMap(X[:,   0], X[:, 1], Y[:, 0],  title='$Y_{trg}$')
            plotIsolines(X[:, 0], X[:, 1], Y[:, 0],  title='$Y_{trg}$')
            plotIsoMap(X[:,   0], X[:, 1], dy[:, 0], title=r'$\Delta y$')
            plotSurface(X[:,  0], X[:, 1], dy[:, 0], title=r'$\Delta y$')
            plotSurface(X[:,  0], X[:, 1], y[:, 0],  title='$y_{prd}$')

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
        y = Neural()(X=X, Y=Y, x=x, hidden=[8, 2], plot=1, epochs=2000,
                     goal=1e-5, trainers='rprop bfgs', trials=8)

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
