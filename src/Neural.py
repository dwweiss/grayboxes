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
      2018-02-08 DWW
"""

import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
try:
    import neurolab as nl
    _hasNeurolab = True
except ImportError:
    _hasNeurolab = False
    print("??? Import from 'neurolab' failed")
try:
    import genetic_algorithm as gea
    _hasGenetic = True
except ImportError:
    _hasGenetic = False
    print('??? Gea not imported')


class NeurGen(object):
    """
    Trains feed-forward network with genetic algorithm
    """

    def __init__(self, minmax, size):
        self.trainf = None
        self.errorf = None

    def init(self):
        pass

    def train(self, X, Y, **kwargs):
        """
        Trains feed-forward network with genetic algorithm

        Args:
            X (2D array_like of float):
                training input

            Y (2D array_like of float):
                training target

            kwargs (dict, optional):
                keywork arguments:

                epochs (int):
                    max number of iterations of single trial,
                    default is 1000

                errorf (function)
                    error function: { nl.error.MSE() | nl.error.SSE() },
                    default is MSE

                f (method):
                    method f(x) from Hybrid trough Empirical,
                    default is None (f(x) = x)

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
                    space separated string
                    if 'all' or None, all training algorithms will be applied,
                    default is 'bfgs'

                transf (function):
                    activation function of hidden layers,
                    default is TanSig()

                trials (int):
                    maximum number of training trials,
                    default is 3        

        Returns:
            errorHistory (array of float):
                error from 'errorf' for each epoch
        """

        # the only non-NeuroLab argument:
        f = kwargs.get('f', None)
        assert f is not None
        self.f = f.__get__(self, self.__class__)

        epochs         = kwargs.get('epochs',   1000)
        errorf         = kwargs.get('errorf',   nl.error.MSE())
        goal           = kwargs.get('goal',     1.0001e-5)
        hidden         = kwargs.get('hidden',   None)
        outputf        = kwargs.get('outputf',  None)
        regularization = kwargs.get('rr',       1.0)
        show           = kwargs.get('show',     None)
        transf         = kwargs.get('transf',   None)

        errorHistory = None
        return errorHistory

    def sim(self, x):
        y = None
        return y


class Neural(object):
    """
    a) Wraps different neural network implementations from
        1) Neurolab: trains exclusively with backpropagation
        2) gea: trains exclusively with genetic algorithm

    b) Compares different training algorithms and different regularisation
       settings

    c) Presents graphically history of norms for each trial

    References:
        - Recommended training algorithms:
            'bfgs':  Broyden–Fletcher–Goldfarb–Shanno algorithm,
                     see: scipy.optimize.fmin_bfgs()
                     ref: wikipedia: Broyden-Fletcher-Goldfarb-Shanno_algorithm
            'rprop': resilient backpropagation (NO REGULARIZATION)
                     ref: wikipedia: Rprop
        - http://neupy.com/docs/tutorials.html#tutorials

    Proposal for number of hidden neurons:
        nHidden = nPoint / ([2..10] * (nInp + nOut))
                             2 -> lowest risk of over-fitting

    Installation of neurolab:
        1) Fetch neurolab.0.3.5.tar.gz file  (or newer)
        2) Change to download directory
        3) python -m pip install .\neurolab.0.3.5.tar.gz
    """

    def __init__(self):
        self._X = None           # input
        self._Y = None           # target
        self._net = None         # network
        self._norm_y = None      # data from normalization of target
        self._xKeys = None       # xKeys for import from data frame
        self._yKeys = None       # yKeys for import from data frame
        self._trainers = ''      # list of training algorithms
        self._bestTrainer = ''   # best train algorithm
        self._finalErrors = []   # final errors of best trial for
        #                          each trainer in 'self._trainers'
        self._finalL2norms = []  # final L2-norm of best trial for each trainer
        self._bestEpochs = []    # epochs of best trial for each trainer
        plt.rcParams.update({'font.size': 14})
        plt.rcParams['legend.fontsize'] = 14

    @property
    def bestTrainer(self):
        return self._bestTrainer

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
        self._xKeys = list(xKeys)
        self._yKeys = list(yKeys)
        assert all(k in df for k in xKeys), 'unknown x-keys: ' + str(xKeys) + \
            'valid keys: ' + df.columns
        assert all(k in df for k in yKeys), 'unknown y-keys: ' + str(yKeys) + \
            ' , valid keys: ' + df.columns
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

            xKeys (1D array_like of string):
                list of column keys for data selection
                use self._xKeys keys if xKeys is None,
                default: ['x0', 'x1', ... ]

            yKeys (1D array_like of string):
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

    def train(self, **kwargs):
        """
        Args:
            kwargs (dict, optional):
                keywork arguments:

                epochs (int):
                    max number of iterations of single trial,
                    default is 1000

                errorf (function)
                    error function: { nl.error.MSE() | nl.error.SSE() },
                    default is MSE

                f (method):
                    method f(x) from Hybrid trough Empirical,
                    default is None (f(x) = x)

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
                    space separated string
                    if 'all' or None, all training algorithms will be applied,
                    default is 'bfgs'

                transf (function):
                    activation function of hidden layers,
                    default is TanSig()

                trials (int):
                    maximum number of training trials,
                    default is 3

        Returns:
            (error, trainer, epochs) for best training trial

        Note:
            The best network has been assigned to 'self._net' before return
        """
        assert not(self._X is None or self._Y is None), 'call import*() first'

        epochs         = kwargs.get('epochs',         1000)
        errorf         = kwargs.get('errorf',         nl.error.MSE())
        f              = kwargs.get('f',              None)
        goal           = kwargs.get('goal',           1e-5)
        hidden         = kwargs.get('hidden',         None)
        outputf        = kwargs.get('outputf',        None)
        plot           = kwargs.get('plot',           1)
        regularization = kwargs.get('regularization', 0.0)
        show           = kwargs.get('show',           None)
        silent         = kwargs.get('silent',         False)
        smartTrials    = kwargs.get('smartTrials',    True)
        trainers       = kwargs.get('trainers',       'bfgs')
        transf         = kwargs.get('transf',         None)
        trials         = kwargs.get('trials',         3)

        if not trainers:
            trainers == 'all'
        if isinstance(trainers, str):
            if trainers == 'all':
                trainers = 'cg gd gdx gdm gda rprop bfgs genetic'
            trainers = trainers.split()
        if f is not None:
            trainers = 'genetic'
        self._trainers = list(set(trainers))                # remove redundancy
        if not self._trainers:
            self._trainers = ['rprop bfgs']

        if errorf is None:
            errorf = nl.error.MSE()
        if show is None:
            show = epochs // 10
        if silent:
            plot = False

        if isinstance(hidden, (int, float)):
            hidden = list([int(hidden)])
        if hidden is None or len(hidden) == 0:
            alpha = 2                 # 2..10, 2 supresses usually over-fitting
            nPoint = self._X.shape[0]
            nInp, nOut = self._X.shape[1], self._Y.shape[1]
            nHidden = max(1, round(nPoint / (alpha * (nInp + nOut))))
            print("+++ auto def of 'nHidden': " + str(nHidden))
            hidden = [nHidden]
        if not isinstance(hidden, list):
            hidden = list(hidden)
        size = hidden.copy()
        size.append(self._Y.shape[1])
        assert size[-1] == self._Y.shape[1]

        trainfDict = {'genetic': None,
                      'bfgs':    nl.train.train_bfgs,
                      'cg':      nl.train.train_cg,
                      'gd':      nl.train.train_gd,
                      'gda':     nl.train.train_gda,
                      'gdm':     nl.train.train_gdm,
                      'gdx':     nl.train.train_gdx,
                      'rprop':   nl.train.train_rprop
                      }

        assert all([x in trainfDict for x in self._trainers])
        if not silent:
            print('+++ trainers:', self._trainers)

        sequenceError = float('inf')
        self._bestTrainer = self._trainers[0]
        self._finalErrors = []
        self._finalL2norms = []
        self._bestEpochs = []
        for trainer in self._trainers:
            trainf = trainfDict[trainer]
            trainerErr = float('inf')
            trainerEpochs = None
            trainerL2norm = None

            if trainer == 'genetic':
                net = NeurGen(nl.tool.minmax(self._X), size, f=f)
            else:
                net = nl.net.newff(nl.tool.minmax(self._X), size)
            net.trainf = trainf
            net.errorf = errorf

            for jTrial in range(trials):
                net.init()
                if trainer == 'rprop':
                    trialErrors = net.train(self._X, self._Y,
                                            epochs=epochs,
                                            show=show, goal=goal)
                else:
                    trialErrors = net.train(self._X, self._Y,
                                            epochs=epochs,
                                            show=show, goal=goal,
                                            rr=regularization)
                if sequenceError > trialErrors[-1]:
                    sequenceError = trialErrors[-1]
                    del self._net
                    self._net = net.copy()
                if (trainerErr < goal and trainerEpochs > len(trialErrors)) or\
                   (trainerErr >= goal and trainerErr > trialErrors[-1]):
                    trainerErr = trialErrors[-1]
                    trainerEpochs = len(trialErrors)
                    trainerL2norm = np.sqrt(np.mean(np.square(
                      self.__call__(self._X) - self._norm_y.renorm(self._Y))))
                if plot:
                    plt.plot(range(len(trialErrors)), trialErrors,
                             label='trial: ' + str(jTrial))
                if smartTrials:
                    if trialErrors[-1] < goal:
                        break

            self._finalErrors.append(trainerErr)
            self._finalL2norms.append(trainerL2norm)
            self._bestEpochs.append(trainerEpochs)
            iBest = self._trainers.index(self._bestTrainer)
            if trainerErr < self._finalErrors[iBest]:
                self._bestTrainer = trainer

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

        iBest = self._trainers.index(self._bestTrainer)
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

        return self._finalErrors[iBest], self._trainers[iBest], \
            self._bestEpochs[iBest]

    def __call__(self, x=None, **kwargs):
        return self.predict(x=x)

    def predict(self, **kwargs):
        x = kwargs.get('x', None)
        if x is None:
            x = self._X

        assert x is not None, 'x is None'
        assert self._net is not None, 'net is not trained'

        x = np.asfarray(x)
        if x.ndim == 1:
            x = x.reshape(x.size, 1)
        if x.shape[1] != self._net.ci:
            x = np.transpose(x)

        y = self._net.sim(x)

        return self._norm_y.renorm(y)

    def plotTestWithTrainData(self):
        for trainer, error, epochs in zip(self._trainers, self._finalErrors,
                                          self._bestEpochs):
            y = self.__call__(self._X)        # prediction
            Y = self._norm_y.renorm(self._Y)  # target

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


# Examples ####################################################################

if __name__ == '__main__':
    ALL = 1

    if 0 or ALL:
        s = 'Example 1'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        X = np.linspace(-1.75 * np.pi, 1.75 * np.pi, 50)
        Y = np.sin(X) * 10 + 0
        net = Neural()
        net.importArrays(X, Y)
        # trainers: 'cg gd gdx gdm gda rprop bfgs'

        net.train(hidden=[6], plot=1, epochs=500, goal=1e-5, trials=5,
                  trainers='cg gdx rprop bfgs', regularization=0.0, show=None)

        dx = 0.5 * (X.max() - X.min())
        x = np.linspace(X.min() - dx, X.max() + dx)
        y = net(x)
        L2_norm = np.sqrt(np.mean(np.square(y - Y)))
        plt.title('Test (' + net._bestTrainer + ') L2: ' +
                  str(round(L2_norm, 2)))
        plt.plot(x, y, '-', X, Y, '.')
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
        err = net.train(goal=1e-6, hidden=[10, 3], plot=1, epochs=2000,
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
        net = Neural()
        net.importArrays(X, Y)
        err = net.train(hidden=6, plot=1, epochs=1000, goal=1e-6,
                        trainers='cg gdx rprop bfgs', trials=5)
        y = net(X)[:, 0]
        X = np.asfarray(X)
        Y = np.asfarray(Y)[:, 0]
        dy = np.subtract(y, Y)
        if X.shape[1] == 2:
            plotWireframe(X[:, 0], X[:, 1], y,  title='$y_{prd}$',
                          labels=['x', 'y', r'$Y_{targ}$'])
            plotWireframe(X[:, 0], X[:, 1], Y,  title='$Y_{trg}$',
                          labels=['x', 'y', r'$Y_{targ}$'])
            plotWireframe(X[:, 0], X[:, 1], dy, title=r'$\Delta y$',
                          labels=['x', 'y', r'$\Delta y$'])
            plotIsolines(X[:, 0], X[:, 1], y,  title='$y_{prd}$')
            plotIsoMap(X[:,   0], X[:, 1], y,  title='$y_{prd}$')
            plotIsoMap(X[:,   0], X[:, 1], Y,  title='$Y_{trg}$')
            plotIsolines(X[:, 0], X[:, 1], Y,  title='$Y_{trg}$')
            plotIsoMap(X[:,   0], X[:, 1], dy, title=r'$\Delta y$')
            plotSurface(X[:,  0], X[:, 1], dy, title=r'$\Delta y$')
            plotSurface(X[:,  0], X[:, 1], y,  title='$y_{prd}$')

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

        error = net.train(X, YY, epochs=10000, show=100, goal=1e-6)
        yTrain = norm_y.renorm(net.sim(X))

        print(error[-1])
        plt.subplot(211)
        plt.plot(error)
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

    if 0 or ALL:
        s = 'Example 5'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        try:
            from plotArrays import plotSurface, plotIsolines, plotIsoMap
        except ImportError:
            print("??? import of 'plotArrays' failed")

        X = np.atleast_2d(np.linspace(-2 * np.pi, 2 * np.pi, 50)).T
        Y = np.sin(X) * 5

        net = Neural()
        net.importArrays(X, Y)
        # trainers: 'cg gd gdx gdm gda rprop bfgs'

        err = net.train(hidden=[8, 2], plot=1, epochs=2000, goal=1e-5,
                        trainers='rprop bfgs', trials=8)
        y = net(X)

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
