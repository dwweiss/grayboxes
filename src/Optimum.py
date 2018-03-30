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
      2018-02-06 DWW
"""

import numpy as np
from scipy.optimize import minimize, basinhopping
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for "projection='3d'"

from Base import Base
from Model import randInit
from Forward import Forward


class Optimum(Forward):
    """
    Examples:
        foo = Optimum()

        X = [[... ]]  input of training
        Y = [[... ]]  target of training
        xIni = [(x00, x01), ... ]
        bounds = [(x0min, x0max), (x1min, x1max), ... ]

        x, y = foo(X=X, Y=Y, x=xIni, f=f)         # train and optimize
        x, y = foo(x=xIni, bounds=bounds, f=f)    # optimize
        x, y = foo(ranges=[(0,1),(1,2)], rand=9)  # generate rand. x & optimize
        res = foo(XY=(X, Y, xKeys, yKeys))        # train only

    Note:
        Optimizer is for single target (Inverse: norm(y-Y), Optimum: one of y)
        Optimizer is only tested for 'Nelder-Mead'
        Penalty solution does not work yet
    """

    def __init__(self, identifier='Optimum', model=None, f=None, trainer=None):
        """
        Args:
            identifier (string, optional):
                class identifier

            model (method, optional):
                generic model

            f (method, optional):
                white box model f(x)

            trainer (method, optional):
                training method
        """
        super().__init__(identifier, model=model, f=f, trainer=trainer)

        self._bounds = None                                     # x-constraints
        self._x = None                       # 1D array of initial or optimal x
        self._y = None                        # 1D array of target or optimal y
        self._history = None       # history[iTrial][jStep] = (x, y, objective)
        self._trialHistory = None     # trialHistory[jStep] = (x, y, objective)

        # the three leading characters of optimizer types are significant
        self._validOptimizers = ['Nelder-Mead', 'Powell', 'CG', 'BFGS',
                                 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA',
                                 'SLSQP', 'dogleg', 'trust-ncg',
                                 'basinhopping']
        self._optimizer = self._validOptimizers[0]

    @property
    def bounds(self):
        """
        Gets x-bounds used in penalty() method

        Returns:
            (2D array of float):
                aray of (min, max) pairs of float:
                    [(x0_min, x0_max), ... ]
        """
        if self._bounds is None:
            return None
        else:
            return np.atleast_2d(self._bounds)

    @bounds.setter
    def bounds(self, value):
        """
        Sets x-bounds used in penalty() method

        Args:
            (2D array_like of float):
                array of (min, max) pairs of float:
                    [(x0_min, x0_max), ... ]
        """
        if value is None:
            self._bounds = None
        else:
            self._bounds = np.atleast_2d(value)

    @property
    def x(self):
        """
        1) Gets initial input before optimization
        2) Gets input at best optimum after optimization

        Returns:
            (1D array of float):
                input initial or input at optimum, index is parameter index
        """
        return self._x

    @x.setter
    def x(self, value):
        """
        1) Sets initial input before optimization
        2) Sets input at best optimum after optimization

        Args:
            value (float or 1D array_like of float):
                input initial or input at optimum, index is parameter index
        """
        if value is None:
            self._x = None
        else:
            self._x = np.atleast_2d(value)

    @property
    def y(self):
        """
        1) Gets target before optimization, only if inverse problem
        2) Gets output at best optimum after optimization

        Returns:
            (1D array of float):
                target or output at optimum, index is parameter index
        """
        return self._y

    @y.setter
    def y(self, value):
        """
        1) Sets target before optimization, only if inverse problem
        2) Sets output at best optimum after optimization

        Args:
            value (1D array_like of float):
                target or output at optimum, index is parameter index
        """
        if value is None:
            self._y = None
        else:
            self._y = np.atleast_1d(value)

    @property
    def optimizer(self):
        """
        Returns:
            (string):
                kind of optimizer
        """
        return str(self._optimizer)

    @optimizer.setter
    def optimizer(self, value):
        """
        Sets optimizer, corrects too short strings and assigns default if value
        is unknown

        Args:
            value (string):
                kind of optimizer
        """
        if value is None:
            value = self._validOptimizers[0]
        value = str(value)
        if len(value) == 1:
            self.write("??? optimizer too short (min. 3 char): '", value,
                       "', continue with: '", self._validOptimizers[0], "'")
            self._optimizer = self._validOptimizers[0]

        valids = [x.lower().startswith(value[:3].lower())
                  for x in self._validOptimizers]
        if not any(valids):
            self.write("??? Unknown optimizer type: '", value,
                       "', continue with: '", self._validOptimizers[0], "'")
            self._optimizer = self._validOptimizers[0]
        else:
            self._optimizer = self._validOptimizers[valids.index(True)]

    def penalty(self, x):
        """
        Penalty to be added to objective function if x is out of self.bounds

        Args:
            x (1D array_like of float):
                input array of single data points

        Returns:
            (float):
                penalty to objective function
        """
        assert self.bounds is None, 'penalty() is not tested'

        penaltyValue = 1e10
        if self.bounds is not None:
            for limit, _x in zip(self.bounds, x):
                if limit[0] is not None and limit[0] > _x:
                    print('+++ penalty limit[0]:', limit[0], ' > x:', _x,
                          'penalty:', penaltyValue)
                    return penaltyValue
                if limit[1] is not None and _x > limit[1]:
                    print('+++ penalty limit[1]:', limit[1], ' < x:', _x,
                          'penalty:', penaltyValue)
                    return penaltyValue
        return 0.0

    def objective(self, x, **kwargs):
        """
        Objective function for optimization

        Args:
            x (2D or 1D array_like of float):
                input array of single or multiple data points

            kwargs (dict, optional):
                keyword arguments

        Returns:
            (float):
                optimization criterion to be minimized
        Note:
            If maximum is wanted, use minus sign in objective: max(x) = -min(x)
        """
        y = np.atleast_2d(self.model.predict(x, **self.kwargsDel(kwargs, 'x')))
        out = y[0]
        opt = out[0]                        # first row, first column of output

        self._trialHistory.append([x, out, opt])

        return opt + self.penalty(x)

    def task(self, **kwargs):
        """
        Args:
            kwargs (dict, optional):
                keyword arguments:

                bounds (2D array_like of float):
                    array of min/max pairs for optimization constraints etc

                optimizer (string):
                    type of optimizer

                y (1D array_like of float):
                    target y of inverse probl.(only if self is of type Inverse)

        Returns:
            self.x, self.y (two 1D array_like of float):
                model input at optimum and corresponding model output
        Note:
            requires initial point(s) self.x for getting number of x-pars
            requires defined self.y as target if inverse problem solution
        """
        Base.task(self, **kwargs)

        bounds = kwargs.get('bounds', None)
        if bounds is not None:
            self.bounds = bounds

        optimizer = kwargs.get('optimizer', None)
        if optimizer is not None:
            self.optimizer = optimizer

        # sets target for Inverse
        if type(self).__name__ in ('Inverse'):
            y = kwargs.get('y', None)
            self.y = np.atleast_1d(y) if y is not None else self.y

        assert self.model is not None

        xIni = self.x.copy()
        trials = xIni.shape[0]
        self._history = []
        for iTrial in range(trials):
            self._trialHistory = []
            x0 = xIni[iTrial]

            if self.optimizer.startswith('bas'):
                res = basinhopping(func=self.objective, x0=x0,
                                   niter=100, T=1.0, stepsize=0.5,
                                   minimizer_kwargs=None, take_step=None,
                                   accept_test=None, callback=None,
                                   interval=50, disp=False, niter_success=None)
                x = np.atleast_1d(res.x)
                y = np.atleast_1d(res.fun)
                success = True
            else:
                res = minimize(fun=self.objective, x0=x0,
                               method=self.optimizer,)
                x = np.atleast_1d(res.x)
                kw = self.kwargsDel(kwargs, 'x')
                y = np.asfarray(self.model.predict(x=res.x, **kw)[0])
                success = res.success

            self._history.append(self._trialHistory)

        if not success:
            self.write('+++ error message: ', res.message)
            x = [None] * x.size
            y = [None] * y.size

        nTrial = len(self._history[0])
        if nTrial > 1:
            self.write('+++ Optima of all trials:')
            for iTrial, history in enumerate(self._history):
                self.write('    [' + str(iTrial) + '] x: ', history[-1][0],
                           '\n        y: ', history[-1][1],
                           '\n        objective: ', history[-1][2])

            # self._history[iTrial][iLast=-1][jObj=2] -> list of best obj.
            finalObjectives = [hist[-1][2] for hist in self._history]
            if self.__class__.__name__ == 'Inverse':
                absFinalObj = np.abs(finalObjectives)
                iTrialBest = finalObjectives.index(min(absFinalObj))
            else:
                iTrialBest = finalObjectives.index(min(finalObjectives))

        else:
            iTrialBest = 0

        # y: self._history[iTrialBest][iLast=-1][jY=1]
        historyBest = self._history[iTrialBest]
        finalBest = historyBest[-1]
        self.x, self.y = finalBest[0], finalBest[1]
        objectiveBest = finalBest[2]
        self.write('+++ Best trial:\n    [' + str(iTrialBest) + '] x: ',
                   self.x, '\n        y: ', self.y, '\n        objective: ',
                   objectiveBest)
        return self.x, self.y

    def plot(self):
        self.plotHistory()
        self.plotObjective()
        self.plotTrajectory()

    def plotHistory(self):
        for iTrial, trialHist in enumerate(self._history):
            print('+++ Plot[iTrial: ' + str(iTrial) + ']',
                  trialHist[-1][1], ' = f(' + str(trialHist[-1][0]) + ')')
            assert len(trialHist[0]) > 1

            # self._history[iTrial][ [x0..nInp-1], [y0..nOut-1], obj ]
            nInp, nOut = len(trialHist[0][0]), len(trialHist[0][1])
            x_seq = []
            for jInp in range(nInp):
                x_seq.append([P[0][jInp] for P in trialHist])
            y_seq = []
            for jOut in range(nOut):
                x_seq.append([P[1][jOut] for P in trialHist])
            obj = [P[2] for P in trialHist]

            for jInp in range(nInp):
                plt.title('objective (x'+str(jInp)+'), trial: '+str(iTrial))

                plt.plot(x_seq[jInp], obj, label='obj(x'+str(jInp)+')')
                plt.scatter([x_seq[jInp][0]], [obj[0]],
                            color='r', marker='o', label='start')
                plt.scatter([x_seq[jInp][-1]], [obj[-1]],
                            color='g', marker='s', label='stop')
                plt.xlabel('x' + str(jInp))
                plt.ylabel('obj')
                plt.grid()
                plt.legend(bbox_to_anchor=(1.1, 1.025), loc='upper left')
                plt.show()

            for jOut, _y in enumerate(y_seq):
                yLab = 'y' + str(jOut)

                plt.title(yLab + '(x0..x' + str(nInp-1) + ')')
                for jInp in range(nInp):

                    plt.plot(x_seq[jInp], _y,
                             label=yLab+'(x'+str(jInp)+')')
                    plt.scatter([x_seq[jInp][0]], [_y[0]], color='r',
                                marker='o', label='' if jInp else 'start')
                    plt.scatter([x_seq[jInp][-1]], [_y[-1]], color='b',
                                marker='s', label='' if jInp else 'stop')
                plt.xlabel('x0..x' + str(nInp-1))
                plt.ylabel(yLab)
                plt.grid()
                plt.legend(bbox_to_anchor=(1.1, 1.025),
                           loc='upper left')
                plt.show()

                plt.title(yLab + '(time), trial:' + str(iTrial))

                plt.plot(range(len(_y)), _y, '-', label=yLab+'(time)')
                plt.scatter([0], [_y[0]],
                            color='r', marker='o', label='start')
                plt.scatter([len(_y)-1], [_y[-1]],
                            color='b', marker='s', label='stop')
                plt.xlabel('time')
                plt.ylabel(yLab)
                # plt.yscale('log', nonposy='clip')
                plt.grid()
                plt.legend(bbox_to_anchor=(1.1, 1.025),
                           loc='upper left')
                plt.show()

                if nInp > 1:
                    plt.title('trajectory(x), trial:' + str(iTrial))

                    plt.plot(x_seq[0], x_seq[1], '-', label='trajctory')
                    plt.xlabel('x0')
                    plt.ylabel('x1')
                    plt.grid()
                    plt.legend(bbox_to_anchor=(1.1, 1.025),
                               loc='upper left')
                    plt.show()

    def plotObjective(self):
        nInp = len(self._history[0][0][0])
        nTrial = len(self._history)

        plt.title('Objective (x0..x' + str(nInp-1)+', trial[0..' +
                  str(nTrial-1) + '])')
        for iTrial, trialHist in enumerate(self._history):
            obj = [P[2] for P in trialHist]
            for jInp in range(nInp):
                x = [P[0][jInp] for P in trialHist]

                plt.plot(obj, x, label='x'+str(jInp)+'['+str(iTrial)+']')
                plt.scatter([obj[0]], [x[0]], color='r', marker='o',
                            label='' if jInp or iTrial else 'start')
                plt.scatter([obj[-1]], [x[-1]], color='b', marker='x',
                            label='' if jInp or iTrial else 'stop')
        plt.ylabel('x0..x' + str(nInp-1))
        plt.xlabel('objective')
        plt.grid()
        plt.legend(bbox_to_anchor=(1.1, 1.025), loc='upper left')
        plt.show()

    def plotTrajectory(self):
        nInp = len(self._history[0][0][0])
        nOut = len(self._history[0][0][1])

        if nInp >= 2:
            print('+++ Trajectory of objective versus (x0, x1), all trials')
            mpl.rcParams['legend.fontsize'] = 10
            fig = plt.figure(figsize=(10, 8))
            ax = fig.gca(projection='3d')
            ax.set_xlabel('x0')
            ax.set_ylabel('x1')
            ax.set_zlabel('obj')
            for iTrial, trialHist in enumerate(self._history):
                x_seq = []
                for jInp in range(nInp):
                    x_seq.append([P[0][jInp] for P in trialHist])
                for jOut in range(nOut):
                    x_seq.append([P[1][jOut] for P in trialHist])
                obj = [P[2] for P in trialHist]
                x0, x1 = x_seq[0], x_seq[1]

                ax.plot(x0, x1, obj, label='trial '+str(iTrial))
                ax.scatter([x0[0]], [x1[0]], [obj[0]], color='r', marker='o',
                           label='' if iTrial else 'start')
                ax.scatter([x0[-1]], [x1[-1]], [obj[-1]], color='b',
                           marker='x', label='' if iTrial else 'stop')
            ax.legend()
            plt.show()

        if nInp >= 3:
            print('+++ Trajectories of x[2..nInp] vs. (x0, x1), all trials')
            for jInpZ in range(2, nInp):
                mpl.rcParams['legend.fontsize'] = 10
                fig = plt.figure(figsize=(10, 8))
                ax = fig.gca(projection='3d')
                ax.set_xlabel('x0')
                ax.set_ylabel('x1')
                ax.set_zlabel('x'+str(jInpZ))
                for iTrial, history in enumerate(self._history):
                    x_seq = []
                    for jInp in range(nInp):
                        x_seq.append([P[0][jInp] for P in history])
                    x0, x1, xZ = x_seq[0], x_seq[1], x_seq[jInpZ]

                    ax.plot(x0, x1, xZ, label='trial '+str(iTrial))
                    ax.scatter([x0[0]], [x1[0]], [xZ[0]],
                               color='g', marker='o',
                               label='' if iTrial else 'start')
                    ax.scatter([x0[-1]], [x1[-1]], [xZ[-1]],
                               color='r', marker='x',
                               label='' if iTrial else 'stop')
                ax.legend()
                plt.show()


# Examples ####################################################################

if __name__ == '__main__':
    ALL = 1

    # user defined method of theoretical models
    def f(self, x, c0=1, c1=1, c2=1, c3=1, c4=1, c5=1, c6=1, c7=1):
        return np.sin(c0 * x[0]) + c1 * (x[1] - 1)**2 + c2

    if 0 or ALL:
        s = 'Test minimizer from Minpack'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        def f1(x):
            y = np.sin(x[0]) - (x[1] - 1)**2
            return y
        res = minimize(fun=f1, x0=(4, 2), method='nelder-mead',)
        print('res.x:', res.x)

    if 0 or ALL:
        s = 'Optimum, assigns series of initial x'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        foo = Optimum(f=f)
        xIni = randInit(n=10, ranges=((-5, 5), (-7, 7)))
        x, y = foo(x=xIni, optimizer='nelder-mead')
        foo.plot()
        print('x:', x, 'y:', y, '\nfoo.x:', foo.x, 'foo.y:', foo.y)

    if 1 or ALL:
        s = 'Optimum, generates series of initial x from ranges'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        foo = Optimum(f=f)
        x, y = foo(rand=10, ranges=((-5, 5), (-7, 7)))
        foo.plot()
        print('x:', x, 'y:', y, '\nfoo.x:', foo.x, 'foo.y:', foo.y)
