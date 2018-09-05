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
      2018-09-04 DWW

  Acknowledgement:
      Modestga is a contribution by Krzyzstof Arendt, SDU, Denmark
"""

import sys
import numpy as np
import scipy.optimize
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # needed for "projection='3d'"
from typing import Any, List, Optional, Sequence, Tuple

from grayboxes.base import Base
from grayboxes.boxmodel import BoxModel
from grayboxes.forward import Forward
try:
    import modestga as mg
except ImportError:
    print('??? Package modestga not imported')


class Minimum(Forward):
    """
    Minimizes objective()

    Examples:
        op = Minimum(f)

        X = [[... ]]  input of training
        Y = [[... ]]  target of training
        xIni = [(x00, x01), ... ]
        bounds = [(x0min, x0max), (x1min, x1max), ... ]

        x, y = op(X=X, Y=Y, x=xIni)          # train lightgray box and optimize
        x, y = op(x=xIni, bounds=bounds)                             # optimize
        x, y = op(x=rand(9, [0, 1], [1, 2]))   # generate random x and optimize
        norm = op(XY=(X, Y, xKeys, yKeys))                         # train only

    Notes:
        - Limited to single target (Inverse: norm(y - Y), Optimum: one of y)
             
        - At the end of every evolution of objective(), the single 
          point input array x, the single point output array y, and 
          the scalar result of the objective function is appended to 
          self._trialHistory
             
        - The list of single point input, output and objective 
          function result is stored self._history             

        - Parent class Forward has no self._x or self._y attribute
    """

    def __init__(self, model: BoxModel, identifier: str='Minimum') -> None:
        """
        Args:
            model:
                Box type model object

            identifier:
                Unique object identifier
        """
        super().__init__(identifier=identifier, model=model)

        self._bounds: Sequence[Tuple[float, float]] = None                              
                                                         # x-constraints
        self._x: np.ndarray = None    # 1D array of initial or optimal x
        self._y: np.ndarray = None     # 1D array of target or optimal y
        
        SINGLE_EVALUATION = Tuple[np.ndarray, np.ndarray, float]
                                                     # [x, y, objective]
        self._trialHistory: List[SINGLE_EVALUATION] = []  
                            # _trialHistory[jEvaluation]=(x,y,objective)
        self._history: List[List[SINGLE_EVALUATION]] = []  
                       # _history[iTrial][jEvaluation] = (x,y,objective)

        # three leading chars are significant, case-insensitive
        self._validMethods = ['Nelder-Mead',
                              'Powell',
                              'CG',
                              'BFGS',
                              # 'Newton-CG',         # requires Jacobian
                              'L-BFGS-B',
                              'TNC',
                              # 'COBYLA',# failed in grayBoxes test case
                              'SLSQP',
                              # 'dogleg',            # requires Jacobian
                              # 'trust-ncg',         # requires Jacobian
                              'basinhopping',           # GLOBAL optimum
                              'differential_evolution',    # GLOBAL opt.
                              ]
        if 'modestga' in sys.modules:
            self._validMethods += ['ga']

        self._method = self._validMethods[0]

    @property
    def x(self) -> np.ndarray:
        """
        1) Gets initial input before optimization
        2) Gets input at best optimum after optimization

        Returns:
            (2D array of float):
                input initial or input at optimum, 
                index is parameter index
        """
        return self._x

    @x.setter
    def x(self, value: Optional[np.ndarray]) -> None:
        """
        Sets x-array

        Args:
            value (1D or 2D array of float):
                input initial or input at optimum, 
                index is parameter index
        """
        if value is None:
            self._x = None
        else:
            self._x = np.atleast_2d(value)

    @property
    def y(self) -> np.ndarray:
        """
        1) Gets target to class Inverse before optimization
        2) Gets best output at optimum after optimization

        Returns:
            (1D array of float):
                target or output at optimum, index is parameter index
        """
        return self._y

    @y.setter
    def y(self, value: Optional[np.ndarray]) -> None:
        """
        Sets y-array

        Args:
            value (1D array of float):
                target or output at optimum, index is parameter index
        """
        if value is None:
            self._y = None
        else:
            self._y = np.atleast_1d(value)

    @property
    def method(self) -> str:
        """
        Returns:
            Optimization method
        """
        return str(self._method)

    @method.setter
    def method(self, value: Optional[str]) -> None:
        """
        Sets optimization method(s), corrects strings which are too short 
        and assigns default if value is unknown method

        Args:
            value:
                optimization method
        """
        if value is None:
            value = self._validMethods[0]
        value = str(value)
        if len(value) == 1:
            self.write("??? optimiz. method too short (min 2 char): '" + 
                       str(value) + "', continue with: '" + 
                       self._validMethods[0] + "'")
            self._method = self._validMethods[0]

        valids = [x.lower().startswith(value[:3].lower())
                  for x in self._validMethods]
        if not any(valids):
            self.write("??? Unknown optimization method: '" + str(value) +
                       "', continue with: '" + str(self._validMethods[0]) +
                       "', validMethods: '" + str(self._validMethods) + "'")
            self._method = self._validMethods[0]
        else:
            self._method = self._validMethods[valids.index(True)]

    def objective(self, x: np.ndarray, **kwargs: Any) -> float:
        """
        Objective function to be minimized

        Args:
            x (2D or 1D array of float):
                input of multiple or single data points,
                shape: (nPoint, nInp) or (nInp,)

        Kwargs:
            Keyword arguments to be passed to model.predict()

        Returns:
            Result of objective function (least sqaures sum)

        Note:
            If maximum is wanted, use minus sign in objective: 
                max(x) = -min(x)
        """
        assert np.atleast_2d(x).shape[0] == 1, str(np.atleast_2d(x).shape[0])

        y = self.model.predict(x, **self.kwargsDel(kwargs, 'x'))
        out = y[0]                                    # first data point
        obj = out[0]                                  # first output

        self._trialHistory.append((x, out, obj))

        return obj

    def task(self, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kwargs:
            bounds (Sequence[Tuple[float, float]]):
                array of min/max pairs for optimization constraints etc

            method (str):
                optimization method

            y (1D array_like of float):
                target of inverse problem (only if 'self' is of type
                Inverse), shape: (nOut)

        Returns:
            (2-tuple of 1D array of float):
                model input at optimum and corresponding model output,
                shape: (nInp,) and shape: (nOut,)
        Note:
            - requires initial point(s) self.x for getting input number: nInp
            - if inverse problem solution then self.y is required as target
        """
        Base.task(self, **kwargs)

        bounds = kwargs.get('bounds', None)

        method = self.kwargsGet(kwargs, ('method', 'methods'), None)
        if method is not None:
            self.method = method
        self.write('+++ Method: ' + str(self.method))

        # sets target for Inverse
        if type(self).__name__ in ('Inverse'):
            y = kwargs.get('y', None)
            self.y = np.atleast_1d(y) if y is not None else self.y

        assert self.model is not None

        xIni = self.x.copy()
        self._history = []
        success = False
        for x0 in xIni:              # xIni.shape[0] is number of trials
            #
            # Note: self._trialHistory list is populated in objective()
            #
            self._trialHistory = []

            if self.method.startswith('bas'):
                res = scipy.optimize.basinhopping(
                    func=self.objective, x0=x0, niter=100, T=1.0, stepsize=0.5,
                    minimizer_kwargs=None, take_step=None, accept_test=None,
                    callback=None, interval=50, disp=False, niter_success=None)
                # x, y = np.atleast_1d(res.x), np.atleast_1d(res.fun)
                success = 'success' in res.message[0]

            elif self.method.startswith('dif'):
                if bounds is None:
                    bounds = [(-10, 10)] * len(x0)
                res = scipy.optimize.differential_evolution(
                    func=self.objective, bounds=bounds, strategy='best1bin',
                    maxiter=None, popsize=15, tol=0.01, mutation=(0.5, 1),
                    recombination=0.7, seed=None, disp=False, polish=True,
                    init='latinhypercube')
                # x, y = res.x, res.fun
                success = res.success

            elif self.method == 'ga':
                validKeys = ['tol', 'options', 'bounds']
                kw = {k: kwargs[k] for k in validKeys if k in kwargs}
                res = mg.minimize(fun=self.objective, x0=x0,
                                  # method=self.method, # TODO .
                                  **kw)
                # x, y = np.atleast_1d(res.x), np.atleast_1d(res.fx)
                success = True  # TODO .
            else:
                res = scipy.optimize.minimize(fun=self.objective, x0=x0,
                                              method=self.method,)
                # x = np.atleast_1d(res.x)
                # kw = self.kwargsDel(kwargs, 'x')
                # y = self.model.predict(x=res.x, **kw)[0]
                success = res.success

            # Note: length of self._history equals number of trials
            self._history.append(self._trialHistory)
        else:
            success, res = None, None

        if not success:
            message = res.message if res is not None else '---'
            self.write("+++ error message: '" + message + "'")
            if type(self).__name__ in ('Inverse'):
                self.x = [None] * self.x.size
                self.y = [None] * self.y.size

        nTrial = len(self._history[0])
        if nTrial > 1:
            self.write('+++ Optima of all trials:')
            for iTrial, history in enumerate(self._history):
                s = ' ' if nTrial < 10 else ''
                self.write('    ['+s+str(iTrial)+'] x: '+str(history[-1][0]))
                self.write('         y: ' + str(history[-1][1]))
                self.write('         objective: ' + str(history[-1][2]))

            # self._history[iTrial][iLast=-1][jObj=2] -> list of best obj.
            finalObjectives = [hist[-1][2] for hist in self._history]
            if self.__class__.__name__ == 'Inverse':
                absFinalObj = np.abs(finalObjectives)
                iBestTrial = finalObjectives.index(min(absFinalObj))
            else:
                iBestTrial = finalObjectives.index(min(finalObjectives))

        else:
            iBestTrial = 0

        # y: self._history[iBestTrial][iLastEvaluation=-1][jY=1]
        historyBest = self._history[iBestTrial]
        finalBest = historyBest[-1]
        self.x, self.y = finalBest[0], finalBest[1]
        objectiveBest = finalBest[2]
        self.write('+++ Best trial:')
        s = ' ' if iBestTrial < 10 else ''
        self.write('    [' + s + str(iBestTrial) + '] x: ' + str(self.x))
        self.write('         y: ' + str(self.y)) 
        self.write('         objective: ' + str(objectiveBest))

        return self.x, self.y

    def plot(self, select: Optional[str]=None) -> None:
        if select is None or not isinstance(select, str):
            select = 'all'
        select = select.lower()
        if select.startswith(('his', 'all')):
            self.plotHistory()
        if select.startswith(('obj', 'all')):
            self.plotObjective()
        if select.startswith(('tra', 'trj', 'all')):
            self.plotTrajectory()

    def plotHistory(self) -> None:
        
        # type(trial) = List[Tuple[x: np.ndarray, y: np.ndarray, obj: float]]
        for iTrial, trial in enumerate(self._history):
            self.write('    Plot[iTrial: ' + str(iTrial) + '] ' + 
                       str(trial[-1][1]) + ' = f(' + 
                       str(trial[-1][0]) + ')')
            assert len(trial[0]) > 1

            nInp, nOut = len(trial[0][0]), len(trial[0][1])
            x_seq = []
            for jInp in range(nInp):
                x_seq.append([P[0][jInp] for P in trial])
            y_seq = []
            for jOut in range(nOut):
                x_seq.append([P[1][jOut] for P in trial])
            obj = [P[2] for P in trial]

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

    def plotObjective(self) -> None:

        # type(self._history) = List[List[
        #              Tuple[x: np.ndarray, y: np.ndarray, obj: float]]]
        nInp = len(self._history[0][0][0])  # trial:0, evaluation:0, x:0
        nTrial = len(self._history)

        plt.title('Objective (x0..x' + str(nInp-1)+', trial[0..' +
                  str(nTrial-1) + '])')

        # type(trial) = List[Tuple[x: np.ndarray, y: np.ndarray, 
        #                          obj: float]]
        for iTrial, trial in enumerate(self._history):
            obj = [P[2] for P in trial]
            for jInp in range(nInp):
                x = [P[0][jInp] for P in trial]

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

    def plotTrajectory(self) -> None:
        nInp = len(self._history[0][0][0])  # trial:0, evaluation:0, x-index=0
        nOut = len(self._history[0][0][1])  # trial:0, evaluation:0, y-index=0

        if nInp >= 2:
            self.write('    Trajectory of objective vs (x0, x1), all trials')
            mpl.rcParams['legend.fontsize'] = 10
            fig = plt.figure(figsize=(10, 8))
            ax = fig.gca(projection='3d')
            ax.set_xlabel('x0')
            ax.set_ylabel('x1')
            ax.set_zlabel('obj')

            # type(trial) = List[Tuple[x: np.ndarray, y: np.ndarray, 
            #                          obj: float]]
            for iTrial, trial in enumerate(self._history):
                x_seq = []
                for jInp in range(nInp):
                    x_seq.append([P[0][jInp] for P in trial])
                for jOut in range(nOut):
                    x_seq.append([P[1][jOut] for P in trial])
                obj = [P[2] for P in trial]
                x0, x1 = x_seq[0], x_seq[1]

                ax.plot(x0, x1, obj, label='trial '+str(iTrial))
                ax.scatter([x0[0]], [x1[0]], [obj[0]], color='r', marker='o',
                           label='' if iTrial else 'start')
                ax.scatter([x0[-1]], [x1[-1]], [obj[-1]], color='b',
                           marker='x', label='' if iTrial else 'stop')
            ax.legend()
            plt.show()

        if nInp >= 3:
            self.write('+++ Trajectory of x[2..nInp] vs. (x0, x1), all trials')
            for jInpZ in range(2, nInp):
                mpl.rcParams['legend.fontsize'] = 10
                fig = plt.figure(figsize=(10, 8))
                ax = fig.gca(projection='3d')
                ax.set_xlabel('x0')
                ax.set_ylabel('x1')
                ax.set_zlabel('x'+str(jInpZ))

                # type(trial) = List[Tuple[x: np.ndarray, y: np.ndarray, 
                #                          obj: float]]
                for iTrial, trial in enumerate(self._history):
                    x_seq = []
                    for jInp in range(nInp):
                        x_seq.append([P[0][jInp] for P in trial])
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
