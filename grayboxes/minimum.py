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
      2019-11-22 DWW

  Acknowledgement:
      Modestga is a contribution by Krzyzstof Arendt, SDU, Denmark
"""

import sys
import numpy as np
import scipy.optimize
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # needed for "projection='3d'"
from typing import Any, List, Optional, Sequence, Tuple, Union

from grayboxes.base import Base
from grayboxes.base import Float1D, Float2D
from grayboxes.boxmodel import BoxModel
from grayboxes.forward import Forward
try:
    import modestga as mg
except ImportError:
    print('??? Package modestga not imported')


class Minimum(Forward):
    """
    Minimizes objective()
    
    Wrappes the theoretical submodel f(x,*args,**kwargs) -> List[float] 
    so that scipy.optimize.minimum() can be employed

    Examples:
        op = Minimum(f)

        X = [[... ]]  input of training
        Y = [[... ]]  target of training
        x_ini = [(x00, x01), ... ]
        bounds = [(x0min, x0max), (x1min, x1max), ... ]

        x, y = op(X=X, Y=Y, x=x_ini)  # train lightgray box and optimize
        x, y = op(x=x_ini, bounds=bounds)                     # optimize
        x, y = op(x=rand(9, [0, 1], [1, 2]))     # random x and optimize
        norm = op(XY=(X, Y, x_keys, y_keys))                # train only

    Notes:
        - Limited to single target (Inverse: norm(y - Y), 
                                    Optimum: one element of of y array)
        - At the end of every evolution of objective(), the single
          point input array x, the single point output array y, and
          the scalar result of the objective function is appended to
          self._single_hist

        - The list of single point input, output and objective
          function result is stored self._multiple_hist

        - Parent class Forward has no self._x or self._y attribute
    """

    def __init__(self, model: BoxModel, identifier: str = 'Minimum') -> None:
        """
        Args:
            model:
                Box type model object

            identifier:
                Unique object identifier
        """
        super().__init__(identifier=identifier, model=model)

        self._bounds: Optional[Sequence[Tuple[float, float]]] = None
                                                         # x-constraints
        self._x_opt: Float1D = None   # 1D array of initial or optimal x
        self._y_opt: Float1D = None    # 1D array of target or optimal y

        SINGLE_EVALUATION = Tuple[Float1D, Float1D, float]
                                                     # [x, y, objective]
        self._single_hist: List[SINGLE_EVALUATION] = []
                            # _trialHistory[jEvaluation]=(x,y,objective)
        self._multiple_hist: List[List[SINGLE_EVALUATION]] = []
                       # _multiple_hist[iTrial][jEvaluation] = (x,y,objective)

        # three leading chars are significant, case-insensitive
        self._valid_optimizers = ['Nelder-Mead',
                                  'Powell',
                                  'CG',
                                  'BFGS',
                                   # 'Newton-CG',    # requires Jacobian
                                  'L-BFGS-B',
                                  'TNC',
                                   # 'COBYLA',# failed in test_lightgray
                                  'SLSQP',
                                   # 'dogleg',       # requires Jacobian
                                   # 'trust-ncg',    # requires Jacobian
                                  'basinhopping',       # GLOBAL optimum
                                  'differential_evolution', # GLOBAL opt
                                   ]
        if 'modestga' in sys.modules:
            self._valid_optimizers += ['ga']

    @property
    def x(self) -> Float2D:
        """
        1) Gets initial input before optimization
        2) Gets input at best optimum after optimization

        Returns:
            initial input or input at optimum,
            index is parameter index
        Note:
            Minimum.x is different from Base.x
        """
        return self._x_opt

    @x.setter
    def x(self, value: Union[Float2D, Float1D]) -> None:
        """
        Sets x-array

        Args:
            value:
                initial input (1D or 2D) or input at optimum (1D),
                
                index is parameter index
        Note:
            Minimum.x is different from Base.x
        """
        if value is None:
            self._x_opt = None
        else:
            self._x_opt = np.atleast_2d(value)

    @property
    def y(self) -> Float1D:
        """
        1) Gets target to class Inverse before optimization
        2) Gets best output at optimum after optimization

        Returns:
            target or output at optimum, index is parameter index
            
        Note:
            Minimum.y is different from Base.y
        """
        return self._y_opt

    @y.setter
    def y(self, value: Float1D) -> None:
        """
        Sets y-array

        Args:
            value:
                target or output at optimum, index is parameter index
        Note:
            Minimum.y is different from Base.y
        """
        if value is None:
            self._y_opt = None
        else:
            self._y_opt = np.atleast_1d(value)

    def objective(self, x: Union[Float2D, Float1D], **kwargs: Any) -> float:
        """
        Objective function to be minimized

        Args:
            x:
                input of single data point,
                  shape: (n_point=1, n_inp) or (n_inp,)
                If 2D array, shape[0] must be 1 

        Kwargs:
            Keyword arguments to be passed to model.predict()

        Returns:
            Result of objective function (least sqaures sum)

        Note:
            If maximum is wanted, use minus sign in objective:
                max(x) = -min(x)
        """
        assert np.atleast_2d(x).shape[0] == 1, str(np.atleast_2d(x).shape[0])

        y = self.model.predict(x, **self.kwargs_del(kwargs, 'x'))
        out = y[0]                                    # first data point
        obj = out[0]                                      # first output

        self._single_hist.append((x, out, obj))

        return obj

    def task(self, **kwargs: Any) -> Union[float, 
                                           Tuple[Float1D, Float1D]]:
        """
        Kwargs:
            bounds (Sequence[Tuple[float, float]]):
                array of min/max pairs for optimization constraints etc

            optimizer (str):
                optimization method

            x (2D or 1D array of float):
                initial values as multiple or single data points,
                shape: (n_point, n_inp) or (n_inp, )
                
            y (1D array of float):
                target of inverse problem (only if 'self' is of type
                Inverse), shape: (n_out,)

        Returns:
            model input at optimum shape: (n_inp,) and corresponding 
                model output: shape: (n_out,)
            or 
            0.0 if this instance is an instance of Inverse
                
        Note:
            - requires initial point(s) self.x for getting number of 
              input: n_inp
            - if inverse problem solution, then self.y is required as 
              target
        """
        Base.task(self, **kwargs)

        bounds = kwargs.get('bounds', None)

        # sets initial values or returns 0.0 (if x is None,model train only)
        self.x = kwargs.get('x', None)
        if self.x is None:
            return 0.

        # sets target for Inverse (if y is None, model training only)
        if type(self).__name__ in ('Inverse'):
            self.y = kwargs.get('y', None)
            if self.y is None:
                return 0.

        optimizer = self.kwargs_get(kwargs, 'optimizer', None)
        if optimizer not in self._valid_optimizers:
            optimizer = self._valid_optimizers[0]
        self.write('+++ optimizer: ' + str(optimizer))

        self._multiple_hist = []
        
        success: Optional[bool] = None
        res: Optional[Any] = None
        x_ini = self.x.copy()
        for x0 in x_ini:            # x_ini.shape[0] is number of trials

            #
            # Note: self._trialHistory list is populated in objective()
            #
            self._single_hist = []

            if optimizer.startswith('bas'):
                res = scipy.optimize.basinhopping(
                    func=self.objective, x0=x0, niter=100, T=1.0, stepsize=0.5,
                    minimizer_kwargs=None, take_step=None, accept_test=None,
                    callback=None, interval=50, disp=False, niter_success=None)
                # x, y = np.atleast_1d(res.x), np.atleast_1d(res.fun)
                success = 'success' in res.message[0]

            elif optimizer.startswith('dif'):
                if bounds is None:
                    bounds = [(-10, 10)] * len(x0)
                res = scipy.optimize.differential_evolution(
                    func=self.objective, bounds=bounds, strategy='best1bin',
                    maxiter=None, popsize=15, tol=0.01, mutation=(0.5, 1),
                    recombination=0.7, seed=None, disp=False, polish=True,
                    init='latinhypercube')
                # x, y = res.x, res.fun
                success = res.success

            elif optimizer == 'ga':
                valid_keys = ['tol', 'options', 'bounds']
                kw = {k: kwargs[k] for k in valid_keys if k in kwargs}
                res = mg.minimize(fun=self.objective, x0=x0,
                                  # method=optimizer, # TODO .
                                  **kw)
                # x, y = np.atleast_1d(res.x), np.atleast_1d(res.fx)
                success = True  # TODO .
            else:
                res = scipy.optimize.minimize(fun=self.objective, x0=x0,
                                              method=optimizer,)
                # x = np.atleast_1d(res.x)
                # kw = self.kwargs_del(kwargs, 'x')
                # y = self.model.predict(x=res.x, **kw)[0]
                success = res.success

            # Note: length of self._multiple_hist equals number of trials
            self._multiple_hist.append(self._single_hist)

        if not success:
            message = res.message if res is not None else '---'
            self.write("+++ error message: '" + message + "'")
            if type(self).__name__ in ('Inverse'):
                self.x = [None] * self.x.size
                self.y = [None] * self.y.size

        n_trial = len(self._multiple_hist[0])
        if n_trial > 1:
            self.write('+++ Optima of all trials:')
            for i_trial, history in enumerate(self._multiple_hist):
                s = ' ' if n_trial < 10 else ''
                self.write('    ['+s+str(i_trial)+'] x: '+str(history[-1][0]))
                self.write('         y: ' + str(history[-1][1]))
                self.write('         objective: ' + str(history[-1][2]))

            # self._multiple_hist[i_trial][iLast=-1][jObj=2] -> list of best obj.
            final_objectives = [hist[-1][2] for hist in self._multiple_hist]
            if self.__class__.__name__ == 'Inverse':
                abs_final_obj = np.absolute(final_objectives)
                i_best_trial = final_objectives.index(min(abs_final_obj))
            else:
                i_best_trial = final_objectives.index(min(final_objectives))

        else:
            i_best_trial = 0

        # y: self._multiple_hist[i_best_trial][iLastEvaluation=-1][jY=1]
        history_best = self._multiple_hist[i_best_trial]
        final_best = history_best[-1]
        self.x, self.y = final_best[0], final_best[1]
        objective_best = final_best[2]
        self.write('+++ Best trial:')
        s = ' ' if i_best_trial < 10 else ''
        self.write('    [' + s + str(i_best_trial) + '] x: ' + str(self.x))
        self.write('         y: ' + str(self.y))
        self.write('         objective: ' + str(objective_best))

        return self.x, self.y

    def plot(self, select: Optional[str] = None) -> None:
        if self.y is None:
            return

        if select is None or not isinstance(select, str):
            select = 'all'
        select = select.lower()
        if select.startswith(('his', 'all')):
            self.plot_multiple_hist()
        if select.startswith(('obj', 'all')):
            self.plot_objective()
        if select.startswith(('tra', 'trj', 'all')):
            self.plot_trajectory()

    def plot_multiple_hist(self) -> None:
        for i_trial, trial in enumerate(self._multiple_hist):
            self.write('    Plot[i_trial: ' + str(i_trial) + '] ' +
                       str(trial[-1][1]) + ' = f(' +
                       str(trial[-1][0]) + ')')
            assert len(trial[0]) > 1
            
            # trial = [[x0, x1, ...], [y0, y1, ...], obj, ... ]

            n_inp, n_out = len(trial[0][0]), len(trial[0][1])
            x_seq = []
            for j_inp in range(n_inp):
                x_seq.append([P[0][j_inp] for P in trial])
            y_seq = []
            for j_out in range(n_out):
                y_seq.append([P[1][j_out] for P in trial])
            obj = [P[2] for P in trial]

            for j_inp in range(n_inp):
                plt.title('objective (x'+str(j_inp)+'), trial: '+str(i_trial))

                plt.plot(x_seq[j_inp], obj, label='obj(x'+str(j_inp)+')')
                plt.scatter([x_seq[j_inp][0]], [obj[0]],
                            color='r', marker='o', label='start')
                plt.scatter([x_seq[j_inp][-1]], [obj[-1]],
                            color='g', marker='s', label='stop')
                plt.xlabel('x' + str(j_inp))
                plt.ylabel('obj')
                plt.grid()
                plt.legend(bbox_to_anchor=(1.1, 1.025), loc='upper left')
                plt.show()

            for j_out, _y in enumerate(y_seq):
                y_lab = 'y' + str(j_out)

                plt.title(y_lab + '(x0..x' + str(n_inp-1) + ')')
                for j_inp in range(n_inp):

                    plt.plot(x_seq[j_inp], _y,
                             label=y_lab+'(x'+str(j_inp)+')')
                    plt.scatter([x_seq[j_inp][0]], [_y[0]], color='r',
                                marker='o', label='' if j_inp else 'start')
                    plt.scatter([x_seq[j_inp][-1]], [_y[-1]], color='b',
                                marker='s', label='' if j_inp else 'stop')
                plt.xlabel('x0..x' + str(n_inp-1))
                plt.ylabel(y_lab)
                plt.grid()
                plt.legend(bbox_to_anchor=(1.1, 1.025),
                           loc='upper left')
                plt.show()

                plt.title(y_lab + '(time), trial:' + str(i_trial))

                plt.plot(range(len(_y)), _y, '-', label=y_lab+'(time)')
                plt.scatter([0], [_y[0]],
                            color='r', marker='o', label='start')
                plt.scatter([len(_y)-1], [_y[-1]],
                            color='b', marker='s', label='stop')
                plt.xlabel('time')
                plt.ylabel(y_lab)
                # plt.yscale('log', nonposy='clip')
                plt.grid()
                plt.legend(bbox_to_anchor=(1.1, 1.025),
                           loc='upper left')
                plt.show()

                if n_inp > 1:
                    plt.title('trajectory(x), trial:' + str(i_trial))

                    plt.plot(x_seq[0], x_seq[1], '-', label='trajctory')
                    plt.xlabel('x0')
                    plt.ylabel('x1')
                    plt.grid()
                    plt.legend(bbox_to_anchor=(1.1, 1.025),
                               loc='upper left')
                    plt.show()

    def plot_objective(self) -> None:
        n_inp = len(self._multiple_hist[0][0][0])   # trial:0 evaluation:0 x:0
        n_trial = len(self._multiple_hist)

        plt.title('Objective (x0..x' + str(n_inp-1)+', trial[0..' +
                  str(n_trial-1) + '])')

        # type(trial) = List[Tuple[x: np.ndarray, y: np.ndarray,
        #                          obj: float]]
        for i_trial, trial in enumerate(self._multiple_hist):
            obj = [P[2] for P in trial]
            for j_inp in range(n_inp):
                x = [P[0][j_inp] for P in trial]

                plt.plot(obj, x, label='x'+str(j_inp)+'['+str(i_trial)+']')
                plt.scatter([obj[0]], [x[0]], color='r', marker='o',
                            label='' if j_inp or i_trial else 'start')
                plt.scatter([obj[-1]], [x[-1]], color='b', marker='x',
                            label='' if j_inp or i_trial else 'stop')
        plt.ylabel('x0..x' + str(n_inp-1))
        plt.xlabel('objective')
        plt.grid()
        plt.legend(bbox_to_anchor=(1.1, 1.025), loc='upper left')
        plt.show()

    def plot_trajectory(self) -> None:
        n_inp = len(self._multiple_hist[0][0][0])  # trial:0 eval:0 x_index=0
        n_out = len(self._multiple_hist[0][0][1])  # trial:0 eval:0 y_index=0

        if n_inp >= 2:
            self.write('    Trajectory of objective vs (x0, x1), all trials')
            mpl.rcParams['legend.fontsize'] = 10
            fig = plt.figure(figsize=(10, 8))
            ax = fig.gca(projection='3d')
            ax.set_xlabel('x0')
            ax.set_ylabel('x1')
            ax.set_zlabel('obj')

            # type(trial) = List[Tuple[x: np.ndarray, y: np.ndarray,
            #                          obj: float]]
            for i_trial, trial in enumerate(self._multiple_hist):
                x_seq = []
                for j_inp in range(n_inp):
                    x_seq.append([P[0][j_inp] for P in trial])
                for j_out in range(n_out):
                    x_seq.append([P[1][j_out] for P in trial])
                obj = [P[2] for P in trial]
                x0, x1 = x_seq[0], x_seq[1]

                ax.plot(x0, x1, obj, label='trial '+str(i_trial))
                ax.scatter([x0[0]], [x1[0]], [obj[0]], color='r', marker='o',
                           label='' if i_trial else 'start')
                ax.scatter([x0[-1]], [x1[-1]], [obj[-1]], color='b',
                           marker='x', label='' if i_trial else 'stop')
            ax.legend()
            plt.show()

        if n_inp >= 3:
            self.write('+++ Trajectory of x[2..n_inp] vs (x0, x1), all trials')
            for j_inp_z in range(2, n_inp):
                mpl.rcParams['legend.fontsize'] = 10
                fig = plt.figure(figsize=(10, 8))
                ax = fig.gca(projection='3d')
                ax.set_xlabel('x0')
                ax.set_ylabel('x1')
                ax.set_zlabel('x'+str(j_inp_z))

                # type of 'trial': List[Tuple[x: np.ndarray,
                #                             y: np.ndarray, obj:float]]
                for i_trial, trial in enumerate(self._multiple_hist):
                    x_seq = []
                    for j_inp in range(n_inp):
                        x_seq.append([P[0][j_inp] for P in trial])
                    x0, x1, x_z = x_seq[0], x_seq[1], x_seq[j_inp_z]

                    ax.plot(x0, x1, x_z, label='trial '+str(i_trial))
                    ax.scatter([x0[0]], [x1[0]], [x_z[0]],
                               color='g', marker='o',
                               label='' if i_trial else 'start')
                    ax.scatter([x0[-1]], [x1[-1]], [x_z[-1]],
                               color='r', marker='x',
                               label='' if i_trial else 'stop')
                ax.legend()
                plt.show()
