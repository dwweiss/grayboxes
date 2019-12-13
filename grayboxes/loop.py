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
      2019-11-29 DWW
"""

from time import time
import numpy as np
from typing import Any

from grayboxes.base import Base


class Loop(Base):
    """
    Controls transient and iterative loops. Instances of this class are
    high-level objects controlling execution of follower and cooperator 
    objects
    """

    def __init__(self, identifier: str = 'Loop') -> None:
        super().__init__(identifier)

        # iteration settings are only relevant if n_it_max > 0
        self.it: int = 0                # actual number of iterations
        self.n_it_min: int = 0          # minimum number of iterations
        self.n_it_max: int = 0          # maximum number of iterations
        self.epsilon: float = 0.        # maximum residuum tolerated
        self.omega: float = 1.          # relaxation factor

        # transient settings are only relevant if t_end > 0
        self.t: float = 0.              # actual time
        self.t_begin: float = 0.        # start time
        self.t_end: float = 0.          # final time
        self.dt: float = 1e-2           # time step size
        self.theta: float = 0.5         # time discretization scheme

    def __str__(self) -> str:
        s = super().__str__()

        if self.is_nonlinear():
            s += '\n'
            s += "{nonLinear: {id: '" + self.identifier + "', " + \
                "it: '" + str(self.it) + "', nItMin: '" + str(self.n_it_min) +\
                "', nItMax: '" + str(self.n_it_max) + "', epsilon: '" + \
                 str(self.epsilon) + "', omega: '" + str(self.omega) + "'}}"
        if self.is_transient():
            if self.tree_level() == 0:
                s += '\n'
                s += "{transient: {t: '" + '{:f}'.format(self.t) + \
                    "', tBegin: '" + str(self.t_begin) + "', t_end: '" + \
                     str(self.t_end) + "', dt: '" + str(self.dt) + \
                    "', theta: '" + str(self.theta) + "'}}"
            else:
                s += "{steady}"
        return s

    def is_nonlinear(self) -> bool:
        return self.n_it_max > 0

    def is_transient(self) -> bool:
        return self.t_end > 0.0

    def set_nonlinear(self, n_it_min: int = 0, n_it_max: int = 0,
                      epsilon: float = 0., omega: float = 1.) -> None:
        self.n_it_max = n_it_max if n_it_max is not None else 0
        self.n_it_max = np.clip(self.n_it_max, 0, int(1e6))

        if self.n_it_max > 0:
            n_it_min = np.clip(n_it_min, 0, n_it_max)
            epsilon = np.clip(epsilon, 0.0, 1.0)
            omega = np.clip(omega, 0.0, 2.0)

            self.n_it_min = n_it_min
            self.epsilon = epsilon
            self.omega = omega

    def set_transient(self, t_begin: float = 0., t_end: float = 0., 
                      dt: float = 0., theta: float = .5, n: int = 100) -> None:
        """
        transient loops can be set with two parameter combinations:
            1. dt and n
            2. dt and t_end 
        optionally, the start time t_begin can be set
            
        """
        self.t_end = t_end if t_end is not None else 0.
        self.t_end = np.clip(self.t_end, 0., int(1e9))
        self.t_begin = t_begin if t_begin is not None else 0.
        self.t_begin = np.clip(self.t_begin, 0., self.t_end)

        if self.t_end == 0.:
            if dt > 0. and n > 0:
                self.t_end = self.t_begin + n * dt

        if self.t_end > 0.:
            if dt < 1e-20:
                assert n > 0, 'n must be greater zero'
            dt = (self.t_end - t_begin) / n
            theta = np.clip(theta, 0., 1.)
            t_begin = np.clip(t_begin, 0., t_end)

            self.t_begin = t_begin
            self.dt = dt
            self.theta = theta                
            
    def initial_condition(self) -> bool:
        ok = True
        for node in self.followers:
            if node:
                if not node.initial_condition():
                    ok = False
        return ok

    def update_nonlinear(self) -> bool:
        ok = True
        for node in self.followers:
            if node:
                if not node.update_nonlinear():
                    ok = False
        return ok

    def update_transient(self) -> bool:
        ok = True
        for node in self.followers:
            if node:
                if not node.update_transient():
                    ok = False
        self.t = self.root().t
        self.dt = self.root().dt
        self.theta = self.root().theta

        return ok

    def control(self, **kwargs: Any) -> float:
        """
        Kwargs:
            Keyword arguments passed to super.control() and task()

        Returns:
            Residuum from range 0.0..1.0 indicating error of task
        """
        # steady and linear: call control() of class Base
        if not self.is_transient() and not self.is_nonlinear():
            return super().control(**kwargs)

        if self.is_root():
            exe_time = time() - self._exe_time_start
            if exe_time >= self._min_exe_time_shown:
                self.write('    Execution time: {:2f} s'.format(round(exe_time,
                           2)))
            self._exe_time_start = time()
        if not self.is_transient():
            s = ' (steady & non-linear: ' + str(self.n_it_max) + ')'
        else:
            s = ' (transient: ' + '{:f}'.format(self.t_end)
            if self.is_nonlinear():
                s += ' & non-linear: ' + str(self.n_it_max) + ')'
            else:
                s += ' & linear)'
        self.write('=== Control' + s)

        ###############################
        def _nonlinear_iteration() -> float:
            self.it = -1
            _res = np.inf
            while self.it < self.n_it_max:
                self.it += 1
                self.write('+++ Iteration: ' + str(self.it))
                self.update_nonlinear()
                _res = self.task(**kwargs)

                if _res <= self.epsilon and self.it >= self.n_it_min:
                    break
            return _res
        ###############################

        # initialize arrays of unknowns, coordinates and boundary cond.
        self.initial_condition()

        if not self.is_transient():
            ###############################
            res = _nonlinear_iteration()             # steady, non-linear
            ###############################
        else:
            self.t = 0.0
            res = np.inf
            while (self.t + 1e-10) < self.t_end:
                self.t += self.dt
                self.write('### Physical time: {:f}'.format(self.t))
                self.update_transient()
                if self.is_nonlinear():
                    ###############################
                    res = _nonlinear_iteration()  # transient, non-linear
                    ###############################
                else:
                    res = self.task(**kwargs)        # transient, linear

        if self.is_root():
            exe_time = time() - self._exe_time_start
            if exe_time >= self._min_exe_time_shown:
                self.write('    Execution time: {:2f}'.format(round(exe_time,
                           2)))
            self._exe_time_start = time()
        self.write('=== Post-processing')

        return res
