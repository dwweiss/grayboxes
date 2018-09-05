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

    def __init__(self, identifier: str='Loop') -> None:
        super().__init__(identifier)

        # iteration settings are only relevant if nItMax > 0
        self.it = 0                     # actual number of iterations
        self.nItMin = 0                 # minimum number of iterations
        self.nItMax = 0                 # maximum number of iterations
        self.epsilon = 0.               # maximum residuum tolerated
        self.omega = 1.                 # relaxation factor

        # transient settings are only relevant if tEnd > 0
        self.t = 0.                     # actual time
        self.tBegin = 0.                # start time
        self.tEnd = 0.                  # final time
        self.dt = 1e-2                  # time step size
        self.theta = 0.5                # time discretization scheme

    def __str__(self) -> str:
        s = super().__str__()

        if self.isNonLinear():
            s += '\n'
            s += "{nonLinear: {id: '" + self.identifier + "', " + \
                "it: '" + str(self.it) + "', nItMin: '" + str(self.nItMin) + \
                "', nItMax: '" + str(self.nItMax) + "', epsilon: '" + \
                str(self.epsilon) + "', omega: '" + str(self.omega) + "'}}"
        if self.isTransient():
            if self.treeLevel() == 0:
                s += '\n'
                s += "{transient: {t: '" + '{:f}'.format(self.t) + \
                    "', tBegin: '" + str(self.tBegin) + "', tEnd: '" + \
                    str(self.tEnd) + "', dt: '" + str(self.dt) + \
                    "', theta: '" + str(self.theta) + "'}}"
            else:
                s += "{steady}"
        return s

    def isNonLinear(self) -> bool:
        return self.nItMax > 0

    def isTransient(self) -> bool:
        return self.tEnd > 0.0

    def setNonLinear(self, nItMin: int=0, nItMax: int=0, epsilon: float=0.0,
                     omega: float=1.0) -> None:
        self.nItMax = nItMax if nItMax is not None else 0
        self.nItMax = np.clip(self.nItMax, 0, int(1e6))

        if self.nItMax > 0:
            nItMin = np.clip(nItMin, 0, nItMax)
            epsilon = np.clip(epsilon, 0.0, 1.0)
            omega = np.clip(omega, 0.0, 2.0)

            self.nItMin = nItMin
            self.epsilon = epsilon
            self.omega = omega

    def setTransient(self, tBegin: float=0.0, tEnd: float=0.0, dt: float=0.0,
                     theta: float=0.5, n: int=100) -> None:
        self.tEnd = tEnd if tEnd is not None else 0.0
        self.tEnd = np.clip(self.tEnd, 0, int(1e6))

        if self.tEnd > 0.0:
            if dt < 1e-20:
                assert n > 0, 'n must be greater zero'
            dt = (self.tEnd - tBegin) / n
            theta = np.clip(theta, 0.0, 1.0)
            tBegin = np.clip(tBegin, 0.0, tEnd)

            self.tBegin = tBegin
            self.dt = dt
            self.theta = theta

    def initialCondition(self) -> None:
        for x in self.followers:
            x.initialCondition()

    def updateNonLinear(self) -> None:
        for x in self.followers:
            x.updateNonLinear()

    def updateTransient(self) -> None:
        for x in self.followers:
            x.updateTransient()
        self.t = self.root().t
        self.dt = self.root().dt
        self.theta = self.root().theta

    def control(self, **kwargs: Any) -> float:
        """
        Kwargs:
            Keyword arguments passed to super.control() and task()

        Returns:
            Residuum from range 0.0..1.0 indicating error of task
        """
        # steady and linear: call control() of base class
        if not self.isTransient() and not self.isNonLinear():
            return super().control(**kwargs)

        if self.isRoot():
            exe_time = time() - self._exe_time_start
            if exe_time >= self._min_exe_time_shown:
                self.write('    Execution time: {:2f} s'.format(round(exe_time,
                           2)))
            self.exe_time_start = time()
        if not self.isTransient():
            s = ' (steady & non-linear: ' + str(self.nItMax) + ')'
        else:
            s = ' (transient: ' + str(self.tEnd)
            if self.isNonLinear():
                s += ' & non-linear: ' + str(self.nItMax) + ')'
            else:
                s += ' & linear)'
        self.write('=== Control' + s)

        ###############################
        def _nonLinearIteration() -> float:
            self.it = -1
            _res = np.inf
            while self.it < self.nItMax:
                self.it += 1
                self.write('+++ Iteration: ' + str(self.it))
                self.updateNonLinear()
                _res = self.task(**kwargs)

                if _res <= self.epsilon and self.it >= self.nItMin:
                    break
            return _res
        ###############################

        # initialize arrays of unknowns, coordinates and boundary cond.
        self.initialCondition()

        if not self.isTransient():
            ###############################
            res = _nonLinearIteration()             # steady, non-linear
            ###############################
        else:
            self.t = 0.0
            res = np.inf
            while (self.t + 1e-10) < self.tEnd:
                self.t += self.dt
                self.write('### Physical time: {:f}'.format(self.t))
                self.updateTransient()
                if self.isNonLinear():
                    ###############################
                    res = _nonLinearIteration()  # transient, non-linear
                    ###############################
                else:
                    res = self.task(**kwargs)        # transient, linear

        if self.isRoot():
            exe_time = time() - self._exe_time_start
            if exe_time >= self._min_exe_time_shown:
                self.write('    Execution time: {:2f}'.format(round(exe_time,
                           2)))
            self._exe_time_start = time()
        self.write('=== Post-processing')
        return res
