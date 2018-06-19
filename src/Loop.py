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
      2018-01-22 DWW
"""

from time import time
import numpy as np

from Base import Base


class Loop(Base):
    """
    Controls transient and iterative loops. Instances of this class are
    high-level objects controlling execution of follower and cooperator objects
    """

    def __init__(self, identifier='Loop'):
        super().__init__(identifier)

        # iteration settings are only relevant if nItMax > 0
        self.it = 0                     # actual number of iterations
        self.nItMin = 0                 # minimum number of iterations
        self.nItMax = 0                 # maximum number of iterations
        self.epsilon = 0.0              # maximum residuum tolerated
        self.omega = 1.0                # relaxation factor

        # transient settings are only relevant if tEnd > 0
        self.t = 0.0                    # actual time
        self.tBegin = 0.                # start time
        self.tEnd = 0.                  # final time
        self.dt = 1e-2                  # time step size
        self.theta = 0.5                # time discretisation sheme

    def __str__(self):
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

    def isNonLinear(self):
        return self.nItMax > 0

    def isTransient(self):
        return self.tEnd > 0.0

    def setNonLinear(self, nItMin=0, nItMax=0, epsilon=0.0, omega=1.0):
        self.nItMax = nItMax if nItMax is not None else 0
        self.nItMax = np.clip(self.nItMax, 0, int(1e6))

        if self.nItMax > 0:
            nItMin = np.clip(nItMin, 0, nItMax)
            epsilon = np.clip(epsilon, 0.0, 1.0)
            omega = np.clip(omega, 0.0, 2.0)

            self.nItMin = nItMin
            self.epsilon = epsilon
            self.omega = omega

    def setTransient(self, tBegin=0.0, tEnd=0.0, dt=0.0, theta=0.5, n=100):
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

    def initialCondition(self):
        for x in self.followers:
            x.initialCondition()

    def updateNonLinear(self):
        for x in self.followers:
            x.updateNonLinear()

    def updateTransient(self):
        for x in self.followers:
            x.updateTransient()
        self.t = self.root().t
        self.dt = self.root().dt
        self.theta = self.root().theta

    def control(self, **kwargs):
        """
        Args:
            kwargs (dict, optional):
                keyword arguments passed to task()

        Return:
            (float):
                residuum from range 0.0..1.0 indicating error of task
        """
        # steady and linear: call control() of base class
        if not self.isTransient() and not self.isNonLinear():
            return super().control(**kwargs)

        if self.isRoot():
            execTime = time() - self._execTimeStart
            if execTime >= self._minExecTimeShown:
                self.write('    Execution time: {:2f} s'.format(round(execTime,
                           2)))
            self.execTimeStart = time()
        if not self.isTransient():
            s = ' (steady & non-linear: ' + str(self.nItMax) + ')'
        else:
            s = ' (transient: ' + str(self.tEnd)
            if self.isNonLinear():
                s += ' & non-linear: ' + str(self.nItMax) + ')'
            else:
                s += ' & linear)'
        self.write('=== Control', s)

        ###############################
        def _nonLinearIteration(self):
            self.it = -1
            while self.it < self.nItMax:
                self.it += 1
                self.write('+++ Iteration: ', self.it)
                self.updateNonLinear()
                res = self.task(**kwargs)

                if res <= self.epsilon and self.it >= self.nItMin:
                    break
            return res
        ###############################

        # initialize arrays of unknowns, coordinates and boundary conditions
        self.initialCondition()

        if not self.isTransient():
            ###############################
            res = _nonLinearIteration(self)             # steady, non-linear
            ###############################
        else:
            self.t = 0.0
            while (self.t + 1e-10) < self.tEnd:
                self.t += self.dt
                self.write('### Physical time: {:f}'.format(self.t))
                self.updateTransient()
                if self.isNonLinear():
                    ###############################
                    res = _nonLinearIteration(self)     # transient, non-linear
                    ###############################
                else:
                    res = self.task(**kwargs)           # transient, linear

        if self.isRoot():
            execTime = time() - self.execTimeStart
            if execTime >= self._minExecTimeShown:
                self.write('    Execution time: {:2f}'.format(round(execTime,
                           2)))
            self.execTimeStart = time()
        self.write('=== Post-processing')
        return res


# Examples ####################################################################

if __name__ == '__main__':
    ALL = 1
    import matplotlib.pyplot as plt

    class HeatConduction(Loop):
        def __init__(self, identifier='test'):
            super().__init__(identifier)
            self.x = None
            self.u = None
            self.uPrev = None
            self.a = None
            self.source = 0.0
            plt.clf()

        def initialCondition(self):
            super().initialCondition()
            self.x = np.linspace(0., 1., num=20)
            self.u = np.sin(self.x * np.pi * 2)
            self.u[0], self.u[-1] = (0., 0.)
            self.uPrev = np.copy(self.u)
            self.a = np.linspace(1, 1, num=len(self.x))
            plt.plot(self.x, self.u, linestyle='--', label='initial')

        def updateTransient(self):
            super().updateTransient()
            self.uPrev, self.u = self.u, self.uPrev

        def updateNonlinear(self):
            super().updateNonlinear()
            self.a = 1 + 0 * self.u

        def task(self):
            super().task()

            n = self.x.size - 1
            for i in range(1, n):
                d2udx2 = (self.uPrev[i+1] - 2 * self.uPrev[i] +
                          self.uPrev[i-1]) / (self.x[1] - self.x[0])**2
                rhs = self.a[i] * d2udx2 + self.source
                self.u[i] = self.uPrev[i] + self.dt * rhs

            plt.plot(self.x, self.u, label=str(round(self.t, 4)))
            return 0.0

        def post(self):
            super().post()

            plt.legend(bbox_to_anchor=(1.1, 1.02), loc='upper left')
            plt.grid()
            plt.show()

    foo = HeatConduction()
    foo.setFollower([Base('follower 1'), Base('follower 2')])
    foo.silent = False

    if 0 or ALL:
        foo.setTransient(tEnd=0, n=8)
        foo.setNonLinear(nItMax=5, nItMin=3, epsilon=0.0)

        print('foo.isTransient:', foo.isTransient())
        print('foo.isNonLinear:', foo.isNonLinear())
        foo()

    if 0 or ALL:
        foo.setTransient(tEnd=0.1, n=8)
        foo.setNonLinear(nItMax=0, nItMin=0, epsilon=0.0)

        print('foo.isTransient:', foo.isTransient())
        print('foo.isNonLinear:', foo.isNonLinear())
        foo()
