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
      2018-08-16 DWW
"""

import unittest	
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('..'))
from grayboxes.base import Base
from grayboxes.loop import Loop


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


foo = HeatConduction('conduction')
foo.setFollower([Base('follower 1'), Base('follower 2')])
foo.silent = False


class TestUM(unittest.TestCase):
    def setUp(self):
        pass
 
    def tearDown(self):
        pass
 
    def test1(self):
        print('foo.isTransient:', foo.isTransient())
        print('foo.isNonLinear:', foo.isNonLinear())

        self.assertTrue(True)

    def test2(self):
        foo.setTransient(tEnd=0, n=8)
        foo.setNonLinear(nItMax=5, nItMin=3, epsilon=0.0)

        print('foo.isTransient:', foo.isTransient())
        print('foo.isNonLinear:', foo.isNonLinear())
        foo()

        self.assertTrue(True)

    def test3(self):
        foo.setTransient(tEnd=0.1, n=8)
        foo.setNonLinear(nItMax=0, nItMin=0, epsilon=0.0)

        print('foo.isTransient:', foo.isTransient())
        print('foo.isNonLinear:', foo.isNonLinear())
        foo()

        self.assertTrue(True)
        
        
if __name__ == '__main__':
    unittest.main()

