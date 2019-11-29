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

import __init__
__init__.init_path()

import unittest
import numpy as np
import matplotlib.pyplot as plt

from grayboxes.base import Base
from grayboxes.loop import Loop


class HeatConduction(Loop):
    def __init__(self, identifier: str = 'test') -> None:
        super().__init__(identifier)
        
        self.x = None
        self.u = None
        self.u_prev = None
        self.a = None
        self.source = 0.0
        plt.clf()

    def initial_condition(self) -> bool:
        ok = super().initial_condition()
        
        self.x = np.linspace(0., 1., num=20)
        self.u = np.sin(self.x * np.pi * 2)
        self.u[0], self.u[-1] = 0., 0.
        self.u_prev = np.copy(self.u)
        self.a = np.linspace(1, 1, num=len(self.x))
        plt.plot(self.x, self.u, linestyle='--', label='initial')
        
        return ok

    def update_transient(self) -> bool:
        ok = super().update_transient()
        
        self.u_prev, self.u = self.u, self.u_prev
        
        return ok

    def update_nonlinear(self) -> bool:
        ok = super().update_nonlinear()
        
        self.a = 1 + 0 * self.u
        
        return ok

    def pre(self) -> bool:
        ok = super().pre()

        if not self.is_transient() and not self.is_nonlinear():
            ok = False
            self.write('??? pre(): steady + linear: skips initial_condition()')
            
        return ok

    def task(self):
        super().task()

        if self.x is None:
            self.write('??? self.x is None --> skips task()')
            return -1.

        n = self.x.size - 1
        for i in range(1, n):
            d2u_dx2 = (self.u_prev[i + 1] - 2 * self.u_prev[i] +
                       self.u_prev[i - 1]) / (self.x[1] - self.x[0]) ** 2
            rhs = self.a[i] * d2u_dx2 + self.source
            self.u[i] = self.u_prev[i] + self.dt * rhs

        plt.plot(self.x, self.u, label=str(round(self.t, 4)))
        
        return 0.0

    def post(self) -> bool:
        ok = super().post()

        if self.x is None:
            self.write('??? self.x is None --> skips post()')
            return False

        plt.legend(bbox_to_anchor=(1.1, 1.02), loc='upper left')
        plt.grid()
        plt.show()
        
        return ok


class TestUM(unittest.TestCase):
    def setUp(self):
        self.foo = HeatConduction('conduction')
        self.foo.set_follower([Base('follower 1'), Base('follower 2')])
        self.foo.silent = False


    def tearDown(self):
        pass


    def test1(self):
        print('-' * 40)
        print('foo.isTransient:', self.foo.is_transient())
        print('foo.isNonLinear:', self.foo.is_nonlinear())
        print()
        self.foo()

        self.assertTrue(True)


    def test2(self):
        print('-' * 40)
        self.foo.set_transient(t_end=0, n=8)
        self.foo.set_nonlinear(n_it_max=5, n_it_min=3, epsilon=0.0)

        print('foo.isTransient:', self.foo.is_transient())
        print('foo.isNonLinear:', self.foo.is_nonlinear())
        self.foo()

        self.assertTrue(True)


    def test3(self):
        print('-' * 40)
        self.foo.set_transient(t_end=0.1, n=8)
        self.foo.set_nonlinear(n_it_max=0, n_it_min=0, epsilon=0.0)

        print('foo.is_transient:', self.foo.is_transient())
        print('foo.is_nonLinear:', self.foo.is_nonlinear())
        self.foo()

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()

