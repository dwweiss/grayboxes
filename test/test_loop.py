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
      2021-01-11 DWW
"""

import initialize
initialize.set_path()

import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Callable, Dict, Optional, Tuple
import unittest

from grayboxes.base import Base
from grayboxes.loop import Loop


class HeatConduction1D(Loop):
    def __init__(self, identifier: str = 'HeatConduction1D', 
                 nx: Optional[int] = None,
                 lx: Optional[float] = None,
                 boundary: Optional[Dict[str, Tuple[str, float]]] = None,
                 diffusivity: Optional[Callable] = None,
                 source: Optional[Callable] = None,
                 u_init: Optional[Callable] = None,
                 ) -> None:
        super().__init__(identifier)
        
        if boundary is None:
            boundary = {'W': ('D', 0.), 'E': ('D', 0.)}  # 2 x Dirichlet
        if diffusivity is None:
            diffusivity = lambda x=0., u=0., t=0.: 1.
        if lx is None:
            lx = 1.
        if nx is None:
            nx = 20
        if source is None:
            source = lambda x=0., u=0., t=0.: 0.
        if u_init is None:
            u_init = lambda x: np.sin(x * np.pi * 2)

        self.boundary: Dict[str, Tuple[str, float]] = boundary
        self.diffusivity: Callable = diffusivity
        self.lx: float = lx
        self.nx: int = nx
        self.source: Callable = source
        self.u: Optional[np.ndarray] = None
        self.u_init: Callable = u_init
        self.u_prev: Optional[np.ndarray] = None
        self.x: Optional[np.ndarray] = None
        
        if not self.silent:
            plt.clf()
    
    def initial_condition(self) -> bool:
        ok = super().initial_condition()

        # initial value of unknown          
        self.u = np.asfarray([self.u_init(x=x) for x in self.x])

        # boundary condition
        assert all(bc[0] == 'D' for bc in self.boundary.values()), \
            str(self.boundary)
        self.u[0] = self.boundary['W'][1]
        self.u[-1] = self.boundary['E'][1]

        self.u_prev = np.copy(self.u)
        
        self.a = np.asfarray([self.diffusivity(x=x, u=u, t=self.t) \
                              for x, u in zip(self.x, self.u)])

        # maximum time step size
        dx_min = np.min([self.x[i+1] - self.x[i] 
                         for i in range(0+1, len(self.x)-(1+1))])
        dt_max = dx_min**2 / (2 * self.a.min())

        self.write('+++ time step limit: ' + str(np.round(dt_max, 5)))     
        if self.dt > dt_max:
            self.write('!!! reduce time step: ' + str(self.dt) 
                       + ' -> ' + str(dt_max))
            self.dt = dt_max
        Fo = self.dt * self.a.min() / dx_min**2
        self.write('+++ Fo: ' + str(Fo))

        # unknown at previous step
        self.u_prev = np.copy(self.u)

        if not self.silent:
            plt.title(r'$\Delta t: $' + str(self.dt) + ' Fo: ' + str(Fo))
            plt.plot(self.x, self.u, linestyle='--', label='initial')
        
        return ok

    def early_stop(self):
        if self.t < 3 * self.dt:
            return False
        return np.mean(np.square((self.u-self.u_prev) 
            / (1. + self.u_prev))) < self.dt * 1e-2

    def update_transient(self) -> bool:
        ok = super().update_transient()
        
        self.a = np.asfarray([self.diffusivity(x=x, u=u, t=self.t) 
                              for x, u in zip(self.x, self.u)])
        self.u_prev, self.u = self.u, self.u_prev
                
        return ok

    def update_nonlinear(self) -> bool:
        ok = super().update_nonlinear()
        
        self.a = np.asfarray([self.diffusivity(x=x, u=u, t=self.t) 
                              for x, u in zip(self.x, self.u)])
        self.u_prev, self.u = self.u, self.u_prev

        return ok

    def pre(self, **kwargs: Any) -> bool:
        ok = super().pre()

        if not self.is_transient() and not self.is_nonlinear():
            ok = False
            self.write('??? pre(): steady + linear: skips initial_condition()')
        
        # spatial discretization
        self.x = np.linspace(0., self.lx, num=self.nx)
        
        return ok

    def task(self, **kwargs: Any):
        res: float = super().task()

        if self.x is None:
            self.write('??? self.x is None --> skips task()')
            return -1.

        n = self.x.size - 1
        for i in range(1, n):
            d2u_dx2 = (self.u_prev[i+1] - 2 * self.u_prev[i] + self.u_prev[i-1]
                      ) / np.square(self.x[1] - self.x[0])
            rhs = self.a[i] * d2u_dx2 + self.source(x=self.x[i], 
                                                    u=self.u_prev[i], t=self.t)
            self.u[i] = self.u_prev[i] + self.dt * rhs

        if not self.silent:
            plt.plot(self.x, self.u, label=str(round(self.t, 4)))
        
        return res

    def post(self, **kwargs: Any) -> bool:
        ok = super().post()

        if self.x is None:
            self.write('??? self.x is None --> skips post()')
            return False

        if not self.silent:
            plt.legend(bbox_to_anchor=(1.1, 1.02), loc='upper left')
            plt.grid()
            plt.show()
        
        return ok


class TestUM(unittest.TestCase):
    def setUp(self):
        self.foo = HeatConduction1D('conduction', nx=50)
        self.foo.set_follower([Base('follower 1'), Base('follower 2')])
        self.foo.silent = False

    def tearDown(self):
        pass


    def _test0(self):
        print('-' * 40)
        self.foo.set_nonlinear(n_it_max=300, n_it_min=100)
         
        self.foo.boundary = {'W': ('D', 0.), 'E': ('D', 0.)}
        self.foo.diffusivity = lambda x, u, t: 1.
        self.foo.source = lambda x, u, t: 0.
        self.foo.u_init = lambda x: np.sin(x * np.pi * 2)

        res = self.foo()

        self.assertAlmostEqual(res, 0.)


    def _test1(self):
        print('-' * 40)
        self.foo.set_transient(t_end=0.1)
        
        self.foo.boundary = {'W': ('D', 0.), 'E': ('D', 0.)}
        self.foo.diffusivity = lambda x, u, t: 1.
        self.foo.source = lambda x, u, t: 0.
        self.foo.u_init = lambda x: np.sin(x * np.pi * 2)

        self.foo()

        self.assertTrue(True)


    def test1a(self):
        print('-' * 40)

        u_W, u_E = 600., 300.
        self.foo.lx = 100e-6
        self.foo.set_transient(t_end=1e-1)
        self.foo.u_init = lambda x: u_E
        
        self.foo.boundary = {'W': ('D', u_W), 'E': ('D', u_E)}
        self.foo.diffusivity = lambda x, u, t: 2e-6
        self.foo.source = lambda x, u, t: 0.

        self.foo()
        
        print('t_stop:', np.round(1e3 * self.foo.t, 3), '[ms]')

        self.assertTrue(True)


    def _test2(self):
        print('-' * 40)
        self.foo.set_transient(t_end=0, n=8)
        self.foo.set_nonlinear(n_it_max=100, n_it_min=50, epsilon=0.0)

        print('foo.is_transient:', self.foo.is_transient())
        print('foo.is_nonlinear:', self.foo.is_nonlinear())
        self.foo()

        self.assertTrue(True)


    def _test3(self):
        print('-' * 40)
        self.foo.set_transient(t_end=0.1, n=8)
        self.foo.set_nonlinear(n_it_max=0, n_it_min=0, epsilon=0.0)

        print('foo.is_transient:', self.foo.is_transient())
        print('foo.is_nonlinear:', self.foo.is_nonlinear())
        self.foo()

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
