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
"""

import __init__
__init__.init_path()

import unittest
import numpy as np
import matplotlib.pyplot as plt

from grayboxes.move import Move
from grayboxes.xyz import xyz


wayp = [xyz(.0,  .0, .0),     #   ^
        xyz(.1,  .1, .0),     # +1|      /\
        xyz(.2,  .2, .0),     #   |    /    \
        xyz(.3,  .1, .0),     #   |  /        \             0.8
        xyz(.4,  .0, .0),     # 0 |/----0.2-----\----0.6-----/-->
        xyz(.5, -.1, .0),     #   |            0.4\        /    x
        xyz(.6, -.2, .0),     #   |                 \    /
        xyz(.7, -.1, .0),     # -1|                   \/
        xyz(.8,  .0, .0)]     #   | trajectory W=W(t)


class TestUM(unittest.TestCase):
    def setUp(self):
        self.orien = [xyz(20 * np.sin(i*3), 4*i-20, i*i-30) 
                      for i in range(len(wayp))]
        print(len(self.orien), [str(self.orien[i]) for i in range(9)])
        
        self.foo = Move('move')
        self.speed = 0.8
        self.foo.set_trajectory(waypoints=wayp, orientations=self.orien, 
                                speed=self.speed)
        
        print('-' * 40)
        print('test:', self.foo)
        print('-' * 40)

    def tearDown(self):
        pass

    def test1(self):
        self.foo.set_trajectory(wayp, orientations=self.orien, speed=self.speed)

        print('-' * 40)
        print('test:', self.foo)
        print('-' * 40)

        self.assertTrue(True)

    def test2(self):
        i_way = self.foo.i_waypoint_ahead(t=0.3)
        print('t=0.3 i:', i_way)
        i_way = self.foo.i_waypoint_ahead(t=0.0)
        print('t=0 i:', i_way)
        print('-' * 40)

        t_range = np.linspace(0, 5, 8)
        for t in t_range:
            i_way = self.foo.i_waypoint_ahead(t)
            print('t:', t, 'i:', i_way)
        print('-' * 40)

        t_end = np.sqrt((2*(1-(-1)))**2 + 0.8**2) / self.speed
        self.foo.set_transient(t_end=t_end, n=100)

        self.foo()

        print('-' * 40)
        self.foo.plot()
        print('-' * 40)

        self.assertTrue(True)

    def test3(self):
        print('foo._waypoints[-1].t:', self.foo._waypoints[-1].t)
        T = np.linspace(0., 2, 100)
        p = self.foo.position(t=0.1)
        print('p:', p)

        x = [p.x for p in self.foo._waypoints]
        y = [p.y for p in self.foo._waypoints]
        t = [p.t for p in self.foo._waypoints]

        P = []
        for time in T:
            p = self.foo.position(time)
            p.t = time
            P.append(p)
        X = [p.x for p in P]
        Y = [p.y for p in P]
        T = [p.t for p in P]

        if 1:
            plt.title('trajectory 1(3)')
            plt.plot(X, Y, label='position()')
            plt.scatter(x, y, label='wayPoints')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend(bbox_to_anchor=(1.1, 1.025), loc='upper left')
            plt.grid()
            plt.show()

            plt.title('trajectory 2(3)')
            plt.plot(X, T, label='position()')
            plt.scatter(x, t, label='wayPoints')
            plt.xlabel('x')
            plt.ylabel('t')
            plt.legend(bbox_to_anchor=(1.1, 1.025), loc='upper left')
            plt.grid()
            plt.show()

            plt.title('trajectory 3(3)')
            plt.plot(Y, T, label='position()')
            plt.scatter(y, t, label='wayPoints')
            plt.xlabel('y')
            plt.ylabel('t')
            plt.legend(bbox_to_anchor=(1.1, 1.025), loc='upper left')
            plt.grid()
            plt.show()

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
