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
from grayboxes.move import Move
from grayboxes.xyz import xyz


way = [xyz(.0,  .0, .0),     #   ^
       xyz(.1,  .1, .0),     # +1|      /\
       xyz(.2,  .2, .0),     #   |    /    \
       xyz(.3,  .1, .0),     #   |  /        \             0.8
       xyz(.4,  .0, .0),     # 0 |/----0.2-----\----0.6-----/-->
       xyz(.5, -.1, .0),     #   |            0.4\        /    x
       xyz(.6, -.2, .0),     #   |                 \    /
       xyz(.7, -.1, .0),     # -1|                   \/
       xyz(.8,  .0, .0)]     #   | trajectory W=W(t)

rot = [xyz(20 * np.sin(i*3), 4*i-20, i*i-30) for i in range(len(way))]
print(len(rot), [str(rot[i]) for i in range(9)])

foo = Move('')
speed = 0.8
foo.setTrajectory(way, rot=rot, speed=speed)

print('-' * 40)
print('test:', foo)
print('-' * 40)


class TestUM(unittest.TestCase):
    def setUp(self):
        pass
 
    def tearDown(self):
        pass
 
    def test1(self):
        speed = 0.8
        foo.setTrajectory(way, rot=rot, speed=speed)
        
        print('-' * 40)
        print('test:', foo)
        print('-' * 40)

        self.assertTrue(True)
        
    def test2(self):
        i = foo.iWayPointAhead(t=0.3)
        print('t=0.3 i:', i)
        i = foo.iWayPointAhead(t=0.0)
        print('t=0 i:', i)
        print('-' * 40)

        tRange = np.linspace(0, 5, 8)
        for t in tRange:
            i = foo.iWayPointAhead(t)
            print('t:', t, 'i:', i)
        print('-' * 40)

        tEnd = np.sqrt((2*(1-(-1)))**2 + 0.8**2) / speed
        foo.setTransient(tEnd=tEnd, n=100)

        foo()

        print('-' * 40)
        foo.plot()
        print('-' * 40)

        self.assertTrue(True)

    def test3(self):
        print('foo._wayPoints[-1].t:', foo._wayPoints[-1].t)
        T = np.linspace(0., 2, 100)
        p = foo.position(t=0.1)
        print('p:', p)

        x = [p.x for p in foo._wayPoints]
        y = [p.y for p in foo._wayPoints]
        t = [p.t for p in foo._wayPoints]

        P = []
        for time in T:
            p = foo.position(time)
            p.t = time
            P.append(p)
        X = [p.x for p in P]
        Y = [p.y for p in P]
        T = [p.t for p in P]

        if 0:
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