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

sys.path.append(os.path.abspath('..'))
from grayboxes.plotarrays import plotCurves, plotSurface, plotWireframe, \
    plotIsoMap, plotIsolines, plotVector, plotTrajectory, plotBarArrays, \
    plotBars, toRegularMesh, clip_xyz


class TestUM(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test1(self):
        # irregular grid
        x = np.random.rand(10)
        y = np.random.rand(x.size)
        z = np.sin(x) + np.cos(y) * x
        plotIsoMap(x, y, z)
        plotIsolines(x, y, z)
        plotIsoMap(x, y, z, title=r'$\alpha$ [$\degree$]')
        print('pl979')
        x, y, z = clip_xyz(x, y, z, zrange=[0.2, 1.2])
        print('pl981')
        plotIsoMap(x, y, z, title=r'$\alpha$ [$\degree$]', triangulation=True)

        self.assertTrue(True)

    def test2(self):
        # irregular grid
        x = np.random.rand(500)
        y = np.random.rand(x.size)
        vx = -(x - 0.5)
        vy = +(y - 0.5)
        plotVector(x, y, vx, vy)

        self.assertTrue(True)

    def test3(self):
        # plot of bars for two 1D arrays y(x)
        plotBars(y1=[20, 35, 30, 35, 27], y1Error=[2, 3, 4, 1, 2],
                 y2=[25, 32, 34, 20, 25], y2Error=[3, 5, 2, 3, 3],
                 y3=[21, 32, 54, 20, 15], y3Error=[3, 5, 2, 3, 3],
                 y4=[21, 32, 54, -20, 25], y4Error=[3, 5, 2, 3, 3],
                 y5=[21, 32, 24, 20, 15], y5Error=[3, 5, 2, 3, 3],
                 y6=[21, 11, 54, 20, 15], y6Error=[3, 5, 2, 3, 3],
                 labels=['x [m]', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6 [mm]'],
                 figsize=(10, 7), title='', yrange=[10, 70])

        plotBars(y1=[20, 35, 30, 35, 27], y1Error=[2, 3, 4, 1, 2],
                 y2=[25, 32, 34, 20, 25], y2Error=[3, 5, 2, 3, 3],
                 y3=[21, 32, 54, 20, 15], y3Error=[3, 5, 2, 3, 3],
                 y4=[21, 32, 54, -20, 25], y4Error=[3, 5, 2, 3, 3],
                 y5=[21, 32, 24, 20, 15], y5Error=[3, 5, 2, 3, 3],
                 y6=[21, 11, 54, 20, 15], y6Error=[3, 5, 2, 3, 3],
                 labels=['x [m]', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6 [mm]'],
                 title='', yrange=[10, 70], legendPosition=(1.1, 1))

        plotBarArrays(
                      # x = [33, 55, 88, 100, 111],
                      yArrays=[[20, 35, 30, 35, 27],
                               [25, 32, 34, 20, 25],
                               [21, 32, 54, 20, 15],
                               [21, 32, 54, -20, 25]],
                      # labels=['x [m]','y1','y2','y3','y4','y5','y6 [mm]'],
                      figsize=(10, 7), title='', yrange=[10, 70])

        self.assertTrue(True)

    def test4(self):
        x = np.linspace(0, 8, 100)
        a = np.sin(x)
        b = np.cos(x)
        c = np.tan(x * 0.5)
        d = 1.2 * np.sin(x)
        plotCurves(x, a, b, c, labels=['x', 'sin', 'cos', 'tan'])
        plotCurves(x, c,       labels=['x', 'tan'])
        plotCurves(x, a, y2=c, labels=['x', 'sin', 'tan'])
        plotCurves(x, a, b, c, d, labels=['x', 'a', 'b', 'c', 'd'],
                   legendPosition=None)

        self.assertTrue(True)

    def test5(self):
        n = 20
        z = np.linspace(-2, 2, n)
        r = z**2 + 1
        theta = np.linspace(-4 * np.pi, 4 * np.pi, n)
        x = r * np.sin(theta)
        y = r * np.cos(theta)
        z2 = z + 0.8
        z3 = z - 0.4
        plotTrajectory(x, y, z, x, y, z2, x, y, z3, zrange=[-5, -3.3])
        plotTrajectory(x, y, z, x, y, z2, x, y, z3, zrange=[-5, -3.3],
                       startPoint=True)

        self.assertTrue(True)

    def test6(self):
        # regular
        n = 16
        X, Y = np.linspace(-5, 5, n), np.linspace(-5, 5, n)
        X, Y = np.meshgrid(X, Y)
        Z = np.sin(np.sqrt(X**2 + Y**2))

        plotSurface(X, Y, Z, xrange=[-3., 4.])
        plotWireframe(X, Y, Z)
        plotIsolines(X, Y, Z, labels=['x', 'y', 'z'])
        plotIsoMap(X, Y, Z, labels=['x', 'y', 'z'])

        X, Y, Z = toRegularMesh(X, Y, Z, nx=n)
        plotSurface(X, Y, Z, xrange=[-3., 4.])
        plotWireframe(X, Y, Z)
        plotIsolines(X, Y, Z, labels=['x', 'y', 'z'])
        plotIsoMap(X, Y, Z, labels=['x', 'y', 'z'], scatter=True)
        plotIsoMap(X, Y, Z, labels=['x', 'y', 'z'])

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
