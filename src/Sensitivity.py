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
      2018-02-23 DWW
"""

import numpy as np
import matplotlib.pyplot as plt

from Forward import Forward
from plotArrays import plotBarArrays


class Sensitivity(Forward):
    """
    Analyses the sensitivity of the generic model for a given input range:
    x_analysis = x_ref +/- Delta x

    Examples:
        foo = Sensitivity()

        X = [[... ]]  input of training
        Y = [[... ]]  target of training
        xCross = [(x00, x01), (x10, x11), ..., (x90, x91)]
        ranges = [(x0min, x0max), (x1min, x1max)]

        dy_dx = foo(X=X, Y=Y, x=xCross)              # training and sensitivity
        dy_dx = foo(x=xCross)        # sensitivity only, x has all cross points
        dy_dx = foo(ranges=ranges, cross=5) # sens. only, compute x from ranges

    Note:
        Parent class Forward is not derived from class Model.
        The instance of the generic model has to be assigned to: self.model
    """

    def __init__(self, identifier='Sensitivity', model=None, f=None,
                 trainer=None):
        """
        Args:
            identifier (string):
                class identifier

            model (method):
                generic model

            f (method):
                white box model f(x)

            trainer (method):
                training method
        """
        super().__init__(identifier, model=model, f=f, trainer=trainer)

        self.axisIndices = None         # point indices for which x[j] is equal
        self.dy_dx = None

    def task(self, **kwargs):
        super().task(**self.kwargsDel(kwargs, 'x'))

        if self.model.x is None:
            return None, None

        # (x, y)_ref is stored as: (self.model.x[0], self.model.y[0])
        x, y = self.model.x, self.model.y
        xRef = x[0]
        nPoint, nInp, nOut = x.shape[0], x.shape[1], y.shape[1]

        self.indicesWithEqualXj = [[] for _ in range(nInp)]

        # i is point index, j is input index and k is output index
        for i in range(1, nPoint):
            for j in range(nInp):
                if np.isclose(x[0, j], x[i, j]):
                    self.indicesWithEqualXj[j].append(i)

        self.dy_dx = np.full((nInp, nOut), np.inf)
        jCenter = nInp // 2
        for k in range(nOut):
            for j in range(nInp):
                xx, yy = [], []
                for i in range(nPoint):
                    if i not in self.indicesWithEqualXj[j]:
                        xx.append(x[i, j])
                        yy.append(y[i, k])
                    yy = [a for _, a in sorted(zip(xx, yy))]
                    xx = sorted(xx)
                dx = (xx[jCenter+1] - xx[jCenter-1]) * 0.5
                grad = np.gradient(yy, dx)
                self.dy_dx[j, k] = grad[jCenter]
        s = np.array2string(self.dy_dx).replace(' ', '').replace('\n',
                                                                 '\n' + ' '*10)
        self.write('    grad: ', s[1:-1])

        return xRef, self.dy_dx

    def plot(self):
        if not self.silent:
            nPoint, nInp = self.model.x.shape
            nOut = self.model.y.shape[1]

            for k in range(nOut):
                for j in range(nInp):
                    plt.title('y' + str(k) + ' (x' + str(j) + ')')
                    plt.xlabel('x' + str(j))
                    plt.ylabel('y' + str(k))
                    xx, yy = [], []
                    for i in range(nPoint):
                        if i not in self.indicesWithEqualXj[j]:
                            xx.append(self.model.x[i, j])
                            yy.append(self.model.y[i, k])
                        yy = [a for _, a in sorted(zip(xx, yy))]
                        xx = sorted(xx)
                    plt.plot(xx, yy, '-o')
                    plt.grid()
                    plt.legend(bbox_to_anchor=(1.1, 1.03), loc='upper left')
                    plt.show()

            for k in range(nOut):
                plt.title('y' + str(k) + ' (x0..' + str(nInp-1) + ')')
                for j in range(nInp):
                    plt.xlabel('x0..' + str(nInp-1))
                    plt.ylabel('y' + str(k))
                    xx, yy = [], []
                    for i in range(nPoint):
                        if i not in self.indicesWithEqualXj[j]:
                            xx.append(self.model.x[i, j])
                            yy.append(self.model.y[i, k])
                        yy = [a for _, a in sorted(zip(xx, yy))]
                        xx = sorted(xx)
                    plt.plot(xx, yy, '-o', label='y'+str(k)+'(x'+str(j)+')')
                plt.grid()
                plt.legend(bbox_to_anchor=(1.1, 1.04), loc='upper left')
                plt.show()

            if nOut > 1:
                for j in range(nInp):
                    plt.title('y0..' + str(nOut-1) + ' (x' + str(j) + ')')
                    for k in range(nOut):
                        plt.xlabel('x' + str(j))
                        plt.ylabel('y0..' + str(nOut-1))
                        xx, yy = [], []
                        for i in range(nPoint):
                            if i not in self.indicesWithEqualXj[j]:
                                xx.append(self.model.x[i, j])
                                yy.append(self.model.y[i, k])
                            yy = [a for _, a in sorted(zip(xx, yy))]
                            xx = sorted(xx)
                        plt.plot(xx, yy, '-o',
                                 label='y'+str(k)+' (x'+str(j)+')')
                    plt.grid()
                    plt.legend(bbox_to_anchor=(1.1, 1.04), loc='upper left')
                    plt.show()

            plotBarArrays(yArrays=self.dy_dx.T, legendPosition=(1.1, 1.03),
                          title=r'Gradient $d y_k \ / \ d x_j$', grid=True,
                          figsize=(6, 4))


# Examples ####################################################################

if __name__ == '__main__':
    ALL = 1

    def f(self, x, c0=1, c1=1, c2=1, c3=1, c4=1, c5=1, c6=1, c7=1):
        return np.sin(x[0]) + (x[1] - 1)**2

    if 0 or ALL:
        s = 'Sensitivity 1'
        print('-' * len(s) + '\n' + s + '\n' + '-' * len(s))

        foo = Sensitivity(f=f)
        xRef, dy_dx = foo(ranges=[(2, 3), (3, 4), (4, 5)], cross=3)
        if dy_dx.shape[0] == 1 or dy_dx.shape[1] == 1:
            dy_dx = dy_dx.tolist()
        print('dy_dx:', dy_dx)
        print('x_ref:', xRef)
