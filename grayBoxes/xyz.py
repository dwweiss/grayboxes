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
      2017-11-21 DWW
"""

from math import sqrt, sin, cos, pi
import numpy as np


class xyz(object):
    """
    Point in 3D space (x, y, z)
    """

    def __init__(self, x=0., y=0., z=0., point=None):
        """
        Args:
            x, y, z (float, optional):
                coordinates of point in 3D space [m]

            point (xyz or xyzt, optional):
                'point' will be assigned to this object if 'point' is not None
        """
        if isinstance(point, xyz):
            self.x, self.y, self.z = point.x, point.y, point.z
        elif point is None:
            self.x = x if x is not None else 0.
            self.y = y if y is not None else 0.
            self.z = z if z is not None else 0.
        else:
            self.x, self.y, self.z = None, None, None
            print("??? xyz: point is not of type 'xyz', type=", type(point))

    def __add__(self, P):
        if isinstance(P, xyz):
            return xyz(self.x + P.x, self.y + P.y, self.z + P.z)
        else:
            return xyz(self.x + P, self.y + P, self.z + P)

    def __sub__(self, P):
        if isinstance(P, xyz):
            return xyz(self.x - P.x, self.y - P.y, self.z - P.z)
        else:
            return xyz(self.x - P, self.y - P, self.z - P)

    def __mul__(self, multiplier):
        if isinstance(multiplier, xyz):
            return self.dot(multiplier)
        else:
            return xyz(self.x * multiplier, self.y * multiplier,
                       self.z * multiplier)

    def __eq__(self, other):
        return np.isclose(self.x, other.x) and np.isclose(self.y, other.y) \
            and np.isclose(self.z, other.z)

    def at(self, i):
        """
        Accesses point components by index

        Args:
            i (int):
                index of component (0: x, 1: y, 2: z)

        Returns:
            (float):
                value of component
        """
        if i == 0:
            return self.x
        elif i == 1:
            return self.y
        elif i == 2:
            return self.z
        else:
            print('??? parameter of at() is out of range:', i)
            assert 0

    def magnitude(self):
        return sqrt(self.x**2 + self.y**2 + self.z**2)

    def unitVector(self):
        magn = self.magnitude()
        if magn < 1e-20:
            return xyz(0., 0., 1.)
        denom = 1.0 / magn
        return xyz(self.x * denom, self.y * denom, self.z * denom)

    def dot(self, P):
        return self.x * P.x + self.y * P.y + self.z * P.z

    def cross(self, P):
        return xyz(self.y * P.z - self.z * P.y,
                   self.z * P.x - self.x * P.z,
                   self.x * P.y - self.y * P.x)

    def translate(self, offset):
        self.x += offset[0]
        self.y += offset[1]
        self.z += offset[2]

    def _rotate2d(self, phiRad, XX, YY, XX0=0., YY0=0.):
        """
        Helper method for rotation in 2D plane

        Args:
            phiRad (float):
                angle in [rad]

            XX, YY (float):
                2D coordinates

            XX0, YY0 (float, optional):
                2D coordinates of center of rotation

        Returns:
            (float, float):
                transformed 2D coordinates
        """
        xx = XX0 + (XX - XX0) * cos(phiRad) - (YY - YY0) * sin(phiRad)
        yy = YY0 + (XX - XX0) * sin(phiRad) + (YY - YY0) * cos(phiRad)
        return xx, yy

    def rotate(self, phiRad, rotAxis):
        """
        Coordinate transformation: rotate this point in Cartesian system

        Args:
            phiRad (array of float):
                angle of counter-clockwise rotation [rad]

            rotAxis (array of float):
                coordinates of rotation axis, one and only one component
                is None; this component indicates the rotation axis.
                e.g. 'rotAxis.y is None' forces rotation around y-axis,
                rotation center is (P0.x, P0.z)
        """
        if rotAxis[0] is None:
            self.y, self.z = self._rotate2d(phiRad, self.y, self.z, rotAxis[1],
                                            rotAxis[2])
        elif rotAxis[1] is None:
            self.z, self.x = self._rotate2d(phiRad, self.z, self.x, rotAxis[2],
                                            rotAxis[0])
        elif rotAxis[2] is None:
            self.x, self.y = self._rotate2d(phiRad, self.x, self.y, rotAxis[0],
                                            rotAxis[1])
        else:
            print('??? invalid definition of rotation axis', rotAxis)
            assert 0

    def rotateDeg(self, phiDeg, rotAxis):
        """
        Coordinate transformation: rotate this point in Cartesian system

        Args:
            phiDeg (array of float):
                angle of counter-clockwise rotation [degrees]

            rotAxis (array of float):
                coordinates of rotation axis, one and only one component
                is None; this component indicates the rotation axis.
                e.g. 'rotAxis.y is None' forces rotation around y-axis,
                rotation center is (P0.x, P0.z)
        """
        self.rotate(phiDeg / 180. * pi, rotAxis)

    def scale(self, scalingFactor):
        if isinstance(scalingFactor, (int, float)):
            if self.x is not None:
                self.x *= float(scalingFactor)
            if self.y is not None:
                self.y *= float(scalingFactor)
            if self.z is not None:
                self.z *= float(scalingFactor)
        elif isinstance(scalingFactor, xyz):
            if self.x is not None:
                self.x *= float(scalingFactor.x)
            if self.y is not None:
                self.y *= float(scalingFactor.y)
            if self.z is not None:
                self.z *= float(scalingFactor.z)
        elif isinstance(scalingFactor, list) and len(scalingFactor) == 3:
            if self.x is not None:
                self.x *= float(scalingFactor[0])
            if self.y is not None:
                self.y *= float(scalingFactor[1])
            if self.z is not None:
                self.z *= float(scalingFactor[2])
        elif isinstance(scalingFactor, list) and len(scalingFactor) == 1:
            if self.x is not None:
                self.x *= float(scalingFactor[0])
            if self.y is not None:
                self.y *= float(scalingFactor[0])
            if self.z is not None:
                self.z *= float(scalingFactor[0])
        else:
            print('??? invalid definition of scaling')
            print('??? scaling:', scalingFactor, ' type(scalingFactor):',
                  type(scalingFactor))
            assert 0

    def __str__(self):
        return '(' + str(self.x) + ' ' + str(self.y) + ' ' + str(self.z) + ')'
#    def __str__(self):
#        s = str(OrderedDict(sorted(self.__dict__.items(),
#                            key=lambda t: t[0])))[len('OrderedDict('):]
#        return s.replace("('", "'").replace("',", "':").replace(")", "")


class xyzt(xyz):
    """
    Point in 3D space with time: (x, y, z, t)
    """

    def __init__(self, x=0., y=0., z=0., t=0., point=None):
        """
        Args:
            x, y, z (float, optional):
                coordinates of point in 3D space [m, m, m]

            t (float, optional):
                time [s]

            point (xyz or xyzt, optional):
                'point' will be assigned to this object if 'point' is not None
        """
        if isinstance(point, xyzt):
            self.x, self.y, self.z, self.t = point.x, point.y, point.z, point.t
        elif isinstance(point, xyz):
            self.x, self.y, self.z, self.t = point.x, point.y, point.z, 0.
        elif point is None:
            self.x = x if x is not None else 0.
            self.y = y if y is not None else 0.
            self.z = z if z is not None else 0.
            self.t = t if t is not None else 0.
        else:
            print("??? xyzt: point is not of type 'xyz' or 'xyzt', type=",
                  type(point))
            self.x, self.y, self.z, self.t = None, None, None, None

    def at(self, i):
        """
        Accesses point components by index

        Args:
            i (int):
                index of component (0:x, 1:y, 2:z, 3:t)
        Returns:
            (float):
                value of component
        """
        if i == 0:
            return self.x
        elif i == 1:
            return self.y
        elif i == 2:
            return self.z
        elif i == 3:
            return self.t
        else:
            print('??? i in at() is out of range, i:', i)
            assert 0

    def __str__(self):
        return super().__str__()[:-1] + '  ' + str(self.t) + ')'


# Examples ####################################################################

if __name__ == '__main__':
    ALL = 1

    if 0 or ALL:
        P0 = xyz(2.2, -1)
        print('P0:', P0)
        P1 = xyz(x=1, z=4)
        print('P1:', P1)
        P2 = xyz(point=P1)
        print('P2:', P2)
        P3 = xyz(point=[])                        # invalid
        print('P3:', P3)

        print('P0.at(1)=P0.y:', P0.at(1))
        print('P0, P1:', P0, P1)
        print('P0 + 1:', P0 + 1)
        print('P0 + P1:', P0 + P1)
        print('P0 - 1:', P0 - 1)
        print('P0 - P1:', P0 - P1)
        print('P0 * 2:', P0 * 2)
        print('P0 * (1/2.):', P0 * (1/2.))
        print('P0 * P1:', P0 * P1)

    if 0 or ALL:
        P4 = xyzt(2.2, -1, t=7)
        P5 = xyzt(point=P1)
        P6 = xyzt(point=P4)
        P7 = xyzt(point={'a': 1, 'b': 2})         # invalid
        print('P4:', P4)
        print('P5:', P5)
        print('P6:', P6)
        print('P5==P1:', P5 == P1)
        print('P5!=P1:', P5 != P1)
