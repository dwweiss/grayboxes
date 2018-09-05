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
      2018-09-05 DWW
"""

__all__ = ['xyz', 'xyzt']

import numpy as np
from typing import Optional, Sequence, Tuple, Union


def _rotate2d(phi_rad: Union[float, Sequence[float]],
              XX: Union[float, Sequence[float], np.ndarray],
              YY: Union[float, Sequence[float], np.ndarray],
              XX0: float = 0., YY0: float = 0.) \
        -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
    """
    Helper method for rotation of (XX,YY) point in 2D plane

    Args:
        phi_rad:
            angle in [rad]

        XX:
            x-coordinates [m]

        YY:
            y-coordinates [m]

        XX0:
            x-coordinate of center of rotation [m]

        YY0:
            y-coordinate of center of rotation [m]

    Returns:
        Transformed 2D coordinates [m]
    """
    xx = XX0 + (XX-XX0) * np.cos(phi_rad) - (YY-YY0) * np.sin(phi_rad)
    yy = YY0 + (XX-XX0) * np.sin(phi_rad) + (YY-YY0) * np.cos(phi_rad)
    return xx, yy


class xyz(object):
    """
    Point in 3D space (x, y, z)
    """

    def __init__(self, x: float=0., y: float=0., z: float=0.,
                 point: Optional['xyz']=None) -> None:
        """
        Args:
            x:
                x-coordinate of point in 3D space [m]
            y:
                y-coordinate of point in 3D space [m]
            z:
                z-coordinate of point in 3D space [m]

            point:
                Point will be assigned to this object if it is not None [m]
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

    def __add__(self, P: Union[float, 'xyz']) -> 'xyz':
        if isinstance(P, xyz):
            return xyz(self.x + P.x, self.y + P.y, self.z + P.z)
        else:
            return xyz(self.x + P, self.y + P, self.z + P)

    def __sub__(self, P: Union[float, 'xyz']) -> 'xyz':
        if isinstance(P, xyz):
            return xyz(self.x - P.x, self.y - P.y, self.z - P.z)
        else:
            return xyz(self.x - P, self.y - P, self.z - P)

    def __mul__(self, multiplier: Union[float, 'xyz']) -> 'xyz':
        if isinstance(multiplier, xyz):
            return xyz(self.x * multiplier.x, self.y * multiplier.y,
                       self.z * multiplier.z)
        else:
            return xyz(self.x * multiplier, self.y * multiplier,
                       self.z * multiplier)

    def __eq__(self, other: 'xyz') -> bool:
        return np.allclose([self.x, self.y, self.z],
                           [other.x, other.y, other.z])

    def at(self, i: int) -> float:
        """
        Accesses point components by index

        Args:
            i:
                Index of component (0: x, 1: y, 2: z)

        Returns:
            Value of component [m]
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

    def magnitude(self) -> float:
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def unitVector(self) -> 'xyz':
        magn = self.magnitude()
        if magn < 1e-20:
            return xyz(0., 0., 1.)
        denom = 1.0 / magn
        return xyz(self.x * denom, self.y * denom, self.z * denom)

    def dot(self, P: 'xyz') -> float:
        return self.x * P.x + self.y * P.y + self.z * P.z

    def cross(self, P: 'xyz') -> 'xyz':
        return xyz(self.y * P.z - self.z * P.y,
                   self.z * P.x - self.x * P.z,
                   self.x * P.y - self.y * P.x)

    def translate(self, offset: Union[Sequence[float], 'xyz']) -> None:
        if isinstance(offset, xyz):
            self.x += offset.x
            self.y += offset.y
            self.z += offset.z
        else:
            self.x += offset[0]
            self.y += offset[1]
            self.z += offset[2]

    def rotate(self, phiRad: Sequence[float],
               rotAxis: Sequence[float]) -> None:
        """
        Coordinate transformation: rotate this point in Cartesian system

        Args:
            phiRad:
                Angle(s) of counter-clockwise rotation [rad]

            rotAxis:
                Coordinate(s) of rotation axis, one and only one component
                is None; this component indicates the rotation axis.
                e.g. 'rotAxis.y is None' forces rotation around y-axis,
                rotation center is (P0.x, P0.z) [m]
        """
        if rotAxis[0] is None:
            self.y, self.z = _rotate2d(phiRad, self.y, self.z, rotAxis[1],
                                       rotAxis[2])
        elif rotAxis[1] is None:
            self.z, self.x = _rotate2d(phiRad, self.z, self.x, rotAxis[2],
                                       rotAxis[0])
        elif rotAxis[2] is None:
            self.x, self.y = _rotate2d(phiRad, self.x, self.y, rotAxis[0],
                                       rotAxis[1])
        else:
            print('??? invalid definition of rotation axis', rotAxis)
            assert 0

    def rotateDeg(self, phiDeg: Sequence[float],
                  rotAxis: Sequence[float]) -> None:
        """
        Coordinate transformation: rotate this point in Cartesian system

        Args:
            phiDeg:
                Angle(s) of counter-clockwise rotation [degrees]

            rotAxis:
                Coordinate(s) of rotation axis, one and only one component
                is None; this component indicates the rotation axis.
                e.g. 'rotAxis.y is None' forces rotation around y-axis,
                rotation center is (P0.x, P0.z)
        """
        self.rotate(np.asfarray(phiDeg) / 180. * np.pi, rotAxis)

    def scale(self, scalingFactor: 
              Union[int, float, Sequence[float], 'xyz']) -> None:
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
        elif isinstance(scalingFactor, (list, tuple)) and \
                len(scalingFactor) == 3:
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

    def __str__(self) -> str:
        return '(' + str(self.x) + ' ' + str(self.y) + ' ' + str(self.z) + ')'


class xyzt(xyz):
    """
    Point in 3D space with time: (x, y, z, t)
    """

    def __init__(self, x: float=0., y: float=0., z: float=0., t: float=0.,
                 point: Optional[Union[xyz, 'xyzt']]=None) -> None:
        """
        Args:
            x:
                x-coordinate of point in 3D space [m]

            y:
                y-coordinate of point in 3D space [m]

            z:
                z-coordinate of point in 3D space [m]

            t:
                time [s]

            point:
                Point will be assigned to this object if it is not None
        """
        super().__init__(x, y, z)

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

    def at(self, i: int) -> int:
        """
        Accesses point components by index

        Args:
            i:
                Index of component (0:x, 1:y, 2:z, 3:t)

        Returns:
            Value of component
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

    def __str__(self) -> str:
        return super().__str__()[:-1] + '  ' + str(self.t) + ')'
