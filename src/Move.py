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
      2018-05-28 DWW
"""

from math import isclose
import numpy as np
import matplotlib.pyplot as plt

from xyz import xyz, xyzt
from Loop import Loop


class Move(Loop):
    """
    Moves an object along a trajectory and provides its velocity, position and
    rotation in 3D space as a function of time.

    The object travels along a given trajectory defined by:
        a) polygons defined by way-points (x,y,z) and constant speed or
        b) polygons defined by way-points (x,y,z,t) for every time step

    Velocity, position at a given time are defined by an array of waypoints.
    The initial position of the object corresponds to the position of the first
    way-point. The initial velocity is zero.

    When the last way-point is reached, the position of the object becomes
    static and its velocity is zero. When a object is positioned between two
    waypoints, it moves with constant velocity between them. To keep an object
    at a certain position, two waypoints with the same position but different
    times should be defined.
    """

    def __init__(self, identifier='Move'):
        super().__init__(identifier=identifier)

        self._wayPoints = None          # array of wayPoints [m, m, m, s]
        self._rotations = None          # rotations in 3D space [rad]
        self._position = xyz()          # object position in 3D space [m]
        self._velocity = xyz()          # velocity in 3D space [m/s]
        self._rotation = xyz()          # rotation in 3D space [rad]
        self._iLastWayPoint = 0         # index of last passed way point
        self._trajectoryHistory = None  # plot data

    def setTrajectory(self, way, rot=None, speed=None, tBegin=0, tEnd=None):
        """
        Defines list of waypoints

        Args:
            way (array_like of xyz or xyzt):
                way points in 3D space [m, m, m] or [m, m, m, s]

            rot (array_like of xyz):
                rotation of waypoints in 3D space [rad], size can be 0, 1, or
                length of wayPoints

            speed (float, optional):
                constant magnitude of object velocity [m/s]

            tBegin (float, optional):
                start time [s]

            tEnd (float, optional):
                end time [s]
        """
        self._wayPoints = list(way)
        if len(self._wayPoints) <= 1:
            return
        self._wayPoints = [xyzt(point=P) for P in self._wayPoints]

        way = [0.]
        for i in range(1, len(self._wayPoints)):
            Dl = (self._wayPoints[i] - self._wayPoints[i-1]).magnitude()
            way.append(way[i-1] + Dl)

        if rot is None or len(rot) == 0:
            rot = [xyz()]
        assert len(rot) == 1 or len(rot) == len(self._wayPoints)
        if len(rot) == 1:
            self._rotations = rot * len(self._wayPoints)
        else:
            self._rotations = [xyz(point=r) for r in rot]

        if speed is None:
            if tEnd is None or isclose(tEnd, 0.):
                for P in self._wayPoints:
                    P.t = 0.
                return
            speed = way[-1] / tEnd

        self._wayPoints[0].t = tBegin
        for i in range(1, len(self._wayPoints)):
            dt = (way[i] - way[i-1]) / speed
            self._wayPoints[i].t = self._wayPoints[i-1].t + dt

        self._position = self._wayPoints[0]
        self._velocity = self.velocity(0)
        self._rotation = self.rotation(0)

        del way

    def iWayPointAhead(self, t=None, iStart=0):
        """
        Finds index of way-point AHEAD of current object position

        Args:
            t (float, optional):
                time [s]. If None, 't' is set to actual time 'self.t'

            iStart (int, optional):
                index of way-point where search should start

        Returns:
            (int):
                Waypoint index ahead of current position. Index is greater 0
        """
        if t is None:
            t = self.t
        if t <= 0.:
            return 0 + 1
        n = len(self._wayPoints)
        if n <= 2:
            return 1
        if iStart is None:
            iStart = 0
        if iStart >= n:
            iStart = 0
        if t <= self._wayPoints[iStart].t:
            return iStart
        if t >= self._wayPoints[-1].t:
            return n - 1

        iPrev = iStart
        iNext = n-1
        while True:
            iCenter = (iPrev + iNext) // 2
            if t < self._wayPoints[iCenter].t:
                iNext = iCenter
            else:
                iPrev = iCenter
            if iNext - iPrev <= 1:
                break
        return iNext

    def initialCondition(self):
        """
        Initializes object positionm velocity and rotation
        """
        super().initialCondition()

        if self._wayPoints is not None:
            self._position = self._wayPoints[0]
            self._velocity = self.velocity(0)
            self._rotation = self.rotation(0)
        else:
            self._position = xyz()
            self._velocity = xyz()
            self._rotation = xyz()

        if self.silent:
            self._trajectoryHistory = None
        else:
            self._trajectoryHistory = \
                [[self._position.x], [self._position.y], [self._position.z],
                 [self._velocity.x], [self._velocity.y], [self._velocity.z],
                 [self._rotation.x], [self._rotation.y], [self._rotation.z]]

    def updateTransient(self):
        """
        Updates object position, velocity and rotation

        Note:
            Actual time is available as self.t
        """
        super().updateTransient()

        self._position = self.position(self.t)
        self._velocity = self.velocity(self.t)
        self._rotation = self.rotation(self.t)

        if self._trajectoryHistory:
            self._trajectoryHistory[0].append(self._position.x)
            self._trajectoryHistory[1].append(self._position.y)
            self._trajectoryHistory[2].append(self._position.z)

            self._trajectoryHistory[3].append(self._velocity.x)
            self._trajectoryHistory[4].append(self._velocity.y)
            self._trajectoryHistory[5].append(self._velocity.z)

            self._trajectoryHistory[6].append(self._rotation.x)
            self._trajectoryHistory[7].append(self._rotation.y)
            self._trajectoryHistory[8].append(self._rotation.z)

    def position(self, t=None):
        """
        Args:
            t (float, optional):
                time [s]

        Returns:
            Value of self._position if t is None and if self._position is not
            None. Otherwise the actual position is calculated as function of
            time [m, m, m]

        Note:
            The calculated position is NOT stored as 'self._position'
        """
        if t is None and self._position is not None:
            return self._position
        if self._wayPoints is None:
            return self._position

        if t is None or t < 0.:
            t = 0.
        if t >= self._wayPoints[-1].t:
            P = self._wayPoints[-1]
            return xyz(P.x, P.y, P.z)

        # P = P' + (P"-P') * (t-t') / (t"-t')
        iAhead = self.iWayPointAhead(t)
        DP = self._wayPoints[iAhead] - self._wayPoints[iAhead-1]
        dt = t - self._wayPoints[iAhead-1].t
        Dt = self._wayPoints[iAhead].t - self._wayPoints[iAhead-1].t
        P = self._wayPoints[iAhead-1] + DP * (dt / Dt)
        return xyz(P.x, P.y, P.z)

    def rotation(self, t=None):
        """
        Args:
            t (float, optional):
                time [s]

        Returns:
            Value of self._rotations if t is None and if self._rotations is not
            None. Otherwise the actual rotation is calculated as function of
            time [rad]

        Note:
            The calculated rotation is NOT stored as 'self._rotation'
        """
        if t is None and self._rotations is not None:
            return self._rotations
        if self._wayPoints is None:
            return xyz()

        if t is None or t < 0.:
            t = 0.
        if t >= self._wayPoints[-1].t:
            return self._rotations[-1]

        # R = R' + (R"-R') * (t-t') / (t"-t')
        iAhead = self.iWayPointAhead(t)
        DR = self._rotations[iAhead] - self._rotations[iAhead-1]
        dt = t - self._wayPoints[iAhead-1].t
        Dt = self._wayPoints[iAhead].t - self._wayPoints[iAhead-1].t
        return self._rotations[iAhead-1] + DR * (dt / Dt)

    def way(self, t=None):
        """
        Args:
            t (float, optional):
                time [s]

        Returns:
            (float):
                Way from start position to stop position defined by the given
                time [m]. If t is None, the length of the full trajectory is
                returned
        """
        if t is None:
            # way from start to stop point
            w = 0.
            for i in range(1, len(self._wayPoints)):
                w += (self._wayPoints[i] - self._wayPoints[i-1]).magnitude()
            return w

        # 'dP' is vector from point behind current position to current position
        iAhead = self.iWayPointAhead(t)
        DP = self._wayPoints[iAhead] - self._wayPoints[iAhead-1]
        dt = t - self._wayPoints[iAhead-1].t
        Dt = self._wayPoints[iAhead].t - self._wayPoints[iAhead-1].t
        dP = DP * (dt / Dt)

        # way from start position to point behind current position
        w = 0.
        for i in range(1, iAhead):
            w += (self._wayPoints[i] - self._wayPoints[i-1]).magnitude()

        # way from waypoint behind current position to current position
        w += dP.magnitude()

        # way from start position to current position
        return w

    def velocity(self, t=None):
        """
        Args:
            t (float, optional):
                time [s]

        Returns:
            (xyz):
                Value of self._velocity if 't' is None and self._velocity is
                not None. Otherwise the actual velocity is calculated as
                function of time [m/s]

        Note:
            The calculated velocity is NOT stored as 'self._velocity'
        """
        if t is None:
            if self._velocity is not None:
                return self._velocity
            t = 0.
        if self._wayPoints is None:
            return xyz()
        iAhead = self.iWayPointAhead(t)
        DP = self._wayPoints[iAhead] - self._wayPoints[iAhead-1]
        Dt = self._wayPoints[iAhead].t - self._wayPoints[iAhead-1].t

        return DP * (1 / Dt)

    def __str__(self):
        return str([str(P) for P in self._wayPoints]) + ' ' + \
            str(self._rotations)

    def plot(self, title=None):
        if title is None:
            title = 'trajectory (x, y)'

        assert len(self._rotations) == len(self._wayPoints)

        X = [P.x for P in self._wayPoints]
        Y = [P.y for P in self._wayPoints]
        Z = [P.z for P in self._wayPoints]

        T = [P.t for P in self._wayPoints]
        W = [self.way(P.t) for P in self._wayPoints]
        S = [self.velocity(P.t).magnitude() for P in self._wayPoints]
        VX = [self.velocity(P.t).x for P in self._wayPoints]
        VY = [self.velocity(P.t).y for P in self._wayPoints]
        VZ = [self.velocity(P.t).z for P in self._wayPoints]

        plt.title('trajectory(x)')
        plt.plot(X, S, '--', label='speed')
        plt.plot(X, T, '--', label='time')
        plt.scatter(self._trajectoryHistory[0], self._trajectoryHistory[1],
                    label='update()')
        plt.plot(X, Y, '-', label='trajectory')
        plt.plot(X, W, '-', label='way')
        plt.xlabel('x')
        plt.ylabel('v, t, trj way(x,y)')
        plt.grid()
        plt.legend()
        plt.show()
        plt.title('trajectory(y)')
        plt.plot(Y, S, '--', label='speed')
        plt.plot(Y, T, '--', label='time')
        plt.scatter(self._trajectoryHistory[2], self._trajectoryHistory[1],
                    label='update()')
        plt.plot(Y, Z, '-', label='trajectory(y,z)')
        plt.plot(Y, W, '-', label='way')
        plt.xlabel('z')
        plt.ylabel('v, t, trj way(y,z)')
        plt.grid()
        plt.legend()
        plt.show()

        plt.title('velocity from wayPoints 1(3)')
        plt.quiver(X, Y, VX, VY, angles='xy', scale_units='xy')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid()
        plt.show()
        plt.title('velocity from wayPoints 2(3)')
        plt.quiver(X, Z, VX, VZ, angles='xy', scale_units='xy')
        plt.xlabel('x')
        plt.ylabel('z')
        plt.grid()
        plt.show()
        plt.title('velocity from wayPoints 3(3)')
        plt.quiver(X, Z, VY, VZ, angles='xy', scale_units='xy')
        plt.xlabel('y')
        plt.ylabel('z')
        plt.grid()
        plt.show()

        plt.title('rotation vs waypoint index')
        tWay = [w.t for w in self._wayPoints]
        plt.plot(tWay, [rot.x for rot in self._rotations], 'o', label='rot x')
        plt.plot(tWay, [rot.y for rot in self._rotations], 'o', label='rot y')
        plt.plot(tWay, [rot.z for rot in self._rotations], 'o', label='rot z')

        tMany = np.linspace(0, self._wayPoints[-1].t, 100)
        plt.plot(tMany, [self.rotation(t).x for t in tMany], label='rot.x(t)')
        plt.plot(tMany, [self.rotation(t).y for t in tMany], label='rot.y(t)')
        plt.plot(tMany, [self.rotation(t).z for t in tMany], label='rot.z(t)')
        plt.legend()
        plt.grid()
        plt.show()

        if self._trajectoryHistory:
            plt.title('velocity from update() 1(3)')
            plt.quiver(self._trajectoryHistory[0], self._trajectoryHistory[1],
                       self._trajectoryHistory[3], self._trajectoryHistory[4],
                       angles='xy', scale_units='xy')
            plt.xlim(0, )
            plt.xlabel('x')
            plt.ylabel('y')
            plt.grid()
            plt.show()
            plt.title('velocity from update() 2(3)')
            plt.quiver(self._trajectoryHistory[0], self._trajectoryHistory[2],
                       self._trajectoryHistory[3], self._trajectoryHistory[5],
                       angles='xy', scale_units='xy')
            plt.xlim(0, )
            plt.xlabel('x')
            plt.ylabel('z')
            plt.grid()
            plt.show()
            plt.title('velocity from update() 3(3)')
            plt.quiver(self._trajectoryHistory[1], self._trajectoryHistory[2],
                       self._trajectoryHistory[4], self._trajectoryHistory[5],
                       angles='xy', scale_units='xy')
            plt.xlim(0, )
            plt.xlabel('y')
            plt.ylabel('z')
            plt.grid()
            plt.show()

            plt.title('rotation from update() 1(3)')
            plt.quiver(self._trajectoryHistory[0], self._trajectoryHistory[1],
                       self._trajectoryHistory[6], self._trajectoryHistory[7],
                       angles='xy', scale_units='xy')
            plt.xlim(0, )
            plt.xlabel('x')
            plt.ylabel('y')
            plt.grid()
            plt.show()
            plt.title('rotation from update() 2(3)')
            plt.quiver(self._trajectoryHistory[0], self._trajectoryHistory[1],
                       self._trajectoryHistory[6], self._trajectoryHistory[8],
                       angles='xy', scale_units='xy')
            plt.xlim(0, )
            plt.xlabel('x')
            plt.ylabel('z')
            plt.grid()
            plt.show()
            plt.title('rotation from update() 3(3)')
            plt.quiver(self._trajectoryHistory[0], self._trajectoryHistory[1],
                       self._trajectoryHistory[7], self._trajectoryHistory[8],
                       angles='xy', scale_units='xy')
            plt.xlim(0, )
            plt.xlabel('y')
            plt.ylabel('z')
            plt.grid()
            plt.show()


# Examples ####################################################################

if __name__ == '__main__':
    ALL = 1
                                 #   y
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

    if 0 or ALL:
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

    if 0 or ALL:
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
