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
      2020-02-06 DWW
"""

from math import isclose
import numpy as np
from typing import Iterable, List, Optional, Union
import matplotlib.pyplot as plt

from grayboxes.loop import Loop
from grayboxes.xyz import xyz, xyzt


class Move(Loop):
    """
    Moves an object along a trajectory and provides its velocity,
    position and rotation in 3D space as a function of time.

    The object travels along a given trajectory defined by:
        a) polygons defined by way-points (x,y,z) and constant speed or
        b) polygons defined by way-points (x,y,z,t) for every time step

    Velocity and position at a given time are defined by an array of
    waypoints. The initial position of the object corresponds to the
    position of the first way-point. The initial velocity is zero.

    When the last way-point is reached, the position of the object
    becomes static and its velocity is zero. When a object is positioned
    between two waypoints, it moves with constant velocity between them.
    To keep an object at a certain position, two waypoints with the same
    position but different times should be defined.
    """

    def __init__(self, identifier: str = 'Move') -> None:
        super().__init__(identifier=identifier)

        self._waypoints: Optional[Union[List[xyz], List[xyzt]]] = None                     
                                            # array of wayPoints [m,m,m,s]
        self._orientations: Optional[Union[xyz, List[xyz]]] = None     
                                          # orientations in 3D space [rad]
        self._actual_position: xyz = xyz()     # actual position in 3D [m]
        self._velocity: xyz = xyz()           # velocity in 3D space [m/s]
        self._actual_orientation: xyz = xyz()   # actual orientation [rad]
        self._i_last_waypoint: int = 0    # index of last passed way point
        self._trajectory_history: Optional[List[List[float]]] = None

    def set_trajectory(self, waypoints: Iterable[Union[xyz, xyzt]],
                       orientations: Optional[Iterable[xyz]] = None,
                       speed: Optional[float] = None,
                       t_begin: float = 0.,
                       t_end: Optional[float] = None) -> None:
        """
        Defines list of way points

        Args:
            waypoints:
                way points in 3D space [m, m, m] or [m, m, m, s]

            rot:
                rotation of way points in 3D space [rad], size can be 0,
                1, or length of wayPoints

            speed:
                constant magnitude of object velocity [m/s]

            t_begin:
                start time [s]

            t_end:
                end time [s]
        """
        self._waypoints = list(waypoints)
        if len(self._waypoints) <= 1:
            return
        self._waypoints = [xyzt(point=P) for P in self._waypoints]

        way: List[float] = [0.]
        for i in range(1, len(self._waypoints)):
            Delta_l = (self._waypoints[i] - self._waypoints[i - 1]).magnitude()
            way.append(way[i-1] + Delta_l)

        if orientations is None or len(orientations) == 0:
            orientations = [xyz()]
        assert len(orientations) == 1 or \
            len(orientations) == len(self._waypoints)
        if len(orientations) == 1:
            self._orientations = orientations * len(self._waypoints)
        else:
            self._orientations = [xyz(point=p) for p in orientations]

        if speed is None:
            if t_end is None or isclose(t_end, 0.):
                for P in self._waypoints:
                    P.t = 0.
                return
            speed = way[-1] / t_end

        self._waypoints[0].t = t_begin
        for i in range(1, len(self._waypoints)):
            dt = (way[i] - way[i-1]) / speed
            self._waypoints[i].t = self._waypoints[i - 1].t + dt

        self._actual_position = self._waypoints[0]
        self._velocity = self.velocity(0.)
        self._actual_orientation = self.orientation(0.)

        del way

    def i_waypoint_ahead(self, t: Optional[float] = None, 
                               i_start: int = 0) -> int:
        """
        Finds index of way-point AHEAD of current object position

        Args:
            t:
                time [s]. If None, 't' is set to actual time 'self.t'

            i_start:
                index of way-point where search should start

        Returns:
            Waypoint index ahead of current object position.
            Index is greater 0
        """
        assert self._waypoints and isinstance(self._waypoints[0], xyzt)
        
        if t is None:
            t = self.t
        if t <= 0.:
            return 0 + 1
        n = len(self._waypoints)
        if n <= 2:
            return 1
        if i_start is None:
            i_start = 0
        if i_start >= n:
            i_start = 0
        if t <= self._waypoints[i_start].t:
            return i_start
        if t >= self._waypoints[-1].t:
            return n - 1

        i_prev = i_start
        i_next = n-1
        while True:
            i_center = (i_prev + i_next) // 2
            if t < self._waypoints[i_center].t:
                i_next = i_center
            else:
                i_prev = i_center
            if i_next - i_prev <= 1:
                break
        return i_next

    def initial_condition(self) -> bool:
        """
        Initializes object positionm velocity and rotation
        """
        ok = super().initial_condition()

        if self._waypoints is not None:
            self._actual_position = self._waypoints[0]
            self._velocity = self.velocity(0)
            self._actual_orientation = self.orientation(0)
        else:
            self._actual_position = xyz()
            self._velocity = xyz()
            self._actual_orientation = xyz()

        if self.silent:
            self._trajectory_history = None
        else:
            # First row (at t=0) of trajectory history  
            self._trajectory_history = [
                [self._actual_position.x], 
                [self._actual_position.y], 
                [self._actual_position.z],
                 
                [self._velocity.x], 
                [self._velocity.y], 
                [self._velocity.z],
                 
                [self._actual_orientation.x], 
                [self._actual_orientation.y], 
                [self._actual_orientation.z]
            ]
                
        return ok

    def update_transient(self) -> bool:
        """
        Updates object position, velocity and rotation

        Note:
            Actual time is available as self.t
        """
        ok = super().update_transient()

        self._actual_position = self.position(self.t)
        self._velocity = self.velocity(self.t)
        self._actual_orientation = self.orientation(self.t)

        if self._trajectory_history:
            self._trajectory_history[0].append(self._actual_position.x)
            self._trajectory_history[1].append(self._actual_position.y)
            self._trajectory_history[2].append(self._actual_position.z)

            self._trajectory_history[3].append(self._velocity.x)
            self._trajectory_history[4].append(self._velocity.y)
            self._trajectory_history[5].append(self._velocity.z)

            self._trajectory_history[6].append(self._actual_orientation.x)
            self._trajectory_history[7].append(self._actual_orientation.y)
            self._trajectory_history[8].append(self._actual_orientation.z)
            
        return ok

    def position(self, t: Optional[float] = None) -> xyz:
        """
        Args:
            t:
                time [s]

        Returns:
            Value of self._actual_position if t is None and if 
            self._actual_position is not None. Otherwise the actual 
            position is calculated as function of time [m, m, m]

        Note:
            The calculated position is NOT stored as 'self._actual_position'
        """
        if t is None and self._actual_position is not None:
            return self._actual_position
        if self._waypoints is None:
            return self._actual_position

        if t is None or t < 0.:
            t = 0.
        if t >= self._waypoints[-1].t:
            P = self._waypoints[-1]
            return xyz(P.x, P.y, P.z)

        # P = P' + (P"-P') * (t-t') / (t"-t')
        i_ahead = self.i_waypoint_ahead(t)
        Delta_P = self._waypoints[i_ahead] - self._waypoints[i_ahead - 1]
        delta_t = t - self._waypoints[i_ahead - 1].t
        Delta_t = self._waypoints[i_ahead].t - self._waypoints[i_ahead - 1].t
        P = self._waypoints[i_ahead - 1] + Delta_P * (delta_t / Delta_t)
        return xyz(P.x, P.y, P.z)

    def orientation(self, t: Optional[float] = None) -> xyz:
        """
        Args:
            t:
                time [s]

        Returns:
            Value of self._orientations if t is None and if 
            self._orientations is not None. Otherwise the actual 
            rotation is calculated as function of time [rad]

        Note:
            The calculated rotation is NOT stored as 'self._actual_orientation'
        """
        if t is None and self._orientations is not None:
            return self._orientations
        if self._waypoints is None:
            return xyz()

        if t is None or t < 0.:
            t = 0.
        if t >= self._waypoints[-1].t:
            return self._orientations[-1]

        # R = R' + (R"-R') * (t-t') / (t"-t')
        i_ahead = self.i_waypoint_ahead(t)
        DR = self._orientations[i_ahead] - \
            self._orientations[i_ahead-1]
        dt = t - self._waypoints[i_ahead - 1].t
        Dt = self._waypoints[i_ahead].t - self._waypoints[i_ahead - 1].t
        return self._orientations[i_ahead-1] + DR * (dt / Dt)

    def way(self, t: Optional[float] = None) -> float:
        """
        Args:
            t:
                time [s]

        Returns:
            Way from start position to stop position defined by the
            given time [m]. If t is None, the length of the full
            trajectory is returned
        """
        if t is None:
            # way from start to stop point
            w = 0.
            for i in range(1, len(self._waypoints)):
                w += (self._waypoints[i] - self._waypoints[i - 1]).magnitude()
            return w

        # 'dP' is vector from point behind current pos. to current pos.
        i_ahead = self.i_waypoint_ahead(t)
        Delta_P = self._waypoints[i_ahead] - self._waypoints[i_ahead - 1]
        delta_t = t - self._waypoints[i_ahead - 1].t
        Delta_t = self._waypoints[i_ahead].t - self._waypoints[i_ahead - 1].t
        delta_P = Delta_P * (delta_t / Delta_t)

        # way from start position to point behind current position
        w = 0.
        for i in range(1, i_ahead):
            w += (self._waypoints[i] - self._waypoints[i - 1]).magnitude()

        # way from waypoint behind current position to current position
        w += delta_P.magnitude()

        # way from start position to current position
        return w

    def velocity(self, t: Optional[float] = None) -> xyz:
        """
        Args:
            t:
                time [s]

        Returns:
            Value of self._velocity if t is None and self._velocity
            is not None. Otherwise the actual velocity is calculated
            as function of time [m/s]

        Note:
            The calculated velocity is NOT stored as 'self._velocity'
        """
        if t is None:
            if self._velocity is not None:
                return self._velocity
            t = 0.
        if self._waypoints is None:
            return xyz()
        i_ahead = self.i_waypoint_ahead(t)
        Delta_P = self._waypoints[i_ahead] - self._waypoints[i_ahead - 1]
        Delta_t = self._waypoints[i_ahead].t - self._waypoints[i_ahead - 1].t

        return Delta_P * (1 / Delta_t)

    def __str__(self) -> str:
        return str([str(P) for P in self._waypoints]) + ' ' + \
               str(self._orientations)

    def plot(self) -> None:
        assert len(self._orientations) == len(self._waypoints)

        X = [P.x for P in self._waypoints]
        Y = [P.y for P in self._waypoints]
        Z = [P.z for P in self._waypoints]

        T = [P.t for P in self._waypoints]
        W = [self.way(P.t) for P in self._waypoints]
        S = [self.velocity(P.t).magnitude() for P in self._waypoints]
        VX = [self.velocity(P.t).x for P in self._waypoints]
        VY = [self.velocity(P.t).y for P in self._waypoints]
        VZ = [self.velocity(P.t).z for P in self._waypoints]

        plt.title('trajectory(x)')
        plt.plot(X, S, '--', label='speed')
        plt.plot(X, T, '--', label='time')
        plt.scatter(self._trajectory_history[0], self._trajectory_history[1],
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
        plt.scatter(self._trajectory_history[2], self._trajectory_history[1],
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

        plt.title('orientation vs waypoint index')
        t_way = [w.t for w in self._waypoints]
        plt.plot(t_way, [o.x for o in self._orientations], 'o', label='ori x')
        plt.plot(t_way, [o.y for o in self._orientations], 'o', label='ori y')
        plt.plot(t_way, [o.z for o in self._orientations], 'o', label='ori z')

        t_many = np.linspace(0, self._waypoints[-1].t, 100)
        plt.plot(t_many, [self.orientation(t).x for t in t_many], 
                          label='rot.x(t)')
        plt.plot(t_many, [self.orientation(t).y for t in t_many], 
                          label='rot.y(t)')
        plt.plot(t_many, [self.orientation(t).z for t in t_many], 
                          label='rot.z(t)')
        plt.legend()
        plt.grid()
        plt.show()

        if self._trajectory_history:
            plt.title('velocity from update() 1(3)')
            plt.quiver(
                self._trajectory_history[0], self._trajectory_history[1],
                self._trajectory_history[3], self._trajectory_history[4],
                angles='xy', scale_units='xy')
            plt.xlim(0, )
            plt.xlabel('x')
            plt.ylabel('y')
            plt.grid()
            plt.show()
            plt.title('velocity from update() 2(3)')
            plt.quiver(
                self._trajectory_history[0], self._trajectory_history[2],
                self._trajectory_history[3], self._trajectory_history[5],
                angles='xy', scale_units='xy')
            plt.xlim(0, )
            plt.xlabel('x')
            plt.ylabel('z')
            plt.grid()
            plt.show()
            plt.title('velocity from update() 3(3)')
            plt.quiver(
                self._trajectory_history[1], self._trajectory_history[2],
                self._trajectory_history[4], self._trajectory_history[5],
                angles='xy', scale_units='xy')
            plt.xlim(0, )
            plt.xlabel('y')
            plt.ylabel('z')
            plt.grid()
            plt.show()

            plt.title('orientation from update() 1(3)')
            plt.quiver(
                self._trajectory_history[0], self._trajectory_history[1],
                self._trajectory_history[6], self._trajectory_history[7],
                angles='xy', scale_units='xy')
            plt.xlim(0, )
            plt.xlabel('x')
            plt.ylabel('y')
            plt.grid()
            plt.show()
            plt.title('orientation from update() 2(3)')
            plt.quiver(
                self._trajectory_history[0], self._trajectory_history[1],
                self._trajectory_history[6], self._trajectory_history[8],
                angles='xy', scale_units='xy')
            plt.xlim(0, )
            plt.xlabel('x')
            plt.ylabel('z')
            plt.grid()
            plt.show()
            plt.title('orientation from update() 3(3)')
            plt.quiver(
                self._trajectory_history[0], self._trajectory_history[1],
                self._trajectory_history[7], self._trajectory_history[8],
                angles='xy', scale_units='xy')
            plt.xlim(0, )
            plt.xlabel('y')
            plt.ylabel('z')
            plt.grid()
            plt.show()
