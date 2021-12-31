import numpy as np
import sys
from numpy.core.numeric import Infinity
from enum import Enum

class BlendingType(Enum):
    noblend = 1
    addingblend = 2
    nonzerovelblend = 3

class TrapezoidalVelocityProfilePlanner:

    def __init__(self, dof, ts, vmax, amax) -> None:
        self.dof = dof
        self.ts = ts
        self.vmax = vmax
        self.amax = amax

    def checkSingleAxisTime(self, joint_idx, q0, qn, *args):

        h = qn - q0
        if len(args) == 0:
            # v0 = 0
            # check if constant velocity exists
            is_vconst = True
            if np.abs(h) >= self.vmax[joint_idx]**2 / self.amax[joint_idx]:
                # with constant velocity
                is_vconst = True
                Ta = self.vmax[joint_idx] / self.amax[joint_idx]
                T = (np.abs(h) * self.amax[joint_idx] + self.vmax[joint_idx]**2) / (self.amax[joint_idx] * self.vmax[joint_idx])
                return is_vconst, Ta, T, self.vmax[joint_idx], h
            else:
                # no constant velocity
                is_vconst = False
                Ta = (np.abs(h) / self.amax[joint_idx])**(1.0/2.0)
                T = 2.0 * Ta
                vlim = self.amax[joint_idx] * Ta
                return is_vconst, Ta, T, vlim, h
        elif len(args) == 1:
            # v0 != 0
            v0 = args[0]

            # check feasibility
            if self.amax[joint_idx] * np.abs(h) < np.abs(v0**2)/2.0:
                print("TVPP infeasible: waypoints and too close and v0 is not realizable.")
                return

            # check if constant velocity exists
            is_vconst = True
            if np.abs(h) * self.amax[joint_idx] >= self.vmax[joint_idx]**2 - v0**2 / 2.0:
                # with constant velocity
                is_vconst = True
                Ta = np.abs((np.sign(h) * self.vmax[joint_idx] - v0) / self.amax[joint_idx])
                Td = np.abs((np.sign(h) * self.vmax[joint_idx] - 0.0) / self.amax[joint_idx])
                T = np.abs(h) / self.vmax[joint_idx] + self.vmax[joint_idx] / (2.0 * self.amax[joint_idx]) * ((1.0 - v0 / (np.sign(h) * self.vmax[joint_idx]))**2 + 1.0)
                return is_vconst, Ta, Td, T, self.vmax[joint_idx], h
            else:
                # no constant velocity
                is_vconst = False
                vlim = (np.abs(h) * self.amax[joint_idx] + v0**2 / 2.0)**(1.0/2.0)
                Ta = np.abs((np.sign(h) * vlim - v0) / self.amax[joint_idx])
                Td = np.abs((np.sign(h) * vlim - 0.0) / self.amax[joint_idx])
                T = Ta + Td
                return is_vconst, Ta, Td, T, vlim, h
        elif len(args) == 2:
            # v0 != 0
            # vn != 0
            v0 = args[0]
            vn = args[1]

            # check feasibility
            if self.amax[joint_idx] * np.abs(h) < np.abs(v0**2 - vn**2)/2.0:
                print("TVPP infeasible: waypoints and too close and v0 / vn is not realizable.")
                return

            # check if constant velocity exists
            is_vconst = True
            if np.abs(h) * self.amax[joint_idx] >= self.vmax[joint_idx]**2 - (v0**2 + vn**2) / 2.0:
                # with constant velocity
                is_vconst = True
                Ta = np.abs((np.sign(h) * self.vmax[joint_idx] - v0) / self.amax[joint_idx])
                Td = np.abs((np.sign(h) * self.vmax[joint_idx] - vn) / self.amax[joint_idx])
                T = np.abs(h) / self.vmax[joint_idx] + self.vmax[joint_idx] / (2.0 * self.amax[joint_idx]) * ((1.0 - v0 / (np.sign(h) * self.vmax[joint_idx]))**2 + (1.0 - vn / (np.sign(h) * self.vmax[joint_idx]))**2)
                return is_vconst, Ta, Td, T, self.vmax[joint_idx], h
            else:
                # no constant velocity
                is_vconst = False
                vlim = (np.abs(h) * self.amax[joint_idx] +  (v0**2 + vn**2) / 2.0)**(1.0/2.0)
                Ta = np.abs((np.sign(h) * vlim - v0) / self.amax[joint_idx])
                Td = np.abs((np.sign(h) * vlim - vn) / self.amax[joint_idx])
                T = Ta + Td
                return is_vconst, Ta, Td, T, vlim, h

    def checkMultiAxisTime(self, q0, qn, *args):

        if len(args) == 0:
            # v0 = 0
            max_T = 0
            max_Ta = 0
            for joint_idx in range(self.dof):
                # check if constant velocity exists
                is_vconst, Ta, T, vlim, h = self.checkSingleAxisTime(joint_idx, q0[joint_idx], qn[joint_idx])
                if T > max_T:
                    max_T = T
                    max_Ta = Ta

            if 2.0 * max_Ta <= max_T:
                is_vconst = True
            else:
                is_vconst = False

            return is_vconst, max_Ta, max_T
        elif len(args) == 1:
            # v0 != 0
            v0 = args[0]

            max_T = 0
            max_Ta = 0
            max_Td = 0
            for joint_idx in range(self.dof):
                # check if constant velocity exists
                is_vconst, Ta, Td, T, vlim, h = self.checkSingleAxisTime(joint_idx, q0[joint_idx], qn[joint_idx], v0[joint_idx])
                if T > max_T:
                    max_T = T
                    max_Ta = Ta
                    max_Td = Td

            if max_Ta + max_Td <= max_T:
                is_vconst = True
            else:
                is_vconst = False

            return is_vconst, max_Ta, max_Td, max_T
        elif len(args) == 2:
            # v0 != 0
            v0 = args[0]
            vn = args[1]

            max_T = 0
            max_Ta = 0
            max_Td = 0
            for joint_idx in range(self.dof):
                # check if constant velocity exists
                is_vconst, Ta, Td, T, vlim, h = self.checkSingleAxisTime(joint_idx, q0[joint_idx], qn[joint_idx], v0[joint_idx], vn[joint_idx])
                if T > max_T:
                    max_T = T
                    max_Ta = Ta
                    max_Td = Td

            if max_Ta + max_Td <= max_T:
                is_vconst = True
            else:
                is_vconst = False

            return is_vconst, max_Ta, max_Td, max_T

    def getProfileWithConstVelocity(self, vmax, amax, Ta, T, q0, qn, *args):
        time_list = []
        position = []
        velocity = [] 
        acceleration = []

        if len(args) == 0:
            # v0 == 0
            v0 = 0.0
            vn = 0.0
            Td = Ta
        elif len(args) == 2:
            # v0 != 0
            v0 = args[0]
            vn = 0.0
            Td = args[1]
        elif len(args) == 3:
            # v0 != 0 and vn != 0
            v0 = args[0]
            vn = args[1]
            Td = args[2]

        itmp = 0
        ttmp = 0.0
        n_Td = 0
        while(ttmp <= T):
            ttmp = self.ts * itmp
            time_list.append(ttmp)

            if ttmp <= Ta:
                position.append(q0 + v0 * ttmp + (np.sign(qn-q0) * vmax - v0) /(2.0 * Ta) * ttmp**2)
                velocity.append(v0 + (np.sign(qn-q0) * vmax - v0) / Ta * ttmp)
                acceleration.append(np.sign(qn-q0) * amax)
            elif ttmp > Ta and ttmp <= T - Td:
                position.append(q0 + v0 * Ta / 2.0 + np.sign(qn-q0) * vmax * (ttmp - Ta / 2.0))
                velocity.append(np.sign(qn-q0) * vmax)
                if ttmp >= T - Td - self.ts/2.0:
                    acceleration.append(-np.sign(qn-q0) * amax)
                else:
                    acceleration.append(0.0)
            elif ttmp > T - Td and ttmp <= T:
                n_Td = n_Td + 1
                position.append(qn - vn * (T - ttmp) - (np.sign(qn-q0) * vmax - vn) / (2.0 * Td) * (T-ttmp)**2)
                velocity.append(vn + (np.sign(qn-q0) * vmax - vn) / Td * (T-ttmp))
                acceleration.append(-np.sign(qn-q0) * amax)
            itmp = itmp + 1

        # add final data
        n_Td = n_Td + 1
        position.append(qn)
        velocity.append(vn)
        acceleration.append(acceleration[-1])

        return time_list, position, velocity, acceleration, n_Td

    def getProfileWithoutConstVelocity(self, vmax, amax, Ta, T, q0, qn, *args):
        time_list = []
        position = []
        velocity = [] 
        acceleration = []

        is_v0_zero = True
        if len(args) == 0:
            # v0 == 0
            v0 = 0.0
            vn = 0.0
            Td = Ta
        elif len(args) == 2:
            # v0 != 0
            is_v0_zero = False
            v0 = args[0]
            vn = 0.0
            Td = args[1]
        elif len(args) == 3:
            # v0 != 0 and vn != 0
            is_v0_zero = False
            v0 = args[0]
            vn = args[1]
            Td = args[2]

        itmp = 0
        ttmp = 0.0
        n_Ta = 0
        while(ttmp <= T):
            ttmp = self.ts * itmp
            time_list.append(ttmp)

            if ttmp <= Ta:
                position.append(q0 + v0 * ttmp + (np.sign(qn-q0) * vmax - v0) /(2.0 * Ta) * ttmp**2)
                velocity.append(v0 + (np.sign(qn-q0) * vmax - v0) / Ta * ttmp)
                if ttmp >= T - Td - self.ts/2.0:
                    acceleration.append(-np.sign(qn-q0) * amax)
                else:
                    acceleration.append(np.sign(qn-q0) * amax)
            elif ttmp > Ta and ttmp <= T:
                n_Ta = n_Ta + 1
                position.append(qn - vn * (T - ttmp) - (np.sign(qn-q0) * vmax - vn) / (2.0 * Td) * (T-ttmp)**2)
                velocity.append(vn + (np.sign(qn-q0) * vmax - vn) / Td * (T-ttmp))
                acceleration.append(-np.sign(qn-q0) * amax)
            itmp = itmp + 1

        # add final data
        n_Ta = n_Ta + 1
        position.append(qn)
        velocity.append(vn)
        acceleration.append(acceleration[-1])

        return time_list, position, velocity, acceleration, n_Ta

    def getSingleAxisSingleMotionProfile(self, joint_idx, q0, qn, *args):
        time_list = []
        position = []
        velocity = []
        acceleration = []

        if len(args) == 0:
            # check if constant velocity exists
            is_vconst, Ta, T, vlim, h = self.checkSingleAxisTime(joint_idx, q0, qn)

            if is_vconst:
                # with constant velocity
                time_list, position, velocity, acceleration, n = self.getProfileWithConstVelocity(vlim, self.amax[joint_idx], Ta, T, q0, qn)
            else:
                # no constant velocity
                time_list, position, velocity, acceleration, n = self.getProfileWithoutConstVelocity(vlim, self.amax[joint_idx], Ta, T, q0, qn)

            return time_list, position, velocity, acceleration

        elif len(args) == 1:
            v0 = args[0]

            # check if constant velocity exists
            is_vconst, Ta, Td, T, vlim, h = self.checkSingleAxisTime(joint_idx, q0, qn, v0)

            if is_vconst:
                # with constant velocity
                time_list, position, velocity, acceleration, n = self.getProfileWithConstVelocity(vlim, self.amax[joint_idx], Ta, T, q0, qn, v0, Td)
            else:
                # no constant velocity
                time_list, position, velocity, acceleration, n = self.getProfileWithoutConstVelocity(vlim, self.amax[joint_idx], Ta, T, q0, qn, v0, Td)

            return time_list, position, velocity, acceleration

        elif len(args) == 2:
            v0 = args[0]
            vn = args[1]

            # check if constant velocity exists
            is_vconst, Ta, Td, T, vlim, h = self.checkSingleAxisTime(joint_idx, q0, qn, v0, vn)

            if is_vconst:
                # with constant velocity
                time_list, position, velocity, acceleration, n = self.getProfileWithConstVelocity(vlim, self.amax[joint_idx], Ta, T, q0, qn, v0, vn, Td)
            else:
                # no constant velocity
                time_list, position, velocity, acceleration, n = self.getProfileWithoutConstVelocity(vlim, self.amax[joint_idx], Ta, T, q0, qn, v0, vn, Td)

            return time_list, position, velocity, acceleration

    def getSingleAxisMultiMotionProfile(self, joint_idx, waypoints, blending, *args):
        time_list = []
        position = []
        velocity = []
        acceleration = []

        is_v0_zero = True
        is_vn_zero = True
        if len(args) == 1:
            is_v0_zero = False
            is_vn_zero = True
            v0 = args[0]
            vn = 0.0
        elif len(args) == 2:
            is_v0_zero = False
            is_vn_zero = False
            v0 = args[0]
            vn = args[1]

        # Compute series of acceleration time
        is_vconst_series = []
        Ta_series = []
        Td_series = []
        T_series = []
        vlim_series = []
        h_series = []

        if is_v0_zero and is_vn_zero:
            # compute acceleration time for each interval: from q[k] to q[k+1]
            for wp_idx in range(len(waypoints)-1):
                # check if constant velocity exists
                is_vconst, Ta, T, vlim, h = self.checkSingleAxisTime(joint_idx, waypoints[wp_idx], waypoints[wp_idx+1])
                is_vconst_series.append(is_vconst)
                Ta_series.append(Ta)
                Td_series.append(Ta)
                T_series.append(T)
                vlim_series.append(vlim)
                h_series.append(h)
        elif is_v0_zero == False or is_vn_zero == False:
            # compute acceleration time for each interval: from q[k] to q[k+1]
            for wp_idx in range(len(waypoints)-1):
                # check if constant velocity exists
                if wp_idx == 0:
                    is_vconst, Ta, Td, T, vlim, h = self.checkSingleAxisTime(joint_idx, waypoints[wp_idx], waypoints[wp_idx+1], v0)
                    is_vconst_series.append(is_vconst)
                    Ta_series.append(Ta) # Ta_series is used for blending, and Td is used for initial blending
                    Td_series.append(Td)
                    T_series.append(T)
                    vlim_series.append(vlim)
                    h_series.append(h)
                elif wp_idx == len(waypoints) - 2:
                    is_vconst, Ta, Td, T, vlim, h = self.checkSingleAxisTime(joint_idx, waypoints[wp_idx], waypoints[wp_idx+1], 0.0, vn)
                    is_vconst_series.append(is_vconst)
                    Ta_series.append(Ta) # Ta_series is used for blending, and Td is used for initial blending
                    Td_series.append(Td)
                    T_series.append(T)
                    vlim_series.append(vlim)
                    h_series.append(h)
                else:
                    is_vconst, Ta, T, vlim, h = self.checkSingleAxisTime(joint_idx, waypoints[wp_idx], waypoints[wp_idx+1])
                    is_vconst_series.append(is_vconst)
                    Ta_series.append(Ta)
                    Td_series.append(Ta)
                    T_series.append(T)
                    vlim_series.append(vlim)
                    h_series.append(h)

        # compute synchronized trajectories
        for wp_idx in range(len(waypoints)-1):
            if is_vconst_series[wp_idx]:
                # with constant velocity
                if is_v0_zero == False and wp_idx == 0:
                    _time_list, _position, _velocity, _acceleration, n_Td = self.getProfileWithConstVelocity(self.vmax[joint_idx], self.amax[joint_idx], Ta_series[wp_idx], T_series[wp_idx], waypoints[wp_idx], waypoints[wp_idx+1], v0, Td_series[wp_idx])
                elif is_vn_zero == False and wp_idx == len(waypoints)-2:
                    _time_list, _position, _velocity, _acceleration, n_Td = self.getProfileWithConstVelocity(self.vmax[joint_idx], self.amax[joint_idx], Ta_series[wp_idx], T_series[wp_idx], waypoints[wp_idx], waypoints[wp_idx+1], 0.0, vn, Td_series[wp_idx])
                else:
                    _time_list, _position, _velocity, _acceleration, n_Td = self.getProfileWithConstVelocity(self.vmax[joint_idx], self.amax[joint_idx], Ta_series[wp_idx], T_series[wp_idx], waypoints[wp_idx], waypoints[wp_idx+1])
            else:
                # no constant velocity
                if is_v0_zero == False and wp_idx == 0:
                    _time_list, _position, _velocity, _acceleration, n_Td = self.getProfileWithoutConstVelocity(vlim_series[wp_idx], self.amax[joint_idx], Ta_series[wp_idx], T_series[wp_idx], waypoints[wp_idx], waypoints[wp_idx+1], v0, Td_series[wp_idx])
                elif is_vn_zero == False and wp_idx == len(waypoints)-2:
                    _time_list, _position, _velocity, _acceleration, n_Td = self.getProfileWithoutConstVelocity(vlim_series[wp_idx], self.amax[joint_idx], Ta_series[wp_idx], T_series[wp_idx], waypoints[wp_idx], waypoints[wp_idx+1], 0.0, vn, Td_series[wp_idx])
                else:
                    _time_list, _position, _velocity, _acceleration, n_Td = self.getProfileWithoutConstVelocity(vlim_series[wp_idx], self.amax[joint_idx], Ta_series[wp_idx], T_series[wp_idx], waypoints[wp_idx], waypoints[wp_idx+1])

            # blending motion
            if blending == BlendingType.noblend:
                # concatinate
                if wp_idx != 0:
                    # Modify last acceleration for waypoint[k]
                    acceleration[-1] = _acceleration[0]

                    # Pop out one data from waypoint[k+1] because first data is stationary
                    _time_list.pop(0)
                    _position.pop(0)
                    _velocity.pop(0)
                    _acceleration.pop(0)

                    # Correct time data
                    _time_list = [(t + time_list[-1]) for t in _time_list]
                    time_list.extend(_time_list)
                    position.extend(_position)
                    velocity.extend(_velocity)
                    acceleration.extend(_acceleration)
                else:
                    time_list = _time_list
                    position = _position
                    velocity = _velocity
                    acceleration = _acceleration

            elif blending == BlendingType.addingblend:
                # concatinate
                if wp_idx != 0:
                    if is_v0_zero == False and wp_idx == 1:
                        # choose min(Td[k-1], Ta[k])
                        Ta_cmpr_list = [Td_series[wp_idx-1], Ta_series[wp_idx]]
                        min_Ta = min(Ta_cmpr_list)
                    else:
                        # choose min(Ta[k-1], Ta[k])
                        Ta_cmpr_list = [Ta_series[wp_idx-1], Ta_series[wp_idx]]
                        min_Ta = min(Ta_cmpr_list)

                    # looking back to (t - Tamin) and start adding two positions, velocities, and accelerations
                    back_idx = n_Td - 1

                    is_direction_change = False
                    # check if direction change happens and resulting acceleration would violate acceleration constraint
                    sign_1st = np.sign(acceleration[-int(back_idx/2)-1])
                    sign_2nd = np.sign(_acceleration[int(back_idx/2)])
                    blended_accel = np.abs(acceleration[-int(back_idx/2)-1] + _acceleration[int(back_idx/2)])
                    if sign_1st == sign_2nd and blended_accel >= self.amax[joint_idx]:
                        is_direction_change = True

                    # If yes, avoid blending
                    if is_direction_change:
                        # Modify last acceleration for waypoint[k]
                        acceleration[-1] = _acceleration[0]

                        # Pop out one data from waypoint[k+1] because first data is stationary
                        _time_list.pop(0)
                        _position.pop(0)
                        _velocity.pop(0)
                        _acceleration.pop(0)

                        # Correct time data
                        _time_list = [(t + time_list[-1]) for t in _time_list]
                        time_list.extend(_time_list)
                        position.extend(_position)
                        velocity.extend(_velocity)
                        acceleration.extend(_acceleration)
                    else:
                        for itmp in range(len(_position)):
                            if itmp < back_idx + 1:
                                position[-back_idx-1 + itmp] = position[-back_idx-1 + itmp] + (_position[itmp] - _position[0])
                                velocity[-back_idx-1 + itmp] = velocity[-back_idx-1 + itmp] + _velocity[itmp]
                                if itmp == 0:
                                        acceleration[-back_idx-1 + itmp] = acceleration[-back_idx-1 + itmp] + _acceleration[itmp]
                                else:
                                    if  np.sign(_acceleration[itmp-1]) != np.sign(_acceleration[itmp]):
                                        acceleration[-back_idx-1 + itmp] = _acceleration[itmp]
                                    else:
                                        acceleration[-back_idx-1 + itmp] = acceleration[-back_idx-1 + itmp] + _acceleration[itmp]
                            else:
                                time_list.append(time_list[-1]+self.ts)
                                position.append(_position[itmp])
                                velocity.append(_velocity[itmp])
                                acceleration.append(_acceleration[itmp])
                else:
                    time_list = _time_list
                    position = _position
                    velocity = _velocity
                    acceleration = _acceleration

        return time_list, position, velocity, acceleration

    def getMultiAxisSingleMotionProfile(self, q0, qn, *args):
        time_list = []
        position = [[] for i in range(self.dof)]
        velocity = [[] for i in range(self.dof)]
        acceleration = [[] for i in range(self.dof)]

        if len(args) == 0:
            # Identify the slowest joint
            is_vconst, max_Ta, max_T = self.checkMultiAxisTime(q0, qn)

            for joint_idx in range(self.dof):
                h = qn[joint_idx] - q0[joint_idx]
                _amax = np.abs(h) / (max_Ta * (max_T - max_Ta))
                _vmax = np.abs(h) / (max_T - max_Ta)

                if is_vconst:
                    # with constant velocity
                    time_list, position[joint_idx], velocity[joint_idx], acceleration[joint_idx], n_Td = self.getProfileWithConstVelocity(_vmax, _amax, max_Ta, max_T, q0[joint_idx], qn[joint_idx])
                else:
                    # no constant velocity
                    time_list, position[joint_idx], velocity[joint_idx], acceleration[joint_idx], n_Td = self.getProfileWithoutConstVelocity(_vmax, _amax, max_Ta, max_T, q0[joint_idx], qn[joint_idx])
        elif len(args) == 1:
            v0 = args[0]

            for joint_idx in range(self.dof):
                if (np.abs(v0[joint_idx]) > np.abs(self.vmax[joint_idx])):
                    print("Initial velocity violates maximum velocity.")
                    return

            # Identify the slowest joint
            is_vconst, max_Ta, max_Td, max_T = self.checkMultiAxisTime(q0, qn, v0)

            for joint_idx in range(self.dof):
                h = qn[joint_idx] - q0[joint_idx]
                _vmax = (2.0 * np.abs(h) - np.sign(h) * v0[joint_idx] * max_Ta) / (2.0 * max_T - max_Ta - max_Td)
                _amax = (_vmax - np.sign(h) * v0[joint_idx]) / max_Ta

                if is_vconst:
                    # with constant velocity
                    time_list, position[joint_idx], velocity[joint_idx], acceleration[joint_idx], n_Td = self.getProfileWithConstVelocity(_vmax, _amax, max_Ta, max_T, q0[joint_idx], qn[joint_idx], v0[joint_idx], max_Td)
                else:
                    # no constant velocity
                    time_list, position[joint_idx], velocity[joint_idx], acceleration[joint_idx], n_Td = self.getProfileWithoutConstVelocity(_vmax, _amax, max_Ta, max_T, q0[joint_idx], qn[joint_idx], v0[joint_idx], max_Td)
        elif len(args) == 2:
            v0 = args[0]
            vn = args[1]

            for joint_idx in range(self.dof):
                if (np.abs(v0[joint_idx]) > np.abs(self.vmax[joint_idx])):
                    print("Initial velocity violates maximum velocity.")
                    return

            # Identify the slowest joint
            is_vconst, max_Ta, max_Td, max_T = self.checkMultiAxisTime(q0, qn, v0, vn)

            for joint_idx in range(self.dof):
                h = qn[joint_idx] - q0[joint_idx]
                _vmax = (2.0 * np.abs(h) - np.sign(h) * v0[joint_idx] * max_Ta - np.sign(h) * vn[joint_idx] * max_Td) / (2.0 * max_T - max_Ta - max_Td)
                _amax = (_vmax - np.sign(h) * v0[joint_idx]) / max_Ta
                #_dmax = (_vmax - np.sign(h) * vn[joint_idx]) / max_Td # TODO

                if is_vconst:
                    # with constant velocity
                    time_list, position[joint_idx], velocity[joint_idx], acceleration[joint_idx], n_Td = self.getProfileWithConstVelocity(_vmax, _amax, max_Ta, max_T, q0[joint_idx], qn[joint_idx], v0[joint_idx], vn[joint_idx], max_Td)
                else:
                    # no constant velocity
                    time_list, position[joint_idx], velocity[joint_idx], acceleration[joint_idx], n_Td = self.getProfileWithoutConstVelocity(_vmax, _amax, max_Ta, max_T, q0[joint_idx], qn[joint_idx], v0[joint_idx], vn[joint_idx], max_Td)

        return time_list, position, velocity, acceleration, n_Td

    def getMultiAxisMultiMotionProfile(self, waypoints, blending, *args):
        time_list = []
        position = [[] for i in range(self.dof)]
        velocity = [[] for i in range(self.dof)]
        acceleration = [[] for i in range(self.dof)]

        is_v0_zero = True
        is_vn_zero = True
        if len(args) == 1:
            is_v0_zero = False
            is_vn_zero = True
            v0 = args[0]
            vn = 0.0
        elif len(args) == 2:
            is_v0_zero = False
            is_vn_zero = False
            v0 = args[0]
            vn = args[1]

        # Compute series of acceleration time
        is_vconst_series = []
        Ta_series = []
        Td_series = []
        T_series = []

        if is_v0_zero and is_vn_zero:
            # compute acceleration time for each interval: from q[k] to q[k+1]
            for wp_idx in range(len(waypoints)-1):
                # check if constant velocity exists
                is_vconst, Ta, T = self.checkMultiAxisTime(waypoints[wp_idx], waypoints[wp_idx+1])
                is_vconst_series.append(is_vconst)
                Ta_series.append(Ta)
                T_series.append(T)
        elif not is_v0_zero or not is_vn_zero:
            # compute acceleration time for each interval: from q[k] to q[k+1]
            for wp_idx in range(len(waypoints)-1):
                # check if constant velocity exists
                if wp_idx == 0:
                    is_vconst, Ta, Td, T = self.checkMultiAxisTime(waypoints[wp_idx], waypoints[wp_idx+1], v0, 0.0)
                    is_vconst_series.append(is_vconst)
                    Ta_series.append(Ta) # Ta_series is used for blending, and Td is used for initial blending
                    Td_series.append(Td)
                    T_series.append(T)
                elif wp_idx == len(waypoints)-1:
                    is_vconst, Ta, Td, T = self.checkMultiAxisTime(waypoints[wp_idx], waypoints[wp_idx+1], 0.0, vn)
                    is_vconst_series.append(is_vconst)
                    Ta_series.append(Ta) # Ta_series is used for blending, and Td is used for initial blending
                    Td_series.append(Td)
                    T_series.append(T)
                else:
                    is_vconst, Ta, T = self.checkMultiAxisTime(waypoints[wp_idx], waypoints[wp_idx+1])
                    is_vconst_series.append(is_vconst)
                    Ta_series.append(Ta)
                    T_series.append(T)

        # blending motion
        for wp_idx in range(len(waypoints)-1):
            # compute synchronized trajectories
            if is_v0_zero == False and wp_idx == 0:
                _time_list, _position, _velocity, _acceleration, n_Td = self.getMultiAxisSingleMotionProfile(waypoints[wp_idx], waypoints[wp_idx+1], v0)
            elif is_vn_zero == False and wp_idx == len(waypoints)-1:
                _time_list, _position, _velocity, _acceleration, n_Td = self.getMultiAxisSingleMotionProfile(waypoints[wp_idx], waypoints[wp_idx+1], 0.0, vn)
            else:
                _time_list, _position, _velocity, _acceleration, n_Td = self.getMultiAxisSingleMotionProfile(waypoints[wp_idx], waypoints[wp_idx+1])

            if blending == BlendingType.noblend:
                # concatinate
                for joint_idx in range(self.dof):
                    if wp_idx != 0:
                        # Modify last acceleration for waypoint[k]
                        acceleration[joint_idx][-1] = _acceleration[joint_idx][0]

                        # Pop out one data from waypoint[k+1] because first data is stationary
                        _time_list.pop(0)
                        _position[joint_idx].pop(0)
                        _velocity[joint_idx].pop(0)
                        _acceleration[joint_idx].pop(0)

                        # Correct time data
                        _time_list = [(t + time_list[-1]) for t in _time_list]
                        if joint_idx == 0:
                            time_list.extend(_time_list)
                        position[joint_idx].extend(_position[joint_idx])
                        velocity[joint_idx].extend(_velocity[joint_idx])
                        acceleration[joint_idx].extend(_acceleration[joint_idx])
                    else:
                        if joint_idx == 0:
                            time_list = _time_list
                        position[joint_idx] = _position[joint_idx]
                        velocity[joint_idx] = _velocity[joint_idx]
                        acceleration[joint_idx] = _acceleration[joint_idx]
            elif blending == BlendingType.addingblend:

                # concatinate
                if wp_idx != 0:

                    if is_v0_zero == False and wp_idx == 1:
                        # choose min(Td[k-1], Ta[k])
                        Ta_cmpr_list = [Td_series[wp_idx-1], Ta_series[wp_idx]]
                        min_Ta = min(Ta_cmpr_list)
                    else:
                        # choose min(Ta[k-1], Ta[k])
                        Ta_cmpr_list = [Ta_series[wp_idx-1], Ta_series[wp_idx]]
                        min_Ta = min(Ta_cmpr_list)

                    # looking back to (t - Tamin) and start adding two positions, velocities, and accelerations
                    back_idx = n_Td - 1

                    is_direction_change = False
                    for joint_idx in range(self.dof):
                        # check if direction change happens and resulting acceleration would violate acceleration constraint
                        sign_1st = np.sign(acceleration[joint_idx][-int(back_idx/2)-1])
                        sign_2nd = np.sign(_acceleration[joint_idx][int(back_idx/2)])
                        blended_accel = np.abs(acceleration[joint_idx][-int(back_idx/2)-1] + _acceleration[joint_idx][int(back_idx/2)])
                        if sign_1st == sign_2nd and blended_accel >= self.amax[joint_idx]:
                            is_direction_change = True
                            break

                    for joint_idx in range(self.dof):
                        # If yes, avoid blending
                        if is_direction_change:
                            # Modify last acceleration for waypoint[k]
                            acceleration[joint_idx][-1] = _acceleration[joint_idx][0]

                            # Pop out one data from waypoint[k+1] because first data is stationary
                            _time_list.pop(0)
                            _position[joint_idx].pop(0)
                            _velocity[joint_idx].pop(0)
                            _acceleration[joint_idx].pop(0)

                            # Correct time data
                            _time_list = [(t + time_list[-1]) for t in _time_list]
                            if joint_idx == 0:
                                time_list.extend(_time_list)
                            position[joint_idx].extend(_position[joint_idx])
                            velocity[joint_idx].extend(_velocity[joint_idx])
                            acceleration[joint_idx].extend(_acceleration[joint_idx])
                        else:
                            for itmp in range(len(_position[joint_idx])):
                                if itmp < back_idx + 1:
                                    position[joint_idx][-back_idx-1 + itmp] = position[joint_idx][-back_idx-1 + itmp] + (_position[joint_idx][itmp] - _position[joint_idx][0])
                                    velocity[joint_idx][-back_idx-1 + itmp] = velocity[joint_idx][-back_idx-1 + itmp] + _velocity[joint_idx][itmp]
                                    if itmp == 0:
                                        acceleration[joint_idx][-back_idx-1 + itmp] = acceleration[joint_idx][-back_idx-1 + itmp] + _acceleration[joint_idx][itmp]
                                    else:
                                        if np.sign(_acceleration[joint_idx][itmp-1]) != np.sign(_acceleration[joint_idx][itmp]):
                                            acceleration[joint_idx][-back_idx-1 + itmp] = _acceleration[joint_idx][itmp]
                                        else:
                                            acceleration[joint_idx][-back_idx-1 + itmp] = acceleration[joint_idx][-back_idx-1 + itmp] + _acceleration[joint_idx][itmp]
                                else:
                                    if joint_idx == 0:
                                        time_list.append(time_list[-1]+self.ts)
                                    position[joint_idx].append(_position[joint_idx][itmp])
                                    velocity[joint_idx].append(_velocity[joint_idx][itmp])
                                    acceleration[joint_idx].append(_acceleration[joint_idx][itmp])
                else:
                    for joint_idx in range(self.dof):
                        if joint_idx == 0:
                            time_list = _time_list
                        position[joint_idx] = _position[joint_idx]
                        velocity[joint_idx] = _velocity[joint_idx]
                        acceleration[joint_idx] = _acceleration[joint_idx]

        return time_list, position, velocity, acceleration