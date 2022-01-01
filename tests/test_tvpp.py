import pytest
import numpy as np
from tvpp.tvpp import TrapezoidalVelocityProfilePlanner

def test_checkSingleAxisTime():
    dof = 6
    ts = 0.016
    vmax = np.zeros(dof) # rad/s
    DEG2RAD = np.pi / 180.0
    vmax[0] = 262.5 * DEG2RAD
    vmax[1] = 240.0 * DEG2RAD
    vmax[2] = 300.0 * DEG2RAD
    vmax[3] = 300.0 * DEG2RAD
    vmax[4] = 300.0 * DEG2RAD
    vmax[5] = 480.0 * DEG2RAD

    amax = np.zeros(dof) # rad/s^2
    amax[0] = 1400.0 * DEG2RAD
    amax[1] = 900.0 * DEG2RAD
    amax[2] = 1300.0 * DEG2RAD
    amax[3] = 1800.0 * DEG2RAD
    amax[4] = 1600.0 * DEG2RAD
    amax[5] = 5000.0 * DEG2RAD

    _tvpp = TrapezoidalVelocityProfilePlanner(dof, ts, vmax, amax)
    is_vconst, Ta, T, vlim, h = _tvpp.checkSingleAxisTime(0, 0.0, 1.0)
    assert is_vconst == True