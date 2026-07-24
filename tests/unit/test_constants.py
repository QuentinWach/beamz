import pytest

from beamz.const import EPS_0, LIGHT_SPEED, MU_0


def test_vacuum_constants_match_the_defined_speed_of_light():
    assert pytest.approx(1.0, rel=1e-12) == EPS_0 * MU_0 * LIGHT_SPEED**2
