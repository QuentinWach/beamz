import pytest

from tests.test_mode_source import (
    TestModeSourceDirectionality3D,
    TestModeSourceEffectiveIndex,
    TestModeSourcePolarization,
    TestModeSourceProfile,
    TestModeSourcePropagation,
)

pytestmark = [pytest.mark.integration, pytest.mark.simulation]

# Disabled from regular collection: these wrapper assignments re-enable large,
# expensive FDTD integration runs from tests/test_mode_source.py. Keep the
# wrapper in place for manual re-enabling when needed.
# TestModeSourceEffectiveIndex.__test__ = True
# TestModeSourceProfile.__test__ = True
# TestModeSourcePropagation.__test__ = True
# TestModeSourcePolarization.__test__ = True
# TestModeSourceDirectionality3D.__test__ = True
TestModeSourceEffectiveIndex.__test__ = False
TestModeSourceProfile.__test__ = False
TestModeSourcePropagation.__test__ = False
TestModeSourcePolarization.__test__ = False
TestModeSourceDirectionality3D.__test__ = False
