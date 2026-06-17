"""The water permittivity must be consistent with the Bjerrum length."""
import math

from pystarc.global_defs.constants import (
    BJERRUM_LENGTH,
    EPS_WATER,
    VACUUM_PERMITTIVITY_KBT,
)


def test_eps_water_consistent_with_bjerrum_length():
    # In the code's internal units the Bjerrum length is
    # l_B = 1 / (4 pi eps_r eps0), so the stored BJERRUM_LENGTH must equal the
    # value implied by EPS_WATER and VACUUM_PERMITTIVITY_KBT.
    """The stored BJERRUM_LENGTH matches the value implied by EPS_WATER and VACUUM_PERMITTIVITY_KBT."""
    lB_from_eps = 1.0 / (4.0 * math.pi * EPS_WATER * VACUUM_PERMITTIVITY_KBT)
    assert abs(lB_from_eps - BJERRUM_LENGTH) < 1e-3
