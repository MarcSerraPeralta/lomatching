import numpy as np
import stim
from lomatching import get_observing_region


def test_get_observing_region():
    circuit = stim.Circuit(
        """
        R 0
        TICK
        H 0 1
        TICK
        H 0
        TICK
        RX 1
        TICK
        CNOT 0 1
        TICK
        M 0 
        TICK
        X 1
        TICK
        M 1
        DETECTOR rec[-1]
        DETECTOR rec[-2]
        """
    )

    obs_region = get_observing_region(circuit, observable=[0])

    expected_obs_region = np.array(
        [
            [b"Z", b"I"],
            [b"X", b"I"],
            [b"Z", b"I"],
            [b"Z", b"Z"],
            [b"I", b"Z"],
            [b"I", b"Z"],
            [b"I", b"Z"],
        ],
        dtype="|S1",
    )
    assert (expected_obs_region == obs_region).all()

    return
