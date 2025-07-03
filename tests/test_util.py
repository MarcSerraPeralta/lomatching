import numpy as np
import stim
from lomatching import get_observing_region, get_measurement_decomposition
from lomatching.util import anticommute, get_reset_matrices


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
            [b"Z", b"I"],
            [b"Z", b"I"],
            [b"I", b"I"],
            [b"I", b"I"],
        ],
        dtype="|S1",
    )
    assert (expected_obs_region == obs_region).all()

    obs_region = get_observing_region(circuit, observable=[1])

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


def test_get_measurement_decomposition():
    circuit = stim.Circuit(
        """
        RX 0
        R 1
        TICK
        CNOT 0 1
        TICK
        M 0 1
        """
    )

    meas_decom = get_measurement_decomposition(circuit)

    expected_meas_decom = {0: None, 1: ((0, 1), 0)}

    assert meas_decom == expected_meas_decom

    circuit = stim.Circuit(
        """
        RX 0
        R 1 2
        TICK
        CNOT 0 1
        CNOT 1 2
        TICK
        M 0 1 2
        """
    )

    meas_decom = get_measurement_decomposition(circuit)

    expected_meas_decom = {0: None, 1: ((0, 1), 0), 2: ((0, 2), 0)}

    assert meas_decom == expected_meas_decom

    circuit = stim.Circuit(
        """
        RX 0
        TICK
        R 1 2
        TICK
        CNOT 0 1
        TICK
        H 0 1 2
        TICK
        H 0 1 2
        TICK
        CNOT 1 2
        TICK
        M 1 2
        TICK 
        M 0
        """
    )

    meas_decom = get_measurement_decomposition(circuit)

    expected_meas_decom = {0: None, 1: ((0, 1), 0), 2: ((0, 2), 0)}

    assert meas_decom == expected_meas_decom

    circuit = stim.Circuit(
        """
        RX 0
        TICK
        R 1 2
        TICK
        M 1 2
        TICK 
        M 0
        """
    )

    meas_decom = get_measurement_decomposition(circuit)

    expected_meas_decom = {0: (0,), 1: (1,), 2: None}

    assert meas_decom == expected_meas_decom

    return


def test_anticommute():
    matrix1 = np.array([["I", "X", "X"]], dtype="S1")
    matrix2 = np.array([["I", "X", "X"]], dtype="S1")

    assert anticommute(matrix1, matrix2) == 0

    matrix1 = np.array([["I", "Y", "Z"]], dtype="S1")
    matrix2 = np.array([["I", "X", "X"]], dtype="S1")

    assert anticommute(matrix1, matrix2) == 0

    matrix1 = np.array([["I", "X", "X"]], dtype="S1")
    matrix2 = np.array([["I", "Z", "X"]], dtype="S1")

    assert anticommute(matrix1, matrix2) == 1

    return


def test_get_reset_matrices():
    circuit = stim.Circuit(
        """
        R 0
        TICK
        RX 2
        TICK
        R
        H 0 1
        TICK
        """
    )

    reset_matrices = get_reset_matrices(circuit)

    expected_reset_matrices = (
        np.array(
            [[b"Z", b"I", b"I"], [b"I", b"I", b"I"], [b"I", b"I", b"I"]], dtype="S1"
        ),
        np.array(
            [[b"I", b"I", b"I"], [b"I", b"I", b"X"], [b"I", b"I", b"I"]], dtype="S1"
        ),
    )

    for r, exp_r in zip(reset_matrices, expected_reset_matrices):
        assert (r == exp_r).all()

    return
