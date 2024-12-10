import numpy as np
import stim

from split_mwpm.greedy_algorithm import standardize_circuit


def test_standardize_circuit():
    circuit = stim.Circuit(
        """
        R 0
        TICK
        TICK
        M 0
        R 1
        TICK
        X 1
        R 0
        TICK
        CNOT 0 1
        TICK
        CNOT 1 0
        TICK
        H 0 1
        TICK
        M 1
        TICK
        TICK
        M 0 
        """
    )

    ops = standardize_circuit(circuit)

    expected_ops = np.array(
        [
            ["R", ""],
            ["I", ""],
            ["M", "R"],
            ["R", "X"],
            ["CX0-1", "CX0-1"],
            ["CX1-0", "CX1-0"],
            ["H", "H"],
            ["I", "M"],
            ["I", ""],
            ["M", ""],
        ]
    )

    assert ops.shape == expected_ops.shape
    assert (ops == expected_ops).all()

    return
