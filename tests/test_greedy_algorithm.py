import numpy as np
import stim

from split_mwpm.greedy_algorithm import (
    standardize_circuit,
    get_time_hypergraph_from_ops,
    get_tracks,
)


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


def test_get_time_hypergraph_from_ops():
    ops = np.array(
        [
            ["R", ""],
            ["I", ""],
            ["M", "R"],
            ["R", "X"],
            ["CX0-1", "CX0-1"],
            ["CX1-0", "CX1-0"],
            ["H", "H"],
            ["I", "M"],
            ["S", ""],
            ["M", ""],
        ]
    )

    edges = get_time_hypergraph_from_ops(ops, detector_frame="post-gate")

    expected_edges = np.array(
        [
            [[0, 0, -1], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 0, 0], [2, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 0, 0], [2, 0, 0], [0, 0, -1], [0, 0, 0]],
            [[0, -1, -1], [0, 0, 0], [3, 0, 0], [4, 0, 0]],
            [[1, 3, 1], [2, 0, 0], [3, 0, 0], [4, 2, 1]],
            [[1, 0, 0], [2, 4, 1], [3, 1, 1], [4, 0, 0]],
            [[2, 0, 0], [1, 0, 0], [4, 0, 0], [3, 0, 0]],
            [[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]],
            [[1, 0, 0], [2, 1, 1], [0, -1, 0], [0, 0, 0]],
            [[1, 0, 0], [2, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, -1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ]
    )

    assert edges.shape == expected_edges.shape
    assert (edges == expected_edges).all()

    edges = get_time_hypergraph_from_ops(ops, detector_frame="pre-gate")

    expected_edges = np.array(
        [
            [[0, 0, -1], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 0, 0], [2, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 0, 0], [2, 0, 0], [0, 0, -1], [0, 0, 0]],
            [[0, -1, -1], [0, 0, 0], [3, 0, 0], [4, 0, 0]],
            [[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]],
            [[1, 3, 0], [2, 0, 0], [3, 0, 0], [4, 2, 0]],
            [[1, 0, 0], [2, 4, 0], [3, 1, 0], [4, 0, 0]],
            [[2, 0, 0], [1, 0, 0], [4, 0, 0], [3, 0, 0]],
            [[1, 0, 0], [2, 0, 0], [0, -1, 0], [0, 0, 0]],
            [[1, 0, 0], [2, 1, 0], [0, 0, 0], [0, 0, 0]],
            [[0, -1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ]
    )

    assert edges.shape == expected_edges.shape
    assert (edges == expected_edges).all()

    return


def test_get_tracks():
    # this comes from the other tests with defect_frame="post-gate"
    edges = np.array(
        [
            [[0, 0, -1], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 0, 0], [2, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 0, 0], [2, 0, 0], [0, 0, -1], [0, 0, 0]],
            [[0, -1, -1], [0, 0, 0], [3, 0, 0], [4, 0, 0]],
            [[1, 3, 1], [2, 0, 0], [3, 0, 0], [4, 2, 1]],
            [[1, 0, 0], [2, 4, 1], [3, 1, 1], [4, 0, 0]],
            [[2, 0, 0], [1, 0, 0], [4, 0, 0], [3, 0, 0]],
            [[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]],
            [[1, 0, 0], [2, 1, 1], [0, -1, 0], [0, 0, 0]],
            [[1, 0, 0], [2, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, -1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ]
    )

    tracks = get_tracks(edges, r_start=1000)

    expected_tracks = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 1],
            [1, 2, 1, 1],
            [2, 2, 1, 1],
            [2, 1, 1, 1],
            [1, 2, 1, 1],
            [1, 2, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ]
    )

    assert tracks.shape == expected_tracks.shape
    assert (tracks == expected_tracks).all()

    tracks = get_tracks(edges)

    expected_tracks = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [2, 1, 1, 2],
            [2, 1, 1, 2],
            [1, 2, 2, 1],
            [1, 2, 2, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ]
    )

    assert tracks.shape == expected_tracks.shape
    assert (tracks == expected_tracks).all()

    return
