import pytest
import numpy as np
import stim
import matplotlib.pyplot as plt

from somatching.greedy_algorithm import (
    get_ops,
    get_time_hypergraph,
    get_track_ordering,
    greedy_algorithm,
    check_ordering,
    plot_time_hypergraph,
    plot_track,
)


def test_greedy_algorithm():
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
        S 0
        TICK
        M 0 
        """
    )

    tracks = greedy_algorithm(circuit, r_start=0, detector_frame="post-gate")

    expected_tracks = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 2, 2, 1],
            [1, 2, 2, 1],
            [2, 1, 1, 2],
            [2, 1, 1, 2],
            [2, 1, 0, 0],
            [2, 1, 0, 0],
        ]
    )

    assert tracks.shape == expected_tracks.shape
    assert (tracks == expected_tracks).all()
    return


def test_get_ops():
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

    ops = get_ops(circuit)

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


def test_get_time_hypergraph():
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

    edges = get_time_hypergraph(ops, detector_frame="post-gate")

    expected_edges = np.array(
        [
            [[0, 0, -1], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 0, 0], [2, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 0, 0], [2, 0, 0], [0, 0, -1], [0, 0, 0]],
            [[0, -1, -1], [0, 0, 0], [3, 0, 0], [4, 0, 0]],
            [[1, 0, 0], [2, 4, 1], [3, 1, 1], [4, 0, 0]],
            [[1, 3, 1], [2, 0, 0], [3, 0, 0], [4, 2, 1]],
            [[2, 0, 0], [1, 0, 0], [4, 0, 0], [3, 0, 0]],
            [[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]],
            [[1, 0, 0], [2, 1, 1], [0, -1, 0], [0, 0, 0]],
            [[1, 0, 0], [2, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, -1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ]
    )

    assert edges.shape == expected_edges.shape
    assert (edges == expected_edges).all()

    edges = get_time_hypergraph(ops, detector_frame="pre-gate")

    expected_edges = np.array(
        [
            [[0, 0, -1], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 0, 0], [2, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 0, 0], [2, 0, 0], [0, 0, -1], [0, 0, 0]],
            [[0, -1, -1], [0, 0, 0], [3, 0, 0], [4, 0, 0]],
            [[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]],
            [[1, 0, 0], [2, 4, 0], [3, 1, 0], [4, 0, 0]],
            [[1, 3, 0], [2, 0, 0], [3, 0, 0], [4, 2, 0]],
            [[2, 0, 0], [1, 0, 0], [4, 0, 0], [3, 0, 0]],
            [[1, 0, 0], [2, 0, 0], [0, -1, 0], [0, 0, 0]],
            [[1, 0, 0], [2, 1, 0], [0, 0, 0], [0, 0, 0]],
            [[0, -1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ]
    )

    assert edges.shape == expected_edges.shape
    assert (edges == expected_edges).all()

    return


def test_get_track_ordering():
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

    tracks = get_track_ordering(edges, r_start=1000)

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

    tracks = get_track_ordering(edges)

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


def test_get_track_ordering_t_start():
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

    tracks = get_track_ordering(edges, r_start=1000, t_start=np.array([2, 2, 2, 2]))

    expected_tracks = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 2],
            [1, 1, 1, 2],
            [3, 1, 1, 1],
            [3, 2, 1, 1],
            [2, 3, 1, 1],
            [2, 3, 1, 1],
            [2, 2, 0, 0],
            [2, 2, 0, 0],
        ]
    )

    assert tracks.shape == expected_tracks.shape
    assert (tracks == expected_tracks).all()

    tracks = get_track_ordering(edges, t_start=np.array([2, 2, 2, 2]))

    expected_tracks = np.array(
        [
            [2, 2, 0, 0],
            [2, 2, 0, 0],
            [2, 2, 1, 1],
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


def test_check_ordering():
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
    correct_tracks = np.array(
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

    check_ordering(correct_tracks, edges)

    bad_tracks = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 2, 2],
            [2, 1, 1, 2],
            [1, 2, 2, 1],
            [1, 2, 2, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ]
    )

    with pytest.raises(ValueError):
        check_ordering(bad_tracks, edges)

    return


def test_plot_time_hypergraph(show_figures):
    edges = np.array(
        [
            [[0, 0, -1], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 0, 0], [2, 0, 0], [0, 0, -1], [0, 0, 0]],
            [[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]],
            [[1, 0, 0], [2, 4, 0], [3, 1, 0], [4, 0, 0]],
            [[1, 0, 0], [2, 1, 0], [4, 0, 0], [3, 0, 0]],
            [[1, 0, 0], [2, 1, 0], [4, 0, 0], [3, 0, 0]],
            [[0, -1, 0], [0, 0, 0], [0, -1, 0], [0, 0, 0]],
        ]
    )
    _, ax = plt.subplots()

    plot_time_hypergraph(ax, edges=edges)

    if show_figures:
        plt.show()
    plt.close()

    return


def test_plot_track(show_figures):
    edges = np.array(
        [
            [[0, 0, -1], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 0, 0], [2, 0, 0], [0, 0, -1], [0, 0, 0]],
            [[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]],
            [[1, 0, 0], [2, 4, 0], [3, 1, 0], [4, 0, 0]],
            [[1, 0, 0], [2, 1, 0], [4, 0, 0], [3, 0, 0]],
            [[1, 0, 0], [2, 1, 0], [4, 0, 0], [3, 0, 0]],
            [[0, -1, 0], [0, 0, 0], [0, -1, 0], [0, 0, 0]],
        ]
    )
    tracks = np.array(
        [
            [2, 1, 0, 0],
            [2, 1, 3, 1],
            [2, 1, 3, 1],
            [2, 2, 2, 1],
            [2, 3, 1, 2],
            [2, 2, 2, 1],
        ]
    )
    _, ax = plt.subplots()

    plot_time_hypergraph(ax, edges=edges)
    plot_track(ax, tracks, track_id=1, color="red")

    if show_figures:
        plt.show()
    plt.close()

    return
