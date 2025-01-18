import numpy as np
import stim
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESET_INSTR = [
    "R",
    "RZ",
    "RX",
]
MEAS_INSTR = [
    "M",
    "MZ",
    "MX",
]
VALID_INSTR = [
    *RESET_INSTR,
    *MEAS_INSTR,
    "TICK",
    "S",
    "H",
    "X",
    "Z",
    "Y",
    "I",
    "CX",
]


def greedy_algorithm(
    circuit: stim.Circuit | np.ndarray,
    detector_frame: str,
    r_start: int = 0,
    t_start: np.ndarray | None = None,
) -> np.ndarray:
    """
    Wrapper for ``get_ops``, ``get_time_hypergraph`` and ``get_track_ordering``.
    See each individual function for more information.
    """
    ops = get_ops(circuit) if isinstance(circuit, stim.Circuit) else circuit
    edges = get_time_hypergraph(ops, detector_frame=detector_frame)
    tracks = get_track_ordering(edges, r_start=r_start, t_start=t_start)
    return tracks


def get_ops(circuit: stim.Circuit) -> np.ndarray:
    """
    Runs an array describing the gate between ticks.

    Parameters
    ----------
    circuit
        Logical circuit with only MZ, RZ, MX, RX, S, H, X, Z, Y, I, CNOT gates.
        Circuit must start with all qubits being reset and end with all qubits
        being measured. TICKs represent QEC cycles.
        Conditional gates based on outcomes are not allowed.
        Qubits can only perform a single operation inbetween QEC cycles.
        The next operation of a measurement must be a reset.

    Returns
    -------
    ops
        Numpy array of size ``(circuit.num_ticks + 1, circuit.num_qubits)``
        representing the gates performed in each qubit between QEC cycles.
    """
    if not isinstance(circuit, stim.Circuit):
        raise TypeError(
            f"'circuit' must be a stim.Circuit, but {type(circuit)} was given."
        )

    circuit = circuit.flattened()
    num_rounds = circuit.num_ticks
    num_qubits = circuit.num_qubits

    # split the circuit into blocks
    blocks = [[]]
    for instr in circuit:
        if instr.name not in VALID_INSTR:
            raise ValueError(f"{instr.name} is not a support instruction.")
        if instr.name == "TICK":
            blocks.append([])
            continue

        blocks[-1].append(instr)

    # indentify operations done in each qubit
    ops = np.empty((num_rounds + 1, num_qubits), dtype=object)
    active_qubits = np.zeros(num_qubits, dtype=bool)
    for r, block in enumerate(blocks):
        curr_ops = {q: [] for q in range(num_qubits)}
        for instr in block:
            name = instr.name
            qubits = np.array([t.value for t in instr.targets_copy()])

            if name in RESET_INSTR:
                active_qubits[qubits] = True

            for i, q in enumerate(qubits):
                if "CX" in name:
                    name = (
                        f"CX{q}-{qubits[i+1]}" if i % 2 == 0 else f"CX{qubits[i-1]}-{q}"
                    )
                curr_ops[q].append(name)
                if not active_qubits[q]:
                    raise ValueError(
                        "A reset must be placed after every measurement and at start."
                    )

            if name in MEAS_INSTR:
                active_qubits[qubits] = False

        if any(len(o) > 1 for o in curr_ops.values()):
            raise ValueError(
                "Qubits must only perform a single operation inbetween QEC cycles."
            )

        # if no instructions, set to idling
        for q in range(num_qubits):
            if curr_ops[q]:
                ops[r][q] = curr_ops[q][0]  # it is a list of a single element
            elif active_qubits[q]:
                ops[r][q] = "I"
            else:
                ops[r][q] = ""

    if active_qubits.any():
        raise ValueError("Circuit must end with all qubits being measured")

    return ops


def get_time_hypergraph(ops: np.ndarray, detector_frame: str) -> np.ndarray:
    """Runs an array describing the time edges between time nodes.

    Parameters
    ----------
    ops
        Array of size ``(num_rounds + 1, num_qubits)``
        representing the gates performed in each qubit between QEC cycles.
        This array can be generated from a ``stim.Circuit`` using
        ``standardize_circuit``.
    detector_frame
        Detector frame that is used when building the detectors.
        It must be either ``"pre-gate"`` or ``"post-gate"``.

    Returns
    -------
    edges
        Numpy array of size ``(num_rounds + 2, 2 * num_qubits, 3)``
        representing the time (hyper)edges. See Notes for more information
        about this format.
        The value ``2 * q + 1`` represents the X stabilizers of qubit with index ``q``
        and ``2 * q + 2`` the Z ones.

    Notes
    -----
    In a given time slice, there are ``2*circuit.num_qubits`` time (hyper)edges
    if the time boundary ones are compressed as follows:
    Ni-A B-Nj
    with Ni&Nj representing decoding sections and A&B representing the type and presence
    of time boundary node, i.e. ``0`` for no boundary, ``-1`` for open
    time boundary. It is not needed to store the type of boundary for the greedy algorithm,
    but it is useful for checking if the condition for fault tolerance is satisfied.
    If (hyper)edges have only support on time consecutive nodes, then a vector
    of length 3 can encode all options, with:
    Time edge = [Ni, 0, Nj, 1, 0, 0] = [Nj, 0, 0]
    Time weight-3 hyperedge = [Ni, 0, Nj, 1, Nk, Rk] = [Nj, Nk, Rk]
    Time-boundary edge = [Ni, 0, 0, 0, A, B] = [0, A, B]
    Inactive qubit = [Ni, 0, 0, 0, 0, 0] = [0, 0, 0]
    with Ni being the node in round r when considering rounds r and r+1,
    with Ni being a decoding section (:math:`N_i \\in [1,...,2N_q]),
    and Ri being the time position of Ni relative to the round of the gate.
    Note that the general structure of the vector is
    [Nj, Nk, Rk] except for the boundary edges which have the form
    [0, A, B] because Nj >= 1.
    """
    if not isinstance(ops, np.ndarray):
        raise TypeError(f"'ops' must be a np.ndarray, but {type(ops)} was given.")
    if detector_frame not in ["pre-gate", "post-gate"]:
        raise ValueError(
            "'detector_frame' must be either 'pre-gate' or 'post-gate', "
            f"but {detector_frame} was given."
        )

    num_rounds, num_qubits = ops.shape[0] - 1, ops.shape[1]
    shift = 0 if detector_frame == "post-gate" else 1
    edges = np.zeros((num_rounds + 2, 2 * num_qubits, 3), dtype=int)

    for r, curr_ops in enumerate(ops):
        for q, curr_op in enumerate(curr_ops):
            if curr_op == "":
                continue
            elif curr_op in ["R", "RZ"]:
                if shift:
                    edges[r][2 * q][2] = -1
                    edges[r + 1][2 * q][0] = 2 * q + 1
                    edges[r + 1][2 * q + 1][0] = 2 * q + 2
                else:
                    edges[r][2 * q][2] = -1
            elif curr_op == "RX":
                if shift:
                    edges[r][2 * q + 1][2] = -1
                    edges[r + 1][2 * q][0] = 2 * q + 1
                    edges[r + 1][2 * q + 1][0] = 2 * q + 2
                else:
                    edges[r][2 * q + 1][2] = -1
            elif curr_op in ["M", "MZ"]:
                if shift:
                    edges[r + 1][2 * q][1] = -1
                else:
                    edges[r + 1][2 * q][1] = -1
                    edges[r][2 * q][0] = 2 * q + 1
                    edges[r][2 * q + 1][0] = 2 * q + 2
            elif curr_op == "MX":
                if shift:
                    edges[r + 1][2 * q + 1][1] = -1
                else:
                    edges[r + 1][2 * q + 1][1] = -1
                    edges[r][2 * q][0] = 2 * q + 1
                    edges[r][2 * q + 1][0] = 2 * q + 2
            elif curr_op in ["I", "X", "Y", "Z"]:
                edges[r + shift][2 * q][0] = 2 * q + 1
                edges[r + shift][2 * q + 1][0] = 2 * q + 2
            elif curr_op == "H":
                edges[r + shift][2 * q][0] = 2 * q + 2
                edges[r + shift][2 * q + 1][0] = 2 * q + 1
            elif curr_op == "S":
                edges[r + shift][2 * q][0] = 2 * q + 1
                edges[r + shift][2 * q + 1][0] = 2 * q + 2
                edges[r + shift][2 * q + 1][1] = 2 * q + 1
                edges[r + shift][2 * q + 1][2] = 1 - shift
            elif "CX" in curr_op:
                control = int(curr_op[2:].split("-")[0])
                target = int(curr_op[2:].split("-")[1])
                if q == target:
                    edges[r + shift][2 * q + 1][0] = 2 * q + 2
                    edges[r + shift][2 * q][0] = 2 * q + 1
                    edges[r + shift][2 * q][1] = 2 * control + 1
                    edges[r + shift][2 * q][2] = 1 - shift
                elif q == control:
                    edges[r + shift][2 * q][0] = 2 * q + 1
                    edges[r + shift][2 * q + 1][0] = 2 * q + 2
                    edges[r + shift][2 * q + 1][1] = 2 * target + 2
                    edges[r + shift][2 * q + 1][2] = 1 - shift
                else:
                    raise ValueError(
                        f"'CX' gate in qubit {q} does not contain this qubit (i.e. {curr_op})."
                    )
            else:
                raise ValueError(f"{curr_op} is not a valid gate.")

    return edges


def get_track_ordering(
    edges: np.ndarray, r_start: int = 0, t_start: np.ndarray | None = None
) -> np.ndarray:
    """
    Returns an array specifying the ordering index for each time node.

    Parameters
    ----------
    edges
        Numpy array of size ``(num_rounds + 2, 2 * num_qubits, 3)``
        representing the time (hyper)edges. See Notes for more information
        about this format.
        The value ``2 * q + 1`` represents the X stabilizers of qubit with index ``q``
        and ``2 * q + 2`` the Z ones.
        See ``get_time_hypergraph_from_ops`` for more information of the
        structure of ``edges``.
    r_start
        (Decoding section) time index in which to set all the decoding sections
        at that given time index (or time slice) to track 1.
        The first nodes have index ``r = 0`` and last nodes have index
        ``num_rounds`` (for a total of ``num_rounds + 1`` nodes).
    t_start
        Initial track indices at ``r_start``. By default, all 1s.

    Returns
    -------
    tracks
        Numpy array of size ``(num_rounds + 1, 2 * num_qubits)`` that
        specifies the ordering index for each time node. The ordering index
        starts at ``1``. Values of ``0`` indicate that the qubit is not active.
    """
    if not isinstance(edges, np.ndarray):
        raise TypeError(f"'edges' must be a np.ndarray, but {type(edges)} was given.")
    if not isinstance(r_start, int):
        raise TypeError(f"'r_start' must be an int, but {type(r_start)} was given.")
    num_rounds, num_tracks = edges.shape[0] - 2, edges.shape[1]
    if t_start is None:
        t_start = np.ones(num_tracks, dtype=int)
    if not isinstance(t_start, np.ndarray):
        raise TypeError(
            f"'t_start' must be a np.ndarray, but {type(t_start)} was given."
        )
    if t_start.dtype != np.int64:
        raise TypeError(f"'t_start' must be an array of dtype = np.int64.")

    r_start = np.clip(r_start, 0, num_rounds + 1 - 1)
    tracks = np.zeros((num_rounds + 1, num_tracks), dtype=int)

    # prepare tracks at r_start
    # set tracks to 1 unless the qubit is inactive
    edges_before = edges[r_start]
    edges_after = edges[r_start + 1]
    inactive = (
        (edges_before[:, 0] == 0)
        * (edges_before[:, 2] == 0)
        * (edges_after[:, 0] == 0)
        * (edges_after[:, 1] == 0)
    )
    tracks[r_start] = np.where(inactive, 0, t_start)  # 1 - inactive.astype(int)

    # process forward in time.
    # tracks[r] and edges[r+1] are used to compute tracks[r+1]
    curr_round = r_start
    while curr_round < num_rounds:
        curr_edges = edges[curr_round + 1]
        # it is important to first process the measurement (it 'kills'
        # tracks) and then process the resets (it 'creates' tracks)
        # for situations like MR. Hyperedges must not be processed
        # until everything has been created because they use the index
        # from another track (which if it has not been updated it would be 0)
        for node_ind, curr_edge in enumerate(curr_edges):
            if curr_edge[0] == 0:
                # measurement or inactive qubit
                tracks[curr_round + 1, node_ind] = 0
            elif curr_edge[1] == 0:
                # time edge (may activate qubits)
                if tracks[curr_round, node_ind] == 0:
                    tracks[curr_round, node_ind] = 1

                other_node_ind = curr_edge[0] - 1  # Ni starts at 1
                tracks[curr_round + 1, other_node_ind] = tracks[curr_round, node_ind]

        for node_ind, curr_edge in enumerate(curr_edges):
            if curr_edge[0] != 0 and curr_edge[1] != 0:
                # time hyperedge (with 2 nodes on 'node_id' and 1 node in 'other_node_id'
                # it may activate qubits
                if tracks[curr_round, node_ind] == 0:
                    tracks[curr_round, node_ind] = 1

                other_node_ind = curr_edge[1] - 1  # Nj starts at 1
                track_i = tracks[curr_round, node_ind]
                track_j = tracks[curr_round, other_node_ind]
                if track_i < track_j:
                    tracks[curr_round + 1, node_ind] = track_i
                elif track_i == track_j:
                    tracks[curr_round + 1, node_ind] = track_i + 1
                else:
                    tracks[curr_round + 1, node_ind] = track_j

        curr_round += 1

    # process backward in time.
    # tracks[r] and edges[r] are used to compute tracks[r-1]
    curr_round = r_start
    while curr_round > 0:
        curr_edges = edges[curr_round]
        # it is important to first process the measurement (it 'kills'
        # tracks) and then process the resets (it 'creates' tracks)
        # for situations like MR. Hyperedges must not be processed
        # until everything has been created because they use the index
        # from another track (which if it has not been updated it would be 0)
        for node_ind, curr_edge in enumerate(curr_edges):
            if curr_edge[0] == 0:
                # reset or inactive qubit
                tracks[curr_round - 1, node_ind] = 0
            elif curr_edge[1] == 0:
                # time edge (may activate qubits)
                if tracks[curr_round, node_ind] == 0:
                    tracks[curr_round, node_ind] = 1

                other_node_ind = curr_edge[0] - 1  # Ni starts at 1
                tracks[curr_round - 1, other_node_ind] = tracks[curr_round, node_ind]

        for node_ind, curr_edge in enumerate(curr_edges):
            if curr_edge[0] != 0 and curr_edge[1] != 0:
                # time hyperedge (with 2 nodes on 'node_id' and 1 node in 'other_node_id')
                # it may activate qubits
                if tracks[curr_round, node_ind] == 0:
                    tracks[curr_round, node_ind] = 1

                other_node_ind = curr_edge[1] - 1  # Nj starts at 1
                track_i = tracks[curr_round, node_ind]
                track_j = tracks[curr_round, other_node_ind]
                if track_i < track_j:
                    tracks[curr_round - 1, node_ind] = track_i
                elif track_i == track_j:
                    tracks[curr_round - 1, node_ind] = track_i + 1
                else:
                    tracks[curr_round - 1, node_ind] = track_j

        curr_round -= 1

    return tracks


def check_ordering(track_ordering: np.ndarray, edges: np.ndarray) -> None:
    """
    Raises an error if the given track ordering for the given time hypergraph
    does not fulfill the conditions of Task 1.

    Parameters
    ----------
    track_ordering
        See output ``tracks`` from ``get_track_ordering``.
    edges
        See output ``edges`` from ``get_time_hypergraph``.
    """
    if not isinstance(track_ordering, np.ndarray):
        raise TypeError(
            f"'track_ordering' must be a np.ndarray, but {type(track_ordering)} was given."
        )
    if not isinstance(edges, np.ndarray):
        raise TypeError(f"'edges' must be a np.ndarray, but {type(edges)} was given.")
    if (track_ordering.shape[0] != edges.shape[0] - 1) or (
        track_ordering.shape[1] != edges.shape[1]
    ):
        raise ValueError("'track_ordering' and 'edges' do not have correct shapes.")

    for r, curr_edges in enumerate(edges):
        if r in [0, edges.shape[0] - 1]:
            # open/close time boundaries
            continue

        for node_ind, curr_edge in enumerate(curr_edges):
            if curr_edge[0] == 0:
                # open/close time boundary
                continue
            elif curr_edge[1] == 0:
                # time edge
                other_node_ind = curr_edge[0] - 1  # Ni starts at 1
                track_1 = track_ordering[r - 1][node_ind]
                track_2 = track_ordering[r][other_node_ind]
                if track_1 != track_2:
                    raise ValueError(
                        f"Incorrectly split time edge (r={r} n={node_ind})."
                    )
            else:
                # time hyperedge
                other_node_ind = curr_edge[1] - 1  # Nj starts at 1
                r_relative = curr_edge[2]
                track_1 = track_ordering[r - 1][node_ind]
                track_2 = track_ordering[r][node_ind]
                track_3 = track_ordering[r - 1 + r_relative][other_node_ind]
                unique_tracks = set([track_1, track_2, track_3])

                if len(unique_tracks) == 3:
                    raise ValueError(
                        "Hyperedge is split into three different subgraphs "
                        f"(r={r} n={node_ind})."
                    )
                if len(unique_tracks) == 1:
                    raise ValueError(
                        f"Hyperedge appears in a single subgraph (r={r} n={node_ind})."
                    )

                counts = {t: 0 for t in unique_tracks}
                for t in [track_1, track_2, track_3]:
                    counts[t] += 1
                reverse = {v: k for k, v in counts.items()}

                if reverse[2] > reverse[1]:
                    raise ValueError(
                        f"Hyperedge is incorrectly split (r={r} n={node_ind})."
                    )

    return


def plot_time_hypergraph(ax: plt.Axes, edges: np.ndarray) -> plt.Axes:
    """Plots the time hypergraph obtained by ``get_time_hypergraph``."""
    if not isinstance(edges, np.ndarray):
        raise TypeError(f"'edges' must be a np.ndarray, but {type(edges)} was given.")
    if len(edges.shape) != 3:
        raise TypeError(
            f"'edges' must be 3D array, but {len(edges.shape)}D array was given."
        )
    if edges.shape[2] != 3:
        raise TypeError(
            f"'edges' does not follow the formatting specified in ``get_time_hypergraph``."
        )

    _, num_stabs, _ = edges.shape
    RADIUS = 0.15
    BIG_RADIUS = 0.2

    # add qubit labels
    for q in range(num_stabs // 2):
        ax.text(0, 2 * q, f"q{q} X", verticalalignment="center_baseline")
        ax.text(0, 2 * q + 1, f"q{q} Z", verticalalignment="center_baseline")

    for t, slice in enumerate(edges):
        for s, stab_edge in enumerate(slice):
            if (stab_edge == 0).all():
                continue
            elif stab_edge[2] == -1:
                # reset open time boundary
                c = mpatches.Circle((t + 1, s), BIG_RADIUS, color="red")
                ax.add_patch(c)
                c = mpatches.Circle((t + 1, s), RADIUS, color="black")
                ax.add_patch(c)
            elif stab_edge[1] == -1:
                # measurement open time boundary
                c = mpatches.Circle((t, s), BIG_RADIUS, color="red")
                ax.add_patch(c)
                c = mpatches.Circle((t, s), RADIUS, color="black")
                ax.add_patch(c)
            elif stab_edge[0] == 0:
                # closed time boundary edge
                pass
            elif stab_edge[1] == 0:
                # time edge
                other_s = stab_edge[0] - 1
                c = mpatches.Circle((t, s), RADIUS, color="black")
                ax.add_patch(c)
                c = mpatches.Circle((t + 1, other_s), RADIUS, color="black")
                ax.add_patch(c)
                ax.plot([t, t + 1], [s, other_s], color="black", linestyle="-")
            else:
                # time hyperedge
                other_s = stab_edge[1] - 1
                shift = stab_edge[2]
                c = mpatches.Circle((t, s), RADIUS, color="black")
                ax.add_patch(c)
                c = mpatches.Circle((t + 1, s), RADIUS, color="black")
                ax.add_patch(c)
                c = mpatches.Circle((t + shift, other_s), RADIUS, color="black")
                ax.add_patch(c)
                p = mpatches.Polygon(
                    ((t, s), (t + 1, s), (t + shift, other_s)),
                    color=np.random.rand(3),
                    alpha=0.5,
                )
                ax.add_patch(p)

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.get_yaxis().set_visible(False)
    ax.set_xlim(xmin=-0.5)
    ax.set_xlabel("QEC round")
    ax.set_ylabel("top to bottom: q0 X, q0 Z, q1 X...")

    return ax
