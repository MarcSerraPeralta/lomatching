import numpy as np
import stim

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
    circuit: stim.Circuit,
    detector_frame: str,
    r_start: int = 1,
    check_ft: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs a greedy algorithm for constructing the decoding subgraphs.

    Parameters
    ----------
    circuit
        Logical circuit with only MZ, RZ, MX, RX, S, H, X, Z, Y, I, CX gates.
        Circuit must start with all qubits being reset and end with all qubits
        being measured. TICKs represent QEC cycles.
        Conditional gates based on outcomes are not allowed.
        Qubits can only perform a single operation inbetween QEC cycles.
    detector_frame
        Detector frame that is used when building the detectors.
        It must be either ``"pre-gate"`` or ``"post-gate"``.
    r_start
        Round in which to start with all tracks in the first decoding subgraph.
        The first QEC cycle is indexed by ``1``.
        By default, 1.
    check_ft
        Checks if the circuit is fault tolerant when decoded with Split-MWPM.

    Returns
    -------
    tracks
        Numpy array of size ``(circuit.num_ticks, 2*circuit.num_qubits)``
        representing the index of the decoding subgraph for each track.
        Value ``-1`` represents that the qubit is inactive.
    tb_before_r
        Numpy array of size ``(circuits.num_ticks, 2*circuit.num_qubits)``
        representing the time boundaries before each given round.
        Value ``0`` represents no time boundary, ``1`` is a closed time boundary,
        and ``2`` is an open time boundary.
        Note that ``tb_before_r[r] != tb_after_r[r-1]`` because of the
        situation where a qubit is measured in e.g. MZ and resetted in RX.
    tb_after_r
        Numpy array of size ``(circuits.num_ticks, 2*circuit.num_qubits)``
        representing the time boundaries after each given round.
        Value ``0`` represents no time boundary, ``1`` is a closed time boundary,
        and ``2`` is an open time boundary.
        Note that ``tb_before_r[r] != tb_after_r[r-1]`` because of the
        situation where a qubit is measured in e.g. MZ and resetted in RX.
    """
    return


def standardize_circuit(circuit: stim.Circuit) -> np.ndarray:
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
                if name == "CX":
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


def get_time_hypergraph_from_ops(
    ops: np.ndarray,
    detector_frame: str,
) -> np.ndarray:
    """Runs an array describing the time edges between nodes.

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
        The index ``2*q`` represents the X stabilizers of qubit with index ``q``
        and ``2*q+1`` the Z ones.

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
    edges = np.zeros((num_rounds + 2, 2 * num_qubits, 3), dtype=int)

    for r, curr_ops in enumerate(ops):
        for q, curr_op in enumerate(curr_ops):
            shift = 0 if detector_frame == "post-gate" else 1
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
                if q == control:
                    edges[r + shift][2 * q + 1][0] = 2 * q + 2
                    edges[r + shift][2 * q][0] = 2 * q + 1
                    edges[r + shift][2 * q][1] = 2 * target + 1
                    edges[r + shift][2 * q][2] = 1 - shift
                elif q == target:
                    edges[r + shift][2 * q][0] = 2 * q + 1
                    edges[r + shift][2 * q + 1][0] = 2 * q + 2
                    edges[r + shift][2 * q + 1][1] = 2 * control + 2
                    edges[r + shift][2 * q + 1][2] = 1 - shift
                else:
                    raise ValueError(
                        f"'CX' gate in qubit {q} does not contain this qubit (i.e. {curr_op})."
                    )
            else:
                raise ValueError(f"{curr_op} is not a valid gate.")

    return edges
