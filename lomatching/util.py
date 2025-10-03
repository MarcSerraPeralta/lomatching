from collections.abc import Sequence

import numpy as np
import stim
from qec_util.mod2 import decompose_into_basis


RESET_OPS = ["R", "RX", "RY", "RZ"]


def get_observing_region(
    circuit: stim.Circuit, observable: Sequence[int]
) -> np.ndarray:
    """
    Returns the observing region for the specified observable.

    Parameters
    ----------
    circuit
        Circuit with ``TICK``s and measurements.
    observable
        List of measurement IDs in the circuit that correspond to the observable.
        The ID corresponds to the index of the measurement in the circuit.
        For example, ID = ``i`` corresponds to the ``i``th measurement in the circuit,
        starting from ``i = 0``.

    Returns
    -------
    obs_region
        Pauli support of the observing region in the ``TICK``s of ``circuit``.
        Therefore, ``obs_region`` has lenght equal to ``circuit.num_ticks`` and
        its elements correspond to a Pauli string of length ``circuit.num_qubits``.

    Notes
    -----
    This function is intended for unencoded circuits.
    """
    if not isinstance(circuit, stim.Circuit):
        raise TypeError(
            f"'circuit' must be a stim.Circuit, but {type(circuit)} was given."
        )
    circuit = circuit.flattened()
    if not isinstance(observable, Sequence):
        raise TypeError(
            f"'observable' must be a Sequence, but {type(observable)} was given."
        )
    if any(not isinstance(i, int) for i in observable):
        raise TypeError("The elements in observable must be integers.")
    if any(i > circuit.num_measurements - 1 for i in observable):
        raise TypeError(
            "The measurement IDs must be smaller than the number of measurements in the circuit."
        )

    # use the stim.Circuit.detecting_regions to compute the observing regions
    obs_id = circuit.num_detectors
    det_instr = stim.CircuitInstruction(
        name="DETECTOR",
        gate_args=[],
        targets=[stim.target_rec(-circuit.num_measurements + i) for i in observable],
    )
    circuit.append(det_instr)
    det_region = circuit.detecting_regions(
        targets=[stim.DemTarget(f"D{obs_id}")], ignore_anticommutation_errors=True
    )[stim.DemTarget(f"D{obs_id}")]

    # format output to one string character array
    obs_region = np.empty((circuit.num_ticks, circuit.num_qubits), dtype="S1")
    obs_region.fill("I")
    for tick, pauli in det_region.items():
        x, z = pauli.to_numpy()
        for n, (xi, zi) in enumerate(zip(x, z)):
            if xi and zi:
                p = "Y"
            elif xi:
                p = "X"
            elif zi:
                p = "Z"
            else:
                p = "I"
            obs_region[tick, n] = p

    return obs_region


def get_measurement_decomposition(
    circuit: stim.Circuit,
) -> dict[int, None | tuple[int, tuple[int, ...]]]:
    """
    Returns a measurement decomposition in terms of reliable and unreliable
    observables.

    Parameters
    ----------
    circuit
        Stim circuit with measurements. This circuit must correspond to an
        unencoded circuit and contain TICKs after every operation layer.

    Returns
    -------
    meas_decom
        Measurement decomposition corresponding to a dictionary with keys
        being all measurement IDs (i.e. index of the measurement in the circuit,
        starting from 0) and with values corresponding to a list of:
            - ``tuple[int]`` corresponding to reliable observables
            - ``int`` corresponding to outcome from the specified measurement
        To compute the outcome from a measurement, one needs to perform the XOR
        of the outcomes described in the corresponding list, unless it is ``None``,
        which then it corresponds to an unreliable observable and its outcome
        needs to be randomly sampled from a 50-50 distribution. If only one int
        is present in the list, then the measurement corresponds to a reliable
        observable and can be decoded normally.
        See Notes for an example.

    Notes
    -----
    An example of measurement decomposition for the following Bell state preparation
    and measurement:

        |+> --@-- MZ m0
              |
        |0> --X-- MZ m1

    is the following: ``{0: None, 1: ((0,1), 0)}``. This means that m0 is unreliable
    and can be sampled from a 50-50 distribution, while m1 needs to be obtained from
    decoding the reliable observable {m0, m1} and XORing with the outcome of m0.
    There exist another measurement decomposition: ``{0: ((0,1), 1), 1: (None,)}``.
    """
    reset_matrices = get_reset_matrices(circuit)

    meas_decom = {}
    unreliable_reset_support = []
    for ind in range(circuit.num_measurements):
        obs_region = get_observing_region(circuit, [ind])
        reset_support = [anticommute(obs_region, reset) for reset in reset_matrices]

        if sum(reset_support) == 0:
            # reliable measurement
            meas_decom[ind] = (ind,)
            continue

        rank = np.linalg.matrix_rank(
            np.array(unreliable_reset_support + [reset_support], dtype=int)
        )
        if rank == len(unreliable_reset_support) + 1:
            # cannot cancel out the observing regions in the anticommuting resets,
            # thus this is an unreliable observable
            meas_decom[ind] = None
            unreliable_reset_support.append(reset_support)
            continue

        decom = decompose_into_basis(
            vector=reset_support, basis=unreliable_reset_support
        )
        meas_decom[ind] = (tuple(decom + [ind]), *decom)

    return meas_decom


def get_reset_matrices(circuit: stim.Circuit) -> tuple[np.ndarray]:
    """
    Returns the list of matrices describing the resets in the circuit.

    Parameters
    ----------
    circuit
        Stim circuit with resets. Resets cannot appear after the last TICK.

    Returns
    -------
    reset_matrices
        List of matrices with shape ``(circuit.num_ticks, circuit.num_qubits)``
        where the entries indicate the type of reset that occurred before a given
        TICK and qubit. If no reset is present, the element is ``"I"``.
        TICK indices start at 0.
    """
    if not isinstance(circuit, stim.Circuit):
        raise TypeError(
            f"'circuit' must be a stim.Circuit, but {type(circuit)} was given."
        )
    circuit = circuit.flattened()

    reset_matrices = []
    curr_tick = 0
    for instr in circuit:
        if instr.name == "TICK":
            curr_tick += 1
            continue
        if instr.name not in RESET_OPS:
            continue
        if len(instr.targets_copy()) == 0:
            # reset affecting no qubit
            continue

        if curr_tick == circuit.num_ticks:
            raise ValueError("A reset appears after the last TICK.")

        if instr.name in ["R", "RZ"]:
            r = "Z"
        elif instr.name == "RX":
            r = "X"
        elif instr.name == "RY":
            r = "Y"

        for gate_target in instr.targets_copy():
            matrix = np.empty((circuit.num_ticks, circuit.num_qubits), dtype="S1")
            matrix.fill("I")
            matrix[curr_tick, gate_target.qubit_value] = r

        reset_matrices.append(matrix)

    return tuple(reset_matrices)


def anticommute(matrix1: np.ndarray, matrix2: np.ndarray) -> int:
    """
    Returns ``1`` if the two Pauli matrices anticommute and ``0`` otherwise.

    Parameters
    ----------
    matrix1, matrix2
        Matrices with elements corresponding to ``"I"``, ``"X"``, ``"Y"``, ``"Z"```.

    Returns
    -------
    anticommute
        ``1`` if the two matrices anticommute and ``0`` otherwise.
    """
    anticommute = 0
    anticommute ^= (
        np.sum(((matrix1 == b"Y") | (matrix1 == b"Z")) & (matrix2 == b"X")) % 2
    )
    anticommute ^= (
        np.sum(((matrix1 == b"X") | (matrix1 == b"Z")) & (matrix2 == b"Y")) % 2
    )
    anticommute ^= (
        np.sum(((matrix1 == b"X") | (matrix1 == b"Y")) & (matrix2 == b"Z")) % 2
    )
    return anticommute
