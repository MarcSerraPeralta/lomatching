from collections.abc import Sequence

import numpy as np
import stim


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
        targets=[
            stim.target_rec(-(circuit.num_measurements - 1) + i) for i in observable
        ],
    )
    circuit.append(det_instr)
    det_region = circuit.detecting_regions(
        targets=[stim.DemTarget(f"D{obs_id}")], ignore_anticommutation_errors=True
    )[stim.DemTarget(f"D{obs_id}")]

    # format output to one string character array
    obs_region = np.empty((circuit.num_ticks, circuit.num_qubits), dtype="S1")
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
