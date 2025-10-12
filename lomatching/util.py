from collections.abc import Collection, Sequence

from itertools import chain
import numpy as np
import numpy.typing as npt
from galois import GF2
import stim

Coords = tuple[float, ...]
PauliRegion = dict[int, stim.PauliString]

RESET_INSTRS = ["R", "RX", "RY", "RZ", "MR", "MRX", "MRY", "MRZ"]
MEAS_INSTRS = ["M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ"]


def get_reliable_observables(circuit: stim.Circuit) -> list[set[int]]:
    """Returns a complete set of reliable observables.

    Parameters
    ----------
    circuit
        Encoded or unencoded circuit. Stim reset operations for any qubit must
        be the only operation done on that qubit between `TICK`s. Qubits must
        be explicitly reset.

    Returns
    -------
    observables
        Complete set of reliable observables in 'circuit'. The observables are
        specified using their observble index from 'circuit'.
    """
    if not isinstance(circuit, stim.Circuit):
        raise TypeError(
            f"'circuit' must be a stim.Circuit, but {type(circuit)} was given."
        )

    resets = get_all_reset_paulistrings(circuit)
    obs_regions = {
        obs_id: get_observing_region(circuit, obs_id)
        for obs_id in range(circuit.num_observables)
    }

    # get a basis of the null space to obtain the missing reliable observables.
    # A * observables = resets  --> null space of A = reliable observables,
    # where A[r, o] = 1 if observable o and reset r anticommute, otherwise 0.
    matrix = np.zeros((len(resets), len(obs_regions)), dtype=int)
    for obs_id, obs_region in obs_regions.items():
        for reset_id, reset in resets.items():
            if not commute(reset, obs_region):
                matrix[reset_id, obs_id] = 1

    null_space = GF2(matrix).null_space()
    reliable_obs: list[set[int]] = []
    for k in range(len(null_space)):
        vector = null_space[k].tolist()
        inds = [i for i, v in enumerate(vector) if v != 0]
        reliable_obs.append(set(inds))

    return reliable_obs


def commute(region_a: PauliRegion, region_b: PauliRegion) -> bool:
    """Checks if two Pauli regions commute."""
    common_ticks = set(region_a).intersection(region_b)

    # keep track of anticommutations otherwise if tick_commute = []
    # it will return False which is not correct.
    tick_anticommute: list[bool] = []
    for tick in common_ticks:
        tick_anticommute.append(not region_a[tick].commutes(region_b[tick]))

    return not bool(sum(tick_anticommute) % 2)


def get_all_reset_paulistrings(circuit: stim.Circuit) -> dict[int, PauliRegion]:
    """Returns all the backpropagated reset regions for all resets in the given circuit.

    Parameters
    ----------
    circuit
        Encoded or unencoded circuit. Stim reset operations for any qubit must
        be the only operation done on that qubit between `TICK`s. Qubits must
        be explicitly reset.

    Returns
    -------
    resets
        Dictionary whose keys are the reset index in the given circuit
        and values are the corresponding backpropagated reset region.
    """
    if not isinstance(circuit, stim.Circuit):
        raise TypeError(
            f"'circuit' must be a stim.Circuit, but {type(circuit)} was given."
        )

    resets: dict[int, PauliRegion] = {}
    current_reset = 0
    current_tick = 0
    for instr in circuit.flattened():
        if instr.name == "TICK":
            current_tick += 1
            continue
        elif instr.name not in RESET_INSTRS:
            continue

        for qubit_id in instr.targets_copy():
            reset_pauli = "Z"
            if instr.name[-1] == "X":
                reset_pauli = "X"
            elif instr.name[-1] == "Y":
                reset_pauli = "Y"

            reset_paulistring = stim.PauliString(circuit.num_qubits)
            reset_paulistring[qubit_id.value] = reset_pauli
            reset_region = {current_tick: reset_paulistring}

            resets[current_reset] = reset_region
            current_reset += 1

    return resets


def get_observing_region(
    circuit: stim.Circuit, observable: Collection[int] | int
) -> PauliRegion:
    """Returns the observing region of an observable in a circuit.

    Parameters
    ----------
    circuit
        Stim circuit with the observable definitions.
    observable
        List of observables in the circuit that define the observable whose
        observing region will be returned. It can also be a single observable.
        The observables are identified with their index and must be present
        in `circuit`.

    Returns
    -------
    obs_region
        Observing region of `observable` defined as a dictionary whose keys
        refer to the `TICK` indices in `circuit` and values correspond to the
        `stim.PauliString` at the corresponding `TICK` location. If a `stim.PauliString`
        is empty at the `TICK`, the corresponding `TICK` index will not be present
        in the dictionary. See `stim.Circuit.detecting_regions` for more information.
    """
    if not isinstance(circuit, stim.Circuit):
        raise TypeError(
            f"'circuit' must be stim.Circuit, but {type(circuit)} was given."
        )
    if isinstance(observable, int):
        observable = [observable]
    if not isinstance(observable, Collection):
        raise TypeError(
            f"'observable' must be a Collection, but {type(observable)} was given."
        )
    if any(not isinstance(o, int) for o in observable):
        raise TypeError("Elements in 'observable' must be integers.")
    if any(o >= circuit.num_observables for o in observable):
        raise TypeError(
            "Elements in 'observable' must be valid observable indices for 'circuit'."
        )

    # remove all observables from the circuit and define the given observable.
    # a trick is to not move the observable definitions and change its argument
    # to the same, so that definitions get XORed by stim.
    new_circuit = stim.Circuit()
    for instr in circuit.flattened():
        if instr.name != "OBSERVABLE_INCLUDE":
            new_circuit.append(instr)
            continue

        if instr.gate_args_copy()[0] not in observable:
            continue

        new_obs = stim.CircuitInstruction(
            name="OBSERVABLE_INCLUDE", gate_args=[0], targets=instr.targets_copy()
        )
        new_circuit.append(new_obs)

    l0_target = stim.DemTarget("L0")
    return new_circuit.detecting_regions(
        targets=[l0_target], ignore_anticommutation_errors=True
    )[l0_target]


def remove_obs_except(
    circuit: stim.Circuit, observables: Sequence[Collection[int]]
) -> stim.Circuit:
    """Removes all observables from the circuit except the specified ones.

    Parameters
    ----------
    circuit
        Circuit with observables.
    observables
        List of observables consisting of collections of observable indicies from
        the circuit.

    Returns
    -------
    new_circuit
        Circuit with only the observables in 'observables'.
    """
    if not isinstance(circuit, stim.Circuit):
        raise TypeError(
            f"'circuit' must be a stim.Circuit, but {type(circuit)} was given."
        )
    if not isinstance(observables, Sequence):
        raise TypeError(
            f"'observables' must be a Sequence, but {type(observables)} was given."
        )

    if any(not isinstance(o, Collection) for o in observables):
        raise TypeError("Elements in 'observables' must be Collection.")
    indices = [i for o in observables for i in o]
    if any(not isinstance(i, int) for i in indices):
        raise TypeError("Elements inside each element in 'observables' must be ints.")
    if max(indices) > circuit.num_observables - 1:
        raise ValueError("Index cannot be larger than 'circuit.num_observables-1'.")

    new_circuit = stim.Circuit()
    circuit_observables: list[stim.CircuitInstruction] = []
    # moving the definition of the observables messes with the rec[-i] definition
    # therefore I need to take care of how many measurements are between the definition
    # and the end of the circuit (where I am going to define the observables)
    measurements: list[int] = []
    for i, instr in enumerate(circuit.flattened()):
        if instr.name == "OBSERVABLE_INCLUDE":
            circuit_observables.append(instr)
            measurements.append(circuit[i:].num_measurements)
        else:
            new_circuit.append(instr)

    for k, observable in enumerate(observables):
        # use a set and symmetric_difference to remove repeated targets (mod 2)
        new_targets: set[int] = set()
        for obs_ind in observable:
            targets = circuit_observables[obs_ind].targets_copy()
            targets = [t.value - measurements[obs_ind] for t in targets]
            new_targets.symmetric_difference_update(targets)
        new_obs = stim.CircuitInstruction(
            "OBSERVABLE_INCLUDE", [stim.target_rec(t) for t in new_targets], [k]
        )
        new_circuit.append(new_obs)

    return new_circuit


def get_qubit_measurements(circuit: stim.Circuit) -> dict[int, dict[int, str]]:
    """Returns the basis of the measured qubits.

    Parameters
    ----------
    circuit
        Circuit with single-qubit measurements in the X or Z basis.
        No other type of measurements are allowed.

    Returns
    -------
    meas_after_tick
        Dictionary with keys corresponding to TICK indicies that contain measurements
        after them and with values corresponding a mapping of the qubits that are
        measured and the basis of their corresponding measurements (`"X"` or `"Z"`).
        Note that TICK indices start at 0.
    """
    if not isinstance(circuit, stim.Circuit):
        raise TypeError(
            f"'circuit' must be a stim.Circuit, but {type(circuit)} was given."
        )

    meas_after_tick: dict[int, dict[int, str]] = {}
    current_tick = -1  # the first TICK found is tick number 0
    for instr in circuit.flattened():
        if instr.name == "TICK":
            current_tick += 1
            continue
        if instr.name not in MEAS_INSTRS:
            continue
        if instr.name[-1] == "Y":
            raise TypeError("Only measurements in the X and Z basis are allowed.")

        qubit_inds = [t.value for t in instr.targets_copy()]
        meas_type = "Z"
        if instr.name[-1] == "X":
            meas_type = "X"

        if current_tick not in meas_after_tick:
            meas_after_tick[current_tick] = {}

        for qubit_ind in qubit_inds:
            meas_after_tick[current_tick][qubit_ind] = meas_type

    return meas_after_tick


def get_subgraph(
    unencoded_circuit: stim.Circuit,
    encoded_circuit: stim.Circuit,
    reliable_obs: Collection[int] | int,
    stab_coords: Sequence[dict[str, Collection[Coords]]],
) -> tuple[stim.DetectorErrorModel, npt.NDArray[np.int64]]:
    """Returns the (decomposed) detector error model and detector indices
    inside the given observable's observing region.

    Parameters
    ----------
    unencoded_circuit
        Unencoded (bare, logical) circuit. `TICK`s represent QEC rounds.
        Conditional gates based on outcomes are not supported. It must contain
        the same observable definitions as `encoded_circuit` (with the same
        observable indices). Only single-qubit measurement in the X and Z
        basis are allowed.
    encoded_circuit
        Encoded (physical) circuit. It must contain the detectors and
        observables used for decoding. The detectors must contain coordinates
        and their last element must be the index of the corresponding
        QEC round or `TICK` of `unencoded_circuit`. `TICK`s are not
        important in `encoded_circuit`. The detectors for the logical measurements
        must have their last element be equal to the index of the last `TICK`
        + 0.5. The QEC code must be CSS.
    reliable_obs
        Reliable observable corresponding to a collection of observable indices
        (or a single index) from `encoded_circuit` and `unencoded_circuit`.
    stab_coords
        Coordinates of the X and Z stabilizers defined in `encoded_circuit` for
        each of the (logical) qubits defined in `unencoded_circuit`. The `i`th
        element in the list must correspond to qubit index `i` in `unencoded_circuit`.
        Each element must be a dictionary with keys `"X"` and `"Z"`, and values
        corresponding to the ancilla coordinates of the specific stabilizer type.

    Returns
    -------
    dem
        Detector error model inside `reliable_obs` observing region.
        The DEM only contains the reliable observable.
    det_inds
        Detector indices inside `reliable_obs` observing region.
        The length of `det_inds` matches the number of detectors in `dem`.
    """
    if not isinstance(unencoded_circuit, stim.Circuit):
        raise TypeError(
            "'unencoded_circuit' must be a stim.Circuit, "
            f"but {type(unencoded_circuit)} was given."
        )
    if not isinstance(encoded_circuit, stim.Circuit):
        raise TypeError(
            "'encoded_circuit' must be a stim.Circuit, "
            f"but {type(encoded_circuit)} was given."
        )
    if unencoded_circuit.num_observables != encoded_circuit.num_observables:
        raise ValueError(
            "'unencoded_circuit' and 'encoded_circuit' must have the same observables."
        )
    num_obs = encoded_circuit.num_observables
    num_ticks = unencoded_circuit.num_ticks

    if isinstance(reliable_obs, int):
        reliable_obs = [reliable_obs]
    if not isinstance(reliable_obs, Collection):
        raise TypeError(
            "'reliable_obs' must be a Collection, but {type(reliable_obs)} was given."
        )
    if any(not (isinstance(o, int) and o < num_obs) for o in reliable_obs):
        raise TypeError(
            "Elements in 'reliable_obs' must be valid observable indices in 'encoded_circuit'."
        )

    if not isinstance(stab_coords, Sequence):
        raise TypeError(
            f"'stab_coords' must be a Sequence, but {type(stab_coords)} was given."
        )
    if len(stab_coords) != unencoded_circuit.num_qubits:
        raise ValueError(
            "Lenght of 'stab_coords' must match the number of qubits in 'unencoded_circuit'."
        )
    if any(not isinstance(l, dict) for l in stab_coords):
        raise TypeError("Elements of 'stab_coords' must be dictionaries.")
    if any(set(l.keys()) < set(["X", "Z"]) for l in stab_coords):
        raise ValueError(
            "Elements of 'stab_coords' must have 'X' and 'Z' as dict keys."
        )
    for stab_type in ["X", "Z"]:
        if any(not isinstance(l[stab_type], Collection) for l in stab_coords):
            raise TypeError(
                "Elements of 'stab_coords' must have collections as dict values."
            )
        for coord in chain(*[l[stab_type] for l in stab_coords]):
            if not isinstance(coord, tuple):
                raise TypeError("Coordinates must be tuples.")
            if any(not isinstance(i, (float, int)) for i in coord):
                raise TypeError("Coordinates must be tuple[float]")

    det_to_coords = encoded_circuit.get_detector_coordinates()
    if any(c == [] for c in det_to_coords.values()):
        raise ValueError("All detectors must have coordinates.")
    if any(len(c) < 2 for c in det_to_coords.values()):
        raise ValueError("All detectors must contain at least two coordinates.")
    # using int(...) to make all the half integers from the measurement be converted to measurements.
    # the tick coordinate starts at 0 (not at 1).
    if int(max([c[-1] for c in det_to_coords.values()])) + 1 != num_ticks:
        raise ValueError(
            "Number of TICKs in `unencoded_circuit` must match the detector coordinate"
            " for number of rounds"
        )

    z_stab_coords = set(chain(*[l["X"] for l in stab_coords]))
    x_stab_coords = set(chain(*[l["Z"] for l in stab_coords]))
    if len(z_stab_coords.intersection(x_stab_coords)) != 0:
        raise ValueError("Coordinate(s) appear in both X and Z type stabilizers.")
    all_stab_coords_circuit = [
        tuple(map(float, c[:-1])) for c in det_to_coords.values()
    ]
    all_stab_coords = z_stab_coords.union(x_stab_coords)
    all_stab_coords = [tuple(map(float, c[:-1])) for c in all_stab_coords]
    if set(all_stab_coords) < set(all_stab_coords_circuit):
        raise ValueError(
            "Not all detectors of 'encoded_circuit' are speficied in 'stab_coords'."
        )

    # create mapping for fast detector selection
    coords_to_stab: dict[Coords, tuple[int, str]] = {}
    for l_ind, l in enumerate(stab_coords):
        for stab_type in ["X", "Z"]:
            for coord in l[stab_type]:
                # ensure they are floats, because they can be integers
                coord = tuple(map(float, coord))
                coords_to_stab[coord] = (l_ind, stab_type)

    obs_region = get_observing_region(unencoded_circuit, reliable_obs)
    meas_after_tick = get_qubit_measurements(unencoded_circuit)

    sub_circuit = stim.Circuit()
    det_inds = []
    current_det = 0
    # need to create new observable corresponding to the reliable observable.
    # moving the definition of the observables messes with the rec[-i] definition
    # therefore I need to take care of how many measurements are between the definition
    # and the end of the circuit (where I am going to define the reliable observable)
    observables: list[stim.CircuitInstruction] = []
    measurements: list[int] = []

    for i, instr in enumerate(encoded_circuit.flattened()):
        if instr.name not in ["OBSERVABLE_INCLUDE", "DETECTOR"]:
            sub_circuit.append(instr)
            continue
        if instr.name == "OBSERVABLE_INCLUDE":
            observables.append(instr)
            measurements.append(encoded_circuit[i:].num_measurements)
            continue

        coord = list(map(float, instr.gate_args_copy()))
        coord, tick = tuple(coord[:-1]), coord[-1]
        qubit_ind, det_type = coords_to_stab[coord]

        if not tick.is_integer():
            # the detector comes from a logical measurement and the tick must be X.5
            if not (tick * 2).is_integer():
                raise ValueError(
                    "Last detector coordinate must be integer or half integer, "
                    f"but '{tick}' was found."
                )

            # if there is a measurement after the TICK in the unencoded circuit
            # that is applied to 'qubit_ind', this means that the logical obsevable
            # must include that measurement (because logical measurements are destructive).
            tick = int(tick)  # remove 0.5 from tick
            if tick not in meas_after_tick:
                if qubit_ind not in meas_after_tick[tick]:
                    raise ValueError(
                        "Found detector with a half integer in the last coordinate "
                        "that does not come from a logical measurement."
                    )

            meas_type = meas_after_tick[tick][qubit_ind]
            if meas_type == det_type:
                sub_circuit.append(instr)
                det_inds.append(current_det)
                current_det += 1

            continue

        # we know that the tick is an integer (coming from a QEC round)
        tick = int(tick)
        if tick not in obs_region:
            continue

        # stim uses encoding 0=I, 1=X, 2=Y, 3=Z.
        if (det_type == "Z") and (obs_region[tick][qubit_ind] in [3, 2]):
            sub_circuit.append(instr)
            det_inds.append(current_det)
        if (det_type == "X") and (obs_region[tick][qubit_ind] in [1, 2]):
            sub_circuit.append(instr)
            det_inds.append(current_det)

        current_det += 1

    # create the reliable observable definition
    new_targets: list[int] = []
    for obs_ind in reliable_obs:
        targets = observables[obs_ind].targets_copy()
        targets = [t.value - measurements[obs_ind] for t in targets]
        new_targets += targets
    new_obs = stim.CircuitInstruction(
        "OBSERVABLE_INCLUDE", [stim.target_rec(t) for t in new_targets], [0]
    )
    sub_circuit.append(new_obs)

    dem = sub_circuit.detector_error_model(decompose_errors=True)

    return dem, np.array(det_inds, dtype=int)
