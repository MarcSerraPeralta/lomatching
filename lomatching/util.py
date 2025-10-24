from collections.abc import Collection, Sequence

from itertools import chain
import numpy as np
import numpy.typing as npt
from galois import GF2
import stim

Coords = tuple[float, ...]
PauliRegion = dict[int, stim.PauliString]

RESET_INSTRS = ["R", "RX", "RY", "RZ", "MR", "MRX", "MRY", "MRZ"]


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
    circuit = circuit.flattened()

    resets: dict[int, PauliRegion] = {}
    current_reset = 0
    current_tick = 0
    for instr in circuit:
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
    circuit = circuit.flattened()

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
    for i, instr in enumerate(circuit):
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


def get_detector_indices_for_subgraphs(
    dem: stim.DetectorErrorModel,
    stab_coords: Sequence[dict[str, Collection[Coords]]],
) -> list[npt.NDArray[np.int64]]:
    """Returns a the detector indices for each of the observing regions in
    the given detector error model.

    Parameters
    ----------
    dem
        Detector error model.
        All detectors must have coordinates defined, with the last coordinate element
        representing time (or number of ticks).
    stab_coords
        Coordinates of the X and Z stabilizers defined in `encoded_circuit` for
        each of the (logical) qubits defined in `unencoded_circuit`. The `i`th
        element in the list must correspond to qubit index `i` in `unencoded_circuit`.
        Each element must be a dictionary with keys `"X"` and `"Z"`, and values
        corresponding to the ancilla coordinates of the specific stabilizer type.

    Returns
    -------
    det_inds
        Detector indices inside each of the observing regions.
        The length of `det_inds` matches the number of observables in ``dem``
        and they are sorted following the observable indices.
    """
    if not isinstance(dem, stim.DetectorErrorModel):
        raise TypeError(f"'dem' must be a stim DEM, but {type(dem)} was given.")
    dem = dem.flattened()

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
                raise TypeError("Coordinates must be tuple[float].")

    det_to_coords = dem.get_detector_coordinates()
    if any(c == [] for c in det_to_coords.values()):
        raise ValueError("All detectors must have coordinates.")
    det_to_coords = {d: tuple(map(float, c)) for d, c in det_to_coords.items()}
    coords_to_det = {c: d for d, c in det_to_coords.items()}

    # create mapping for fast detector selection
    coords_to_stab: dict[Coords, tuple[int, str]] = {}
    for l_ind, l in enumerate(stab_coords):
        for stab_type in ["X", "Z"]:
            for coord in l[stab_type]:
                # ensure they are floats, because they can be integers
                coord = tuple(map(float, coord))
                coords_to_stab[coord] = (l_ind, stab_type)

    # get boundary edges that flip the observables
    bd_edges_obs = {l: [] for l in range(dem.num_observables)}
    for instr in dem:
        if instr.type != "error":
            continue

        dets = [t.val for t in instr.targets_copy() if t.is_relative_detector_id()]
        if len(dets) != 1:
            continue

        obs = [t.val for t in instr.targets_copy() if t.is_logical_observable_id()]
        for o in obs:
            bd_edges_obs[o] += dets

    # identify (logical qubit, stabilizer type, time) for each boundary edge
    lst_obs = {l: set() for l in range(dem.num_observables)}
    for obs, dets in bd_edges_obs.items():
        for det in dets:
            coords = det_to_coords[det]
            l_ind, stab = coords_to_stab[coords[:-1]]
            time = coords[-1]
            lst_obs[obs].add((l_ind, stab, time))

    # get all detectors for the given (logical qubit, stability type, time)
    det_inds = []
    for obs in lst_obs:
        inds = []
        for l_ind, stab, time in lst_obs[obs]:
            coords = [(*c, time) for c in stab_coords[l_ind][stab]]
            inds += [coords_to_det[c] for c in coords]
        # indices need to be sorted to match the indices order in the dem!
        det_inds.append(np.array(sorted(inds), dtype=int))

    return det_inds


def get_circuit_subgraph(
    circuit: stim.Circuit, det_inds: Collection[int]
) -> stim.Circuit:
    """Returns the given circuit but with only the specified detectors."""
    if not isinstance(circuit, stim.Circuit):
        raise TypeError(
            f"'circuit' must be a stim.Circuit, but {type(circuit)} was given."
        )
    circuit = circuit.flattened()
    if not isinstance(det_inds, Collection):
        raise TypeError(
            f"'det_inds' must be a collection, but {type(det_inds)} was given."
        )
    if any(not isinstance(d, (int, np.int64)) for d in det_inds):
        raise TypeError("Elements in 'det_inds' must be integers.")

    new_circuit = stim.Circuit()
    curr_det_ind = 0
    for instr in circuit:
        if instr.name != "DETECTOR":
            new_circuit.append(instr)
            continue

        if curr_det_ind in det_inds:
            new_circuit.append(instr)

        curr_det_ind += 1

    return new_circuit
