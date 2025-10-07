from collections.abc import Collection, Sequence

from itertools import chain
import numpy as np
import numpy.typing as npt
import stim

Coords = tuple[float, ...]


def get_reliable_observables(circuit: stim.Circuit) -> list[set[int]]:
    """
    Note: this function assumes that, if a logical qubit is being measured or
    reset between QEC rounds, then the logical measurement or logical reset
    is the only operation it is doing between the QEC rounds. For example, the
    following circuits are not allowed: `TICK - R 0 - H 0 - TICK` and
    `TICK - M 0 - R 0 - TICK`. The corresponding valid circuits would be
    `TICK - R 0 - TICK - H 0 - TICK` and `TICK - M 0 - TICK - R 0 - TICK`.

    THAT IT IS NOT NECESSARY IF THE ENCODED CIRCUIT IS GIVEN!!!

    1) get obsrving regions (using stim.Circuit.detecting_regions)
    2) get "reset stim.Pauli"
    3) apply algorithm from our paper to get the fragile and reliable observables
    """
    return


def get_observing_region(
    circuit: stim.Circuit, observable: Collection[int] | int
) -> dict[int, stim.PauliString]:
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
    return new_circuit.detecting_regions(targets=[l0_target])[l0_target]


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
        observable indices).
    encoded_circuit
        Encoded (physical) circuit. It must contain the detectors and
        observables used for decoding. The detectors must contain coordinates
        and their last element must be the index of the corresponding
        QEC round or `TICK` of `unencoded_circuit`. `TICK`s are not
        important in `encoded_circuit`. The QEC code must be CSS.
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
    num_dets = encoded_circuit.num_detectors

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
            if any(not isinstance(i, float) for i in coord):
                raise TypeError("Coordinates must be tuple[float]")

    det_to_coords = encoded_circuit.get_detector_coordinates()
    if any(c == [] for c in det_to_coords.values()):
        raise ValueError("All detectors must have coordinates.")
    if any(len(c) < 2 for c in det_to_coords.values()):
        raise ValueError("All detectors must contain at least two coordinates.")
    if any(not c[-1].is_integer() for c in det_to_coords.values()):
        raise ValueError("Last coordinate of all detectors must be an integer.")
    if max([c[-1] for c in det_to_coords.values()]) != num_dets:
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
    if set(all_stab_coords) < set(all_stab_coords_circuit):
        raise ValueError(
            "Not all detectors of 'encoded_circuit' are speficied in 'stab_coords'."
        )

    # create mapping for fast detector selection
    coords_to_stab: dict[Coords, tuple[int, str]] = {}
    for l_ind, l in enumerate(stab_coords):
        for stab_type in ["X", "Z"]:
            for coord in l[stab_type]:
                coords_to_stab[coord] = (l_ind, stab_type)

    obs_region = get_observing_region(unencoded_circuit, reliable_obs)

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
        coord, tick = tuple(coord[:-1]), int(coord[-1])
        if tick not in obs_region:
            continue

        qubit_ind, det_type = coords_to_stab[coord]
        # stim uses encoding 0=I, 1=X, 2=Y, 3=Z.
        if (det_type == "Z") and (obs_region[tick][qubit_ind] in [3, 2]):
            sub_circuit.append(instr)
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
