import numpy as np
import stim
from pymatching import Matching
import matplotlib.pyplot as plt

from .greedy_algorithm import (
    greedy_algorithm,
    plot_track,
    plot_time_hypergraph,
    get_time_hypergraph,
    get_ops,
)


class BatchSplitMatching:
    """
    Decodes a logical Clifford circuit run on unrotated surface codes in one go.
    The circuit must have all the measurements at the end.
    """

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        circuit: stim.Circuit,
        logicals: list[list[str]],
        stab_coords: dict[str, tuple[float | int, float | int, float | int]],
        detector_frame: str,
    ):
        """
        Initializes ``BatchSplitMatching``.

        Parameters
        ----------
        dem
            Detector error model.
        circuit
            Logical circuit with only MZ, RZ, MX, RX, S, H, X, Z, Y, I, CNOT gates.
            Circuit must start with all qubits being reset and end with all qubits
            being measured. TICKs represent QEC cycles.
            Conditional gates based on outcomes are not allowed.
            Qubits can only perform a single operation inbetween QEC cycles.
            The next operation of a measurement must be a reset.
            It can be a stim.Circuit or a np.ndarray (see ``get_ops``).
        logicals
            Definition of the logicals as done in the circuit.
            E.g. if one has defined L0 = Z0*Z1, then the ``logicals``
            should be ``[["Z0", "Z1"]]``. They must be ordered following
            the logical observable indices in the circuit.
        stab_coords
            Dictionary with keys corresponding to Z0, X0, Z1, X1... (?-stabs for each logical) in the
            detector error model and the keys being the coordinates of all stabilizers
            associated with that logical qubit.
            The observable IDs must also match with the qubit indeces from ``circuit``.
        """
        det_to_coords = dem.get_detector_coordinates()
        if any(c == [] for c in det_to_coords.values()):
            raise ValueError("All detectors must have coordinates.")
        coords_to_det = {tuple(v): k for k, v in det_to_coords.items()}

        self.dem = dem
        self.circuit = circuit
        self.logicals = logicals
        self.stab_coords = stab_coords
        self.detector_frame = detector_frame
        self.det_to_coords = det_to_coords
        self.coords_to_det = coords_to_det
        self.detector_frame = detector_frame

        self._prepare_decoder()

        return

    def _prepare_decoder(self):
        """
        Prepares all the variables required for running ``self.decode``
        and ``self.decode_batch``.
        """
        self.decoding_subgraphs = {}

        for k, logical in enumerate(self.logicals):
            tracks = greedy_algorithm(
                self.circuit,
                detector_frame=self.detector_frame,
                r_start=999_999_999,
                t_start=get_initial_tracks(logical, self.circuit.num_qubits),
            )
            # fig, ax = plt.subplots()
            # plot_time_hypergraph(ax, get_time_hypergraph(get_ops(self.circuit), self.detector_frame))
            # plot_track(ax, tracks, 1)
            # plt.show()
            self.decoding_subgraphs[k] = get_subgraph(
                self.dem, tracks, self.stab_coords, self.coords_to_det, k
            )

        return

    def decode(self, defects: np.ndarray) -> np.ndarray:
        logical_correction = np.zeros(len(self.logicals))
        for k, _ in enumerate(self.logicals):
            mwpm = Matching(self.decoding_subgraphs[k])
            prediction = mwpm.decode(defects)
            logical_correction[k] = prediction[k]
        return logical_correction

    def decode_batch(self, defects: np.ndarray) -> np.ndarray:
        logical_correction = np.zeros((len(defects), len(self.logicals)))
        for k, _ in enumerate(self.logicals):
            mwpm = Matching(self.decoding_subgraphs[k])
            prediction = mwpm.decode_batch(defects)
            logical_correction[:, k] = prediction[:, k]
        return logical_correction


def get_initial_tracks(logical: list[str], num_qubits: int) -> np.ndarray:
    """Returns initial track indices for ``greedy_algorithm``."""
    shift = {"X": 0, "Z": 1}
    t_start = [2] * (2 * num_qubits)
    for l in logical:
        index = 2 * int(l[1:]) + shift[l[0]]
        t_start[index] = 1
    return np.array(t_start)


def get_subgraph(
    dem: stim.DetectorErrorModel,
    tracks: np.ndarray,
    stab_coords: dict,
    coords_to_det: dict,
    logical_id: int,
) -> stim.DetectorErrorModel:
    dets_track_1 = []
    for t, slice in enumerate(tracks):
        if t == len(tracks) - 1:
            # logical measurements
            t -= 0.5

        for k, s in enumerate(slice):
            if s == 1:
                # track 1
                prefix = "Z" if k % 2 == 1 else "X"
                label = f"{prefix}{k//2}"
                dets_track_1 += [
                    coords_to_det[(*list(map(float, xy)), float(t))]
                    for xy in stab_coords[label]
                ]
    dets_track_1 = set(dets_track_1)

    subdem = stim.DetectorErrorModel()
    for dem_instr in dem.flattened():
        if dem_instr.type != "error":
            subdem.append(dem_instr)
            continue

        det_ids = set(
            i.val for i in dem_instr.targets_copy() if i.is_relative_detector_id()
        )
        subdet_ids = det_ids.intersection(dets_track_1)
        if len(subdet_ids) == 0:
            continue

        log_ids = set(
            i.val for i in dem_instr.targets_copy() if i.is_logical_observable_id()
        )
        sublog_ids = set([logical_id]) if logical_id in log_ids else set()

        targets = [stim.target_relative_detector_id(d) for d in subdet_ids]
        targets += [stim.target_logical_observable_id(l) for l in sublog_ids]

        new_instr = stim.DemInstruction(
            "error", args=dem_instr.args_copy(), targets=targets
        )
        subdem.append(new_instr)

    # this is just for pymatching to not complain about "no perfect matching could
    # not be found" because some nodes are not connected
    all_nodes = set(range(dem.num_detectors))
    dets_no_track_1 = all_nodes.difference(dets_track_1)
    for det in dets_no_track_1:
        new_instr = stim.DemInstruction(
            "error", args=[0.5], targets=[stim.target_relative_detector_id(det)]
        )
        subdem.append(new_instr)

    return subdem
