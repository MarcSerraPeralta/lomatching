import numpy as np
import stim
from pymatching import Matching

from .greedy_algorithm import greedy_algorithm


class MoMatching:
    """
    Decodes all observables in a logical Clifford circuit run on a surface code
    with only reliable observables and qubit only measured at the end of the circuit.
    """

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        unencoded_circuit: stim.Circuit,
        logicals: list[list[str]],
        stab_coords: dict[str, list[tuple[float | int, float | int, float | int]]],
        detector_frame: str,
    ):
        """
        Initializes ``MoMatching``.

        Parameters
        ----------
        dem
            Detector error model.
            The ``(x, y, t)`` coordinates must be specified for every detector,
            with ``t`` an integer except representing the QEC round. For the
            detectors built from data-qubit outcomes (from logical measurements),
            they need to be placed at ``t+1/2``, with ``t`` the integer of the
            last QEC round. Gauge detectors in resets must also be specified.
        unencoded_circuit
            Unencoded (bare) circuit with only MZ, RZ, MX, RX, S, H, X, Z, Y, I, CNOT gates.
            Circuit must start with all qubits being reset and end with all qubits
            being measured. TICKs represent QEC cycles.
            Conditional gates based on outcomes are not allowed.
            Qubits can only perform a single operation inbetween QEC cycles.
            The next operation of a measurement must be a reset.
            It can be a ``stim.Circuit`` or a np.ndarray (see ``get_ops``).
        logicals
            Definition of the reliable logicals as done in the circuit.
            For example, if one has defined L0 = Z0*Z1, then ``logicals``
            should be ``[["Z0", "Z1"]]``. The logical ``Lk`` must be defined in
            the ``k``th entry of the ``logicals`` list.
        stab_coords
            Dictionary with keys corresponding to ``"Z0"``, ``"X0"``, ``"Z1"``,
            ``"X1"``... (with ``Zk`` referring to the Z-type stabilizers for the
            ``k``th logical qubit and similarly for ``Xk``) and the values being
            a list of the ``(x, y)`` coordinates of all corresponding stabilizers.
            The observable IDs must also match with the qubit indices from ``circuit``.
        detector_frame
            Frame used when defining the detectors. Must be either ``"pre-gate"``
            or ``"post-gate"``.

        Notes
        -----
        See example of usage in the ``README.md`` file.
        """
        det_to_coords = dem.get_detector_coordinates()
        if any(c == [] for c in det_to_coords.values()):
            raise ValueError("All detectors must have coordinates.")
        coords_to_det = {tuple(v): k for k, v in det_to_coords.items()}

        self.dem = dem
        self.unencoded_circuit = unencoded_circuit
        self.logicals = logicals
        self.stab_coords = stab_coords
        self.detector_frame = detector_frame
        self.det_to_coords = det_to_coords
        self.coords_to_det = coords_to_det
        self.detector_frame = detector_frame

        self.decoding_subgraphs = {}
        self.mwpm_subgraphs = {}

        self._prepare_decoder()

        return

    def _prepare_decoder(self):
        """
        Prepares all the variables required for running ``MoMatching.decode``
        and ``MoMatching.decode_batch``.
        """
        for k, logical in enumerate(self.logicals):
            tracks = greedy_algorithm(
                self.unencoded_circuit,
                detector_frame=self.detector_frame,
                r_start=999_999_999,
                t_start=get_initial_tracks(logical, self.unencoded_circuit.num_qubits),
            )
            self.decoding_subgraphs[k] = get_subgraph(
                self.dem, tracks, self.stab_coords, self.coords_to_det, k
            )
            self.mwpm_subgraphs[k] = Matching(self.decoding_subgraphs[k])

        return

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Decodes the syndrome of a single run of the logical circuit."""
        logical_correction = np.zeros(len(self.logicals))
        for k, _ in enumerate(self.logicals):
            prediction = self.mwpm_subgraphs[k].decode(syndrome)
            logical_correction[k] = prediction[k]
        return logical_correction

    def decode_batch(self, syndrome: np.ndarray) -> np.ndarray:
        """Decodes the syndrome of a multiple runs of the logical circuit."""
        logical_correction = np.zeros((len(syndrome), len(self.logicals)))
        for k, _ in enumerate(self.logicals):
            prediction = self.mwpm_subgraphs[k].decode_batch(syndrome)
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
    stab_coords: dict[str, list[tuple[float | int, float | int, float | int]]],
    coords_to_det: dict[tuple[float, float, float], int],
    logical_id: int,
) -> stim.DetectorErrorModel:
    """Returns the decoding subgraph for the specified logical."""
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
    # not be found" because inactive nodes are not connected
    all_nodes = set(range(dem.num_detectors))
    dets_no_track_1 = all_nodes.difference(dets_track_1)
    for det in dets_no_track_1:
        new_instr = stim.DemInstruction(
            "error", args=[0.5], targets=[stim.target_relative_detector_id(det)]
        )
        subdem.append(new_instr)

    return subdem
