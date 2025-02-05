import numpy as np
import stim
from pymatching import Matching
from ldpc.bp_decoder import BpDecoder

from .greedy_algorithm import greedy_algorithm
from .util import (
    comb_probs_to_w,
    dem_to_hplc,
    dem_only_errors,
    valid_decomposition,
    get_edges_dict,
    standardize_edge,
)


class BeliefSoMatching:
    """
    Decodes the observables (from a logical measurement) in a logical Clifford circuit
    run on a surface code. It runs belief-propagation on the full hypergraph
    to update the probabilities of the subgraphs.
    """

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        circuit: stim.Circuit,
        logicals: list[list[str]],
        stab_coords: dict[str, tuple[float | int, float | int, float | int]],
        detector_frame: str,
        ignore_decomposition_failures: bool = False,
        max_iter: int = 20,
        bp_method: str = "product_sum",
        **kargs_bp,
    ):
        """
        Initializes ``BeliefSoMatching``.

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
            It can be a ``stim.Circuit`` or a ``np.ndarray``
            (see ``somatching.greedy_algorithm.get_ops``).
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
        detector_frame
            Frame used when defining the detectors. Must be either ``"pre-gate"``
            or ``"post-gate"``.
        ignore_decomposition_failures
            Ignore hyperedge decomposition failures when building the DEM subgraphs
            for ``pymatching.Matching``. By default ``False``. If the circuit distance
            is lower than the distance of the surface code(s), set to ``True``.
        max_iter
            Maximum number of iterations for belief propagation.
        bp_method
            Belief-propagation method, for more information see ``ldpc.bp_decoder.BpDecoder``.
        kargs_bp
            Extra arguments for ``ldpc.bp_decoder.BpDecoder``.
        """
        det_to_coords = dem.get_detector_coordinates()
        if any(c == [] for c in det_to_coords.values()):
            raise ValueError("All detectors must have coordinates.")
        coords_to_det = {tuple(v): k for k, v in det_to_coords.items()}

        self.dem = dem
        self.dem_errors = dem_only_errors(self.dem)

        # for generating the subgraphs
        self.circuit = circuit
        self.logicals = logicals
        self.stab_coords = stab_coords
        self.detector_frame = detector_frame
        self.det_to_coords = det_to_coords
        self.coords_to_det = coords_to_det
        self.detector_frame = detector_frame

        # for BP
        self.h, self.p, self.l, _ = dem_to_hplc(dem)
        self.bp_decoder = BpDecoder(
            self.h,
            error_channel=self.p,
            input_vector_type="syndrome",
            max_iter=max_iter,
            bp_method=bp_method,
            **kargs_bp,
        )

        # for running Matching in the subgraphs
        self.ignore_decomposition_failures = ignore_decomposition_failures
        self.h_sub = []
        self.l_sub = []
        self.e_sub_to_h_supp = []

        self._prepare_decoder()

        return

    def _prepare_decoder(self):
        """
        Prepares all the variables required for running ``self.decode``
        and ``self.decode_batch``.
        """
        self.h_sub = []
        self.l_sub = []
        self.e_sub_to_h_supp = []

        for k, logical in enumerate(self.logicals):
            tracks = greedy_algorithm(
                self.circuit,
                detector_frame=self.detector_frame,
                r_start=999_999_999,
                t_start=get_initial_tracks(logical, self.circuit.num_qubits),
            )
            active_dets, inactive_dets = get_active_inactive_dets(
                tracks, stab_coords=self.stab_coords, coords_to_det=self.coords_to_det
            )
            active_e_sub, active_h_sub = split_active_errors(
                self.dem_errors, active_dets, log_obs_id=k
            )
            # this is just for pymatching to not complain about "no perfect matching could
            # not be found" because some nodes are not connected
            for det in inactive_dets:
                instr = stim.DemInstruction(
                    "error", args=[0.5], targets=[stim.target_relative_detector_id(det)]
                )
                active_e_sub.append((999_999_999, instr))

            primitive_dem = get_primitive_dem(active_e_sub)
            h_sub_decom = get_hyperedge_decomposition(
                primitive_dem,
                active_h_sub,
                ignore_decomposition_failures=self.ignore_decomposition_failures,
            )
            e_sub_to_h_supp = get_e_sub_to_h_supp(active_e_sub, h_sub_decom)
            h_sub, _, l_sub, _ = dem_to_hplc(primitive_dem)

            self.h_sub.append(h_sub)
            self.l_sub.append(l_sub)
            self.e_sub_to_h_supp.append(e_sub_to_h_supp)

        return

    def decode(self, defects: np.ndarray) -> np.ndarray:
        corr = self.bp_decoder.decode(defects)
        if self.bp_decoder.converge:
            return (self.l @ corr) % 2

        llrs = self.bp_decoder.log_prob_ratios
        if np.isnan(llrs).any():
            raise ValueError("Returned `log_prob_ratios` from BP are NaN.")
        p_h = 1 / (1 + np.exp(llrs))

        logical_correction = np.zeros(len(self.logicals))
        for k, _ in enumerate(self.logicals):
            mwpm = Matching.from_check_matrix(
                check_matrix=self.h_sub[k],
                weights=[
                    comb_probs_to_w(p_h[sup]) if 999_999_999 not in sup else 0.5
                    for sup in self.e_sub_to_h_supp[k]
                ],
                faults_matrix=self.l_sub[k],
                use_virtual_boundary_node=True,
            )
            prediction = mwpm.decode(defects)
            logical_correction[k] = prediction[k]

        return logical_correction

    def decode_batch(self, defects: np.ndarray) -> np.ndarray:
        logical_correction = np.zeros((len(defects), len(self.logicals)), dtype=bool)
        for i in range(len(defects)):
            logical_correction[i] = self.decode(defects[i])
        return logical_correction


def get_initial_tracks(logical: list[str], num_qubits: int) -> np.ndarray:
    """Returns initial track indices for ``greedy_algorithm``."""
    shift = {"X": 0, "Z": 1}
    t_start = [2] * (2 * num_qubits)
    for l in logical:
        index = 2 * int(l[1:]) + shift[l[0]]
        t_start[index] = 1
    return np.array(t_start)


def get_active_inactive_dets(
    tracks: np.ndarray,
    stab_coords: dict,
    coords_to_det: dict,
) -> tuple[set[int], set[int]]:
    dets_track_1 = []
    for t, slice in enumerate(tracks):
        if t == len(tracks) - 1:
            # logical measurement at the end
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
    other_dets = set(range(len(coords_to_det))).difference(dets_track_1)

    return dets_track_1, other_dets


def split_active_errors(
    dem_errors: stim.DetectorErrorModel,
    active_dets: set[int],
    log_obs_id: int,
) -> tuple[
    list[tuple[int, stim.DemInstruction]], list[tuple[int, stim.DemInstruction]]
]:
    active_edges = []
    active_hyperedges = []
    for ind, dem_instr in enumerate(dem_errors):
        det_ids = set(
            i.val for i in dem_instr.targets_copy() if i.is_relative_detector_id()
        )
        log_ids = set(
            i.val for i in dem_instr.targets_copy() if i.is_logical_observable_id()
        )
        active_dets_error = det_ids.intersection(active_dets)
        active_logs_error = log_ids.intersection([log_obs_id])
        targets = [stim.target_relative_detector_id(d) for d in active_dets_error]
        targets += [stim.target_logical_observable_id(d) for d in active_logs_error]
        dem_instr_sub = stim.DemInstruction(
            type="error",
            targets=targets,
            args=dem_instr.args_copy(),
        )

        if len(active_dets_error) == 0:
            continue  # inactive error
        elif len(active_dets_error) <= 2:
            active_edges.append((ind, dem_instr_sub))
        else:
            active_hyperedges.append((ind, dem_instr_sub))

    return active_edges, active_hyperedges


def get_primitive_dem(
    active_edges: list[tuple[int, stim.DemInstruction]],
) -> stim.DetectorErrorModel:
    primitive_dem = stim.DetectorErrorModel()
    for _, instr in active_edges:
        primitive_dem.append(instr)
    return primitive_dem


def get_hyperedge_decomposition(
    primitive_dem: stim.DetectorErrorModel,
    h_sub: list[tuple[int, stim.DemInstruction]],
    ignore_decomposition_failures: bool = False,
) -> dict[int, list[int]]:
    """Returns {h_ind: list[e_sub_ind]} with e_sub_ind given by index in primitive dem,
    and h_ind given by index in dem_errors.
    """
    decomposition = {}
    mwpm = Matching(primitive_dem)
    num_dets = primitive_dem.num_detectors
    edges_dict = get_edges_dict(
        primitive_dem, ignore_decomposition_failures=ignore_decomposition_failures
    )

    for h_ind, dem_instr in h_sub:
        det_inds = np.array(
            [i.val for i in dem_instr.targets_copy() if i.is_relative_detector_id()],
            dtype=int,
        )
        defects = np.zeros(num_dets, dtype=bool)
        defects[det_inds] = True
        edges = mwpm.decode_to_edges_array(defects)
        edges_inds = [edges_dict[standardize_edge(edge)] for edge in edges]
        edges_instr = [primitive_dem[i] for i in edges_inds]
        if (not valid_decomposition(dem_instr, edges_instr)) and (
            not ignore_decomposition_failures
        ):
            raise ValueError(
                f"Invalid decomposition found:\n{dem_instr}\n{edges_instr}"
            )
        decomposition[h_ind] = edges_inds

    return decomposition


def get_e_sub_to_h_supp(
    active_e_sub: list[tuple[int, stim.DemInstruction]],
    h_sub_decom: dict[int, list[int]],
) -> list[np.ndarray]:
    """Returns list where the index i corresponds to e_sub[i] or primitive_dem[i]
    and the corresponding array gives the indices of the errors in dem_errors
    whose probabilities need to be combined to obtain the probabilitiy of e_sub[i].

    Note that h_sub_decom has the decomposition of h_sub in terms of the edges
    in primitive_dem, which does not correspond to the same indices in dem_errors.
    """
    e_sub_to_h_supp = [[] for _ in active_e_sub]
    # add edges support on themselves
    for e_sub_ind, (e_ind, _) in enumerate(active_e_sub):
        e_sub_to_h_supp[e_sub_ind].append(e_ind)

    # add support of the hyperedges on the edges
    for h_ind, decom_sub in h_sub_decom.items():
        for e_sub_ind in decom_sub:
            e_sub_to_h_supp[e_sub_ind].append(h_ind)

    e_sub_to_h_supp = [np.array(sup, dtype=int) for sup in e_sub_to_h_supp]

    return e_sub_to_h_supp
