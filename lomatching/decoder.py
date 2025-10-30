from collections.abc import Sequence, Collection

from itertools import chain
import numpy as np
import numpy.typing as npt
import stim
from pymatching import Matching
from ldpc import BpDecoder
from scipy.sparse import spmatrix

from .util import (
    get_detector_indices_for_subgraphs,
    get_circuit_subgraph,
    Coords,
    combine_probs,
    convert_to_numba_type,
    dem_to_hpl_list,
    dem_to_hld_graph,
    _list_to_csc_matrix,
)


class MoMatching:
    """
    Decodes all observables in an (unconditional) logical transversal-Clifford
    circuit with ``pymatching.Matching``.
    """

    def __init__(
        self,
        encoded_circuit: stim.Circuit,
        stab_coords: Sequence[dict[str, Collection[Coords]]],
        allow_gauge_detectors: bool = False,
    ):
        """
        Initializes a ``MoMatching`` decoder.

        Parameters
        ----------
        encoded_circuit
            Encoded (physical) circuit. It must contain the detectors and
            observables used for decoding. The detectors must contain coordinates
            and their last element must be the index of the corresponding
            QEC round or time. The defined observables must be reliable, see
            ``lomatching.get_reliable_observables`` and
            ``lomatching.remove_obs_except``. The QEC code must be CSS and have
            boundaries were the the logical Paulis terminate.
        stab_coords
            Coordinates of the X and Z stabilizers defined in `encoded_circuit` for
            each of the (logical) qubits. The ``i``th element in the list must correspond
            to logical qubit index ``i``. Each element must be a dictionary with keys
            ``"X"`` and ``"Z"``, and values corresponding to the ancilla coordinates of
            the specific stabilizer type.
        allow_gauge_detectors
            Allow gauge detectors when calling ``stim.Circuit.detector_error_model``.

        Notes
        -----
        See example in the ``README.md`` file.
        """
        if not isinstance(encoded_circuit, stim.Circuit):
            raise TypeError(
                "'encoded_circuit' must be a stim.Circuit, "
                f"but {type(encoded_circuit)} was given."
            )
        self._encoded_circuit: stim.Circuit = encoded_circuit
        self._num_obs: int = encoded_circuit.num_observables
        self._num_dets: int = encoded_circuit.num_detectors

        self._dem_subgraphs: list[stim.DetectorErrorModel] = []
        self._matching_subgraphs: list[Matching] = []
        self._det_inds_subgraphs: list[npt.NDArray[np.int64]] = []

        self._dem: stim.DetectorErrorModel = self._encoded_circuit.detector_error_model(
            allow_gauge_detectors=allow_gauge_detectors
        )

        self._det_inds_subgraphs = get_detector_indices_for_subgraphs(
            self._dem, stab_coords
        )

        for obs in range(self._num_obs):
            subcircuit = get_circuit_subgraph(
                encoded_circuit, self._det_inds_subgraphs[obs]
            )
            subgraph = subcircuit.detector_error_model(
                decompose_errors=True,
                allow_gauge_detectors=allow_gauge_detectors,
            )
            self._dem_subgraphs.append(subgraph)
            self._matching_subgraphs.append(Matching(subgraph))

        return

    @property
    def dem(self):
        return self._dem.copy()

    def decode(self, syndrome: npt.NDArray[np.int64 | np.bool]) -> npt.NDArray[np.bool]:
        """Decodes the given syndrome vector and returns the corrections for the observables."""
        if len(syndrome.shape) != 1:
            raise TypeError(
                f"'syndrome' must be a vector, but shape {syndrome.shape} was given."
            )

        obs_correction = np.zeros(self._num_obs, dtype=bool)
        for k in range(self._num_obs):
            subsyndrome = syndrome[self._det_inds_subgraphs[k]]
            obs_correction[k] = self._matching_subgraphs[k].decode(subsyndrome)[k]
        return obs_correction

    def decode_batch(
        self, syndromes: npt.NDArray[np.int64 | np.bool]
    ) -> npt.NDArray[np.bool]:
        """Decodes the given batch of syndromes and returns the corrections for the observables."""
        if len(syndromes.shape) != 2:
            raise TypeError(
                f"'syndromes' must be a matrix, but shape {syndromes.shape} was given."
            )
        if syndromes.shape[1] != self._num_dets:
            raise TypeError(
                "'syndromes.shape[1]' must match the number of detectors "
                f"({self._num_dets}), but {syndromes.shape[1]} was given."
            )

        obs_correction = np.zeros((len(syndromes), self._num_obs), dtype=bool)
        for k in range(self._num_obs):
            subsyndrome = syndromes[:, self._det_inds_subgraphs[k]]
            subcorrection = self._matching_subgraphs[k].decode_batch(subsyndrome)
            obs_correction[:, k] = subcorrection[:, k]
        return obs_correction


class BeliefMoMatching:
    """
    Decodes all observables in an (unconditional) logical transversal-Clifford
    circuit with ``ldpc.BpDecoder`` and ``pymatching.Matching``.
    """

    def __init__(
        self,
        encoded_circuit: stim.Circuit,
        stab_coords: Sequence[dict[str, Collection[Coords]]],
        allow_gauge_detectors: bool = False,
        **kargs_bp,
    ):
        """
        Initializes a ``BeliefMoMatching`` decoder.

        Parameters
        ----------
        encoded_circuit
            Encoded (physical) circuit. It must contain the detectors and
            observables used for decoding. The detectors must contain coordinates
            and their last element must be the index of the corresponding
            QEC round or time. The defined observables must be reliable, see
            ``lomatching.get_reliable_observables`` and
            ``lomatching.remove_obs_except``. The QEC code must be CSS and have
            boundaries were the the logical Paulis terminate.
        stab_coords
            Coordinates of the X and Z stabilizers defined in `encoded_circuit` for
            each of the (logical) qubits. The ``i``th element in the list must correspond
            to logical qubit index ``i``. Each element must be a dictionary with keys
            ``"X"`` and ``"Z"``, and values corresponding to the ancilla coordinates of
            the specific stabilizer type.
        allow_gauge_detectors
            Allow gauge detectors when calling ``stim.Circuit.detector_error_model``.
        **kargs_bp
            Extra arguments for ``ldpc.BpDecoder``.

        Notes
        -----
        See example in the ``README.md`` file.
        """
        if not isinstance(encoded_circuit, stim.Circuit):
            raise TypeError(
                "'encoded_circuit' must be a stim.Circuit, "
                f"but {type(encoded_circuit)} was given."
            )
        self._encoded_circuit: stim.Circuit = encoded_circuit
        self._num_obs: int = encoded_circuit.num_observables
        self._num_dets: int = encoded_circuit.num_detectors

        self._graphdems_h: list[spmatrix] = []
        self._graphdems_l: list[spmatrix] = []
        self._probs_indices: list[list[list[int]]] = []
        self._det_inds_subgraphs: list[npt.NDArray[np.int64]] = []

        self._dem: stim.DetectorErrorModel = self._encoded_circuit.detector_error_model(
            allow_gauge_detectors=allow_gauge_detectors
        )
        h, p, l = dem_to_hpl_list(self._dem)
        self._hyperdem_h: spmatrix = _list_to_csc_matrix(
            h, shape=(self._num_dets, len(h))
        )
        self._hyperdem_l: spmatrix = _list_to_csc_matrix(
            l, shape=(self._num_obs, len(l))
        )
        self._hyperdem_p: npt.NDArray[np.float64] = np.array(p, dtype=float)

        self._bp_decoder: BpDecoder = BpDecoder(
            self._hyperdem_h,
            error_channel=self._hyperdem_p,
            **kargs_bp,
        )

        self._det_inds_subgraphs = get_detector_indices_for_subgraphs(
            self._dem, stab_coords
        )

        for obs in range(self._num_obs):
            subcircuit = get_circuit_subgraph(
                encoded_circuit,
                self._det_inds_subgraphs[obs],
                obs_inds=[obs],
                keep_detector_definitions=True,
            )
            subgraph = subcircuit.detector_error_model(
                decompose_errors=True,
                allow_gauge_detectors=allow_gauge_detectors,
            )

            # find duplicates of errors in hyperdem_h when restricting to observing region
            duplicates = {}
            det_inds_subgraph = set(self._det_inds_subgraphs[obs].tolist())
            for err_id, dets in enumerate(h):
                dets_o = frozenset(dets).intersection(det_inds_subgraph)
                if dets_o not in duplicates:
                    duplicates[dets_o] = [err_id]
                else:
                    duplicates[dets_o].append(err_id)

            # get inputs for Matching
            h_graph, l_graph, edge_support = dem_to_hld_graph(
                subgraph, self._det_inds_subgraphs[obs]
            )

            # edge_support: from g to h_o
            # map: from h_o to hyperdem_h
            g_to_hyperdem_h = [
                list(chain(*[duplicates[h_o] for h_o in h_os])) for h_os in edge_support
            ]

            self._probs_indices.append(g_to_hyperdem_h)
            self._graphdems_h.append(h_graph)
            self._graphdems_l.append(l_graph)

        return

    @property
    def dem(self):
        return self._dem.copy()

    def decode(self, syndrome: npt.NDArray[np.int64 | np.bool]) -> npt.NDArray[np.bool]:
        """Decodes the given syndrome vector and returns the corrections for the observables."""
        if len(syndrome.shape) != 1:
            raise TypeError(
                f"'syndrome' must be a vector, but shape {syndrome.shape} was given."
            )

        corr = self._bp_decoder.decode(syndrome)
        if self._bp_decoder.converge:
            return (self._hyperdem_l @ corr) % 2
        llrs = self._bp_decoder.log_prob_ratios
        ps_h = 1 / (1 + np.exp(llrs))

        obs_correction = np.zeros(self._num_obs, dtype=bool)
        for k in range(self._num_obs):
            ps_e = combine_probs(ps_h, convert_to_numba_type(self._probs_indices[k]))
            matching = Matching.from_check_matrix(
                self._graphdems_h[k],
                weights=-np.log(ps_e),
                faults_matrix=self._graphdems_l[k],
                use_virtual_boundary_node=True,
            )
            subsyndrome = syndrome[self._det_inds_subgraphs[k]]
            obs_correction[k] = matching.decode(subsyndrome)[k]
        return obs_correction

    def decode_batch(
        self, syndromes: npt.NDArray[np.int64 | np.bool]
    ) -> npt.NDArray[np.bool]:
        """Decodes the given batch of syndromes and returns the corrections for the observables."""
        if len(syndromes.shape) != 2:
            raise TypeError(
                f"'syndromes' must be a matrix, but shape {syndromes.shape} was given."
            )
        if syndromes.shape[1] != self._num_dets:
            raise TypeError(
                "'syndromes.shape[1]' must match the number of detectors "
                f"({self._num_dets}), but {syndromes.shape[1]} was given."
            )

        obs_correction = np.zeros((len(syndromes), self._num_obs), dtype=bool)
        for k, syndrome in enumerate(syndromes):
            obs_correction[k] = self.decode(syndrome)
        return obs_correction
