from collections.abc import Sequence, Collection

import numpy as np
import numpy.typing as npt
import stim
from pymatching import Matching

from .util import get_detector_indices_for_subgraphs, get_circuit_subgraph, Coords


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
        self, syndrome: npt.NDArray[np.int64 | np.bool]
    ) -> npt.NDArray[np.bool]:
        """Decodes the given batch of syndromes and returns the corrections for the observables."""
        if len(syndrome.shape) != 2:
            raise TypeError(
                f"'syndrome' must be a matrix, but shape {syndrome.shape} was given."
            )
        if syndrome.shape[1] != self._num_dets:
            raise TypeError(
                "'syndrome.shape[1]' must match the number of detectors "
                f"({self._num_dets}), but {syndrome.shape[1]} was given."
            )

        obs_correction = np.zeros((len(syndrome), self._num_obs), dtype=bool)
        for k in range(self._num_obs):
            subsyndrome = syndrome[:, self._det_inds_subgraphs[k]]
            subcorrection = self._matching_subgraphs[k].decode_batch(subsyndrome)
            obs_correction[:, k] = subcorrection[:, k]
        return obs_correction
