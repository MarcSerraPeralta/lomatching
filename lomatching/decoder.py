from collections.abc import Sequence, Collection

import numpy as np
import numpy.typing as npt
import stim
from pymatching import Matching

from .util import get_reliable_observables as get_reliable_obs
from .util import get_subgraph, Coords


class MoMatching:
    """
    Decodes all reliable observables in an (unconditional) logical Clifford
    circuit with `pymatching.Matching`.
    """

    def __init__(
        self,
        unencoded_circuit: stim.Circuit,
        encoded_circuit: stim.Circuit,
        stab_coords: Sequence[dict[str, Collection[Coords]]],
    ):
        """
        Initializes a ``MoMatching`` decoder.

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
        stab_coords
            Coordinates of the X and Z stabilizers defined in `encoded_circuit` for
            each of the (logical) qubits defined in `unencoded_circuit`. The `i`th
            element in the list must correspond to qubit index `i` in `unencoded_circuit`.
            Each element must be a dictionary with keys `"X"` and `"Z"`, and values
            corresponding to the ancilla coordinates of the specific stabilizer type.

        Notes
        -----
        See example in the ``README.md`` file.
        """
        self._unencoded_circuit: stim.Circuit = unencoded_circuit
        self._encoded_circuit: stim.Circuit = encoded_circuit

        self._reliable_obs: list[set[int]] = get_reliable_obs(unencoded_circuit)
        self._encoded_circuit_with_only_reliable_obs: stim.Circuit = (
            remove_obs_except_reliables(encoded_circuit, self._reliable_obs)
        )

        self._matching_subgraphs: list[Matching] = []
        self._det_inds_subgraphs: list[npt.NDArray[np.int64]] = []
        for obs in self._reliable_obs:
            subgraph, det_inds = get_subgraph(
                unencoded_circuit, encoded_circuit, obs, stab_coords
            )
            self._matching_subgraphs.append(Matching(subgraph))
            self._det_inds_subgraphs.append(det_inds)

        return

    @property
    def reliable_observables(self) -> list[set[int]]:
        """Reliable observables in the circuit, where each one is given as a set
        integers corresponding to observable indicies from the circuit."""
        return [set(o) for o in self._reliable_obs]

    @property
    def encoded_circuit_with_only_reliable_observables(self) -> stim.Circuit:
        """This circuit is useful for sampling errors."""
        return self._encoded_circuit_with_only_reliable_obs.copy()

    @property
    def num_detectors(self) -> int:
        return self._encoded_circuit.num_detectors

    def decode(self, syndrome: npt.NDArray[np.int64 | np.bool]) -> npt.NDArray[np.bool]:
        """Decodes the given syndrome vector and returns the corrections for
        the reliable observables, see `MoMatching.reliable_observables`."""
        if len(syndrome.shape) != 1:
            raise TypeError(
                f"'syndrome' must be a vector, but shape {syndrome.shape} was given."
            )

        obs_corection = np.zeros(len(self.reliable_observables), dtype=bool)
        for k, _ in enumerate(self.reliable_observables):
            subsyndrome = syndrome[self._det_inds_subgraphs[k]]
            obs_corection[k] = self._matching_subgraphs[k].decode(subsyndrome)[0]
        return obs_corection

    def decode_batch(
        self, syndrome: npt.NDArray[np.int64 | np.bool]
    ) -> npt.NDArray[np.bool]:
        """Decodes the given batch of syndromes and returns the corrections for
        the reliable observables, see `MoMatching.reliable_observables`."""
        if len(syndrome.shape) != 2:
            raise TypeError(
                f"'syndrome' must be a matrix, but shape {syndrome.shape} was given."
            )
        if syndrome.shape[1] != self.num_detectors:
            raise TypeError(
                "'syndrome.shape[1]' must match the number of detectors "
                f"({self.num_detectors}), but {syndrome.shape[1]} was given."
            )

        obs_corection = np.zeros(
            (len(syndrome), len(self.reliable_observables)), dtype=bool
        )
        for k, _ in enumerate(self.reliable_observables):
            subsyndrome = syndrome[:, self._det_inds_subgraphs[k]]
            obs_corection[:, k] = self._matching_subgraphs[k].decode(subsyndrome)[:, 0]
        return obs_corection
