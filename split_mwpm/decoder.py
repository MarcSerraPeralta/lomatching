from collections.abc import Collection

import stim
import numpy as np
from pymatching import Matching

from qec_util.dem_instrs import get_detectors, get_logicals, xor_probs


class SplitMatching:
    """
    Decoder for "matchable" codes and circuits involving logical gates
    that keep the Z-type stabilizer generators unchanged, and whose
    detectors are built in the "r" or "r-1" frames.
    """

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        z_coords: Collection[Collection[int | float]],
        x_coords: Collection[Collection[int | float]],
    ) -> None:
        """Initializes ``SplitMatching``.

        Parameters
        ----------
        dem
            DEM containing coordinates for all detectors.
        x_coords
            List of the coordinates associated with all the X-type detectors
            in ``dem``.
        z_coords
            List of the coordinates associated with all the Z-type detectors
            in ``dem``.
        """
        if not isinstance(dem, stim.DetectorErrorModel):
            raise TypeError(
                f"'dem' is not a stim.DetectorErrorModel, but a {type(dem)}."
            )
        self.dem = dem.flattened()
        self.num_dets = dem.num_detectors
        self.num_logs = dem.num_observables

        if not isinstance(z_coords, Collection):
            raise TypeError(
                f"'z_coords' must be a collection, but a {type(z_coords)} was given."
            )
        if not isinstance(x_coords, Collection):
            raise TypeError(
                f"'x_coords' must be a collection, but a {type(x_coords)} was given."
            )
        if any(not isinstance(c, Collection) for c in z_coords):
            raise TypeError("Elements in 'z_coords' must be collections.")
        if any(not isinstance(c, Collection) for c in x_coords):
            raise TypeError("Elements in 'x_coords' must be collections.")
        if any(any(not isinstance(d, (int, float)) for d in c) for c in z_coords):
            raise TypeError("The elements in 'z_coords' must contain ints or floats.")
        if any(any(not isinstance(d, (int, float)) for d in c) for c in x_coords):
            raise TypeError("The elements in 'x_coords' must contain ints or floats.")
        self.z_coords = set(tuple(map(float, c)) for c in z_coords)
        self.x_coords = set(tuple(map(float, c)) for c in x_coords)

        if self.z_coords.intersection(self.x_coords) != set():
            raise ValueError("'z_coords' and 'x_coords' have common elements.")

        # Get coordinates of all detectors
        errors = []
        dets_to_round = {}
        self.x_dets = set()
        self.z_dets = set()
        for dem_instr in self.dem:
            if dem_instr.type == "error":
                errors.append(dem_instr)
            elif dem_instr.type == "detector":
                det = dem_instr.targets_copy()[0].val
                targets = tuple(dem_instr.args_copy())
                if len(targets) == 0:
                    raise ValueError(
                        f"All detectors must have coordinates:\n{dem_instr}"
                    )
                num_round = targets[-1]
                coords = targets[:-1]
                dets_to_round[det] = num_round
                if coords in self.z_coords:
                    self.z_dets.add(det)
                elif coords in self.x_coords:
                    self.x_dets.add(det)
                else:
                    raise ValueError(
                        f"{det} has coorindates not present in 'z_coords' or 'x_coords'."
                    )
            else:
                raise TypeError(f"Unknown {dem_instr} in 'dem'.")

        if len(self.x_dets.union(self.z_dets)) != self.num_dets:
            raise ValueError("Not all detectors have coordinates.")

        # Process the errors
        dem_z = stim.DetectorErrorModel()
        dem_x = stim.DetectorErrorModel()
        self.zedges_to_xdets = {}
        hyperedges = []
        meas_faults = {}
        for error in errors:
            dets = get_detectors(error)
            logs = get_logicals(error)
            z_dets = self.z_dets.intersection(dets)
            x_dets = self.x_dets.intersection(dets)

            if len(x_dets) <= 2 and len(z_dets) == 0:
                # Pauli or measurement errors.
                dem_x.append(error)
                continue
            if (
                len(z_dets) == 2
                and abs(dets_to_round[list(z_dets)[0]] - dets_to_round[list(z_dets)[1]])
                == 1
                and len(logs) == 0
            ):
                # Possible Z-type measurement error.
                z_dets = frozenset(z_dets)
                if z_dets not in meas_faults:
                    meas_faults[z_dets] = [error]
                else:
                    meas_faults[z_dets].append(error)
                continue
            if len(z_dets) <= 2 and len(x_dets) == 0:
                # As it is not a Z-type measurement error, it can only be a Pauli error.
                dem_z.append(error)
                z_dets = frozenset(z_dets)
                self.zedges_to_xdets[z_dets] = np.array([], dtype=int)
                continue

            hyperedges.append(error)

        # Process the Z-type measurement errors
        for z_dets, faults in meas_faults.items():
            # pick the most probable one, and in case of tie pick the
            # one triggering less detectors
            faults = sorted(faults, key=lambda x: len(x.targets_copy()))
            faults = sorted(faults, key=lambda x: x.args_copy()[0], reverse=True)
            fault = faults[0]
            hyperedges += faults[1:]

            dets = get_detectors(fault)
            x_dets = np.array(list(self.x_dets.intersection(dets)), dtype=int)
            self.zedges_to_xdets[z_dets] = x_dets
            dem_z.append(fault)

        # Process hyperedges to update the probabilities of 'dem_z' and 'dem_x'
        zedges_to_zind = {
            frozenset(self.z_dets.intersection(get_detectors(err))): k
            for k, err in enumerate(dem_z)
        }
        xedges_to_xind = {
            frozenset(self.x_dets.intersection(get_detectors(err))): k
            for k, err in enumerate(dem_x)
        }
        z_probs = [[] for _, _ in enumerate(dem_z)]
        x_probs = [[] for _, _ in enumerate(dem_x)]

        mwpm_dem_z = dem_z.copy()
        mwpm_dem_x = dem_x.copy()
        for d in range(self.num_dets):
            instr = stim.DemInstruction("detector", args=[], targets=[stim.target_relative_detector_id(d)])
            mwpm_dem_z.append(instr)
            mwpm_dem_x.append(instr)
        mwpm_z = Matching(mwpm_dem_z)
        mwpm_x = Matching(mwpm_dem_x)

        for error in hyperedges:
            #print("hyperedge", error)
            dets = get_detectors(error)
            logs = get_logicals(error)
            z_dets = np.array(list(self.z_dets.intersection(dets)), dtype=int)
            x_dets = np.array(list(self.x_dets.intersection(dets)), dtype=int)

            print("hyperedge")
            print(len(dets), len(z_dets), len(x_dets))

            syndrome_z = np.zeros(self.num_dets, dtype=bool)
            syndrome_z[z_dets] ^= True
            syndrome_x = np.zeros(self.num_dets, dtype=bool)
            syndrome_x[x_dets] ^= True

            z_edges = mwpm_z.decode_to_edges_array(syndrome_z)
            for z_edge in z_edges:
                z_edge = frozenset(i for i in z_edge if i != -1)
                xdets_flipped = self.zedges_to_xdets[z_edge]
                syndrome_x[xdets_flipped] ^= True

                zind = zedges_to_zind[z_edge]
                prob = dem_z[zind].args_copy()[0]
                z_probs[zind].append(prob)
                
                #print(dem_z[zind])

            x_edges = mwpm_x.decode_to_edges_array(syndrome_x)
            for x_edge in x_edges:
                x_edge = frozenset(i for i in x_edge if i != -1)

                xind = xedges_to_xind[x_edge]
                prob = dem_x[xind].args_copy()[0]
                x_probs[xind].append(prob)

                #print(dem_x[xind])

        self.dem_z = stim.DetectorErrorModel()
        self.dem_x = stim.DetectorErrorModel()
        for probs, dem_instr in zip(z_probs, dem_z):
            prob = dem_instr.args_copy()[0]
            new_prob = xor_probs(prob, *probs)
            new_instr = stim.DemInstruction(
                "error", args=[new_prob], targets=dem_instr.targets_copy()
            )
            self.dem_z.append(new_instr)
        for probs, dem_instr in zip(x_probs, dem_x):
            prob = dem_instr.args_copy()[0]
            new_prob = xor_probs(prob, *probs)
            new_instr = stim.DemInstruction(
                "error", args=[new_prob], targets=dem_instr.targets_copy()
            )
            self.dem_x.append(new_instr)

        # Create variables used in self.decode
        self.zedges_to_logs = {}
        for error in self.dem_z:
            dets = get_detectors(error)
            logs = get_logicals(error)
            z_dets = frozenset(self.z_dets.intersection(dets))
            self.zedges_to_logs[z_dets] = np.array(list(logs), dtype=int)

        self.z_dets = np.array(list(self.z_dets), dtype=int)
        self.x_dets = np.array(list(self.x_dets), dtype=int)

        for l in range(self.num_logs):
            instr = stim.DemInstruction(
                "logical_observable",
                args=[],
                targets=[stim.target_logical_observable_id(l)],
            )
            self.dem_z.append(instr)
            self.dem_x.append(instr)
        for d in range(self.num_dets):
            instr = stim.DemInstruction("detector", args=[], targets=[stim.target_relative_detector_id(d)])
            self.dem_z.append(instr)
            self.dem_x.append(instr)

        self.mwpm_z = Matching(self.dem_z)
        self.mwpm_x = Matching(self.dem_x)

        return

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        if syndrome.shape != (self.num_dets,):
            raise ValueError(
                f"'syndrome' must have shape {(self.num_dets,)}, "
                f"but shape {syndrome.shape} was given."
            )
        if syndrome.dtype != bool:
            raise ValueError("'syndrome' must be np.ndarray[bool].")

        log_flips = np.zeros(self.num_logs, dtype=bool)

        syndrome_z = syndrome.copy()
        syndrome_z[self.x_dets] = False
        syndrome_x = syndrome.copy()
        syndrome_x[self.z_dets] = False

        z_edges = self.mwpm_z.decode_to_edges_array(syndrome_z)
        for z_edge in z_edges:
            z_edge = frozenset(i for i in z_edge if i != -1)
            xdets_flipped = self.zedges_to_xdets[z_edge]
            logs_flipped = self.zedges_to_logs[z_edge]
            syndrome_x[xdets_flipped] ^= True
            log_flips[logs_flipped] ^= True

        log_flips ^= self.mwpm_x.decode(syndrome_x).astype(bool)

        return log_flips
