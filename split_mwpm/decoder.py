import stim
import numpy as np
from pymatching import Matching

from qec_util.dem_instrs import (
    get_detectors,
    get_logicals,
    sorted_dem_instr,
    remove_detectors,
)
from qec_util.dems import get_flippable_detectors


class SplitMatching:
    """Decoder that decodes first the Z-type detectors, then removes the
    output faults from the X-type detectors, and finally decodes the
    X-type detectors.
    """

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        dem_z: stim.DetectorErrorModel,
        dem_x: stim.DetectorErrorModel,
    ) -> None:
        """Initializes the ``SplitMatching``.

        Parameters
        ----------
        dem
            DEM containing all detectors.
            It should not contain any gauge detectors to avoid problems.
        dem_z
            DEM containing only Z-type detectors that has been decomposed into edges.
        dem_x
            DEM containing only X-type detectors that has been decomposed into edges.
        """
        if not isinstance(dem, stim.DetectorErrorModel):
            raise TypeError(
                f"'dem' is not a stim.DetectorErrorModel, but a {type(dem)}."
            )
        if not isinstance(dem_z, stim.DetectorErrorModel):
            raise TypeError(
                f"'dem_z' is not a stim.DetectorErrorModel, but a {type(dem_z)}."
            )
        if not isinstance(dem_x, stim.DetectorErrorModel):
            raise TypeError(
                f"'dem_x' is not a stim.DetectorErrorModel, but a {type(dem_x)}."
            )
        if (dem.num_detectors != dem_z.num_detectors) or (
            dem.num_detectors != dem_x.num_detectors
        ):
            raise ValueError(
                "'dem', 'dem_z', and 'dem_x' have a different number of detectors."
            )

        self.dem = dem.flattened()
        self.dem_z = dem_z.flattened()
        self.dem_x = dem_x.flattened()

        self.x_dets = np.array(list(get_flippable_detectors(dem_x)), dtype=int)
        self.z_dets = np.array(list(get_flippable_detectors(dem_z)), dtype=int)
        self.num_logs = dem.num_observables
        self.num_dets = dem.num_detectors

        # remove stim.DemInstruction that leads to a logical flip but triggers no detectors
        new_dem_z = stim.DetectorErrorModel()
        for dem_instr in self.dem_z:
            if (dem_instr.type == "error") and (len(get_detectors(dem_instr)) == 0):
                continue
            new_dem_z.append(dem_instr)
        self.dem_z = new_dem_z

        new_dem_x = stim.DetectorErrorModel()
        for dem_instr in self.dem_x:
            if (dem_instr.type == "error") and (len(get_detectors(dem_instr)) == 0):
                continue
            new_dem_x.append(dem_instr)
        self.dem_x = new_dem_x

        # create dictionary to index 'dem_z' given one of its stim.DemInstruction.
        # It assumes that no instruction is repeated in the DEM
        self.demz_to_index = {
            _sorted_dem_instr_without_p(d): k for k, d in enumerate(self.dem_z)
        }

        # map errors from 'dem' to errors in 'dem_z'.
        # Note that it ensures that both detectors and logicals are the same.
        g_to_gz = {}
        for k, dem_instr in enumerate(self.dem):
            if dem_instr.type != "error":
                continue

            dem_instr = remove_detectors(dem_instr, dets=self.x_dets)
            dem_instr = _sorted_dem_instr_without_p(dem_instr)

            if len(get_detectors(dem_instr)) == 0:
                g_to_gz[k] = None
                continue

            g_to_gz[k] = self.demz_to_index[dem_instr]

        # reverse the map to have a function that maps errors in 'dem_z'
        # to errors in 'dem'. This is used to do the XOR to the
        # X-type defects based on the output errors in MWPM(dem_z)
        gz_to_g_list = {kz: [] for kz, _ in enumerate(dem_z)}
        for k, kz in g_to_gz.items():
            if kz is None:
                continue
            gz_to_g_list[kz].append(k)

        # there are cases in which a single error in 'dem_z' corresponds to
        # multiple errors in 'dem'. In the "r" frame, this could be due to
        # IncNoise + S_L + IncNoise,in which the first IncNoise propagates
        # to X and Z errors which trigger the same Z-type detectors as just
        # X errors in the second IncNoise. Therefore there are two error
        # mechanisms that trigger the same Z-type detectors.
        # The current approach to dealing with these errors is by keeping the
        # one with largest probability, and in the case of equal probability
        # choosing the one with fewer triggered detectors.
        gz_to_g = {}
        for kz, ks in gz_to_g_list.items():
            if len(ks) == 0:
                gz_to_g[kz] = None
                continue
            elif len(ks) == 1:
                gz_to_g[kz] = ks[0]
                continue

            data = {}
            for k in ks:
                dem_instr = self.dem[k]
                prob = dem_instr.args_copy()[0]
                dets = get_detectors(dem_instr)
                data[k] = (prob, dets)

            ks = sorted(ks, key=lambda x: data[x][1])
            ks = sorted(ks, key=lambda x: data[x][0], reverse=True)
            gz_to_g[kz] = ks[0]

        # prepare gz_to_g to be used directly with the output of
        # Matching.decode_to_edges_array
        self.zedges_to_xdets = {}  # {set: np.array}
        self.zedges_to_logs = {}  # {set: np.array}
        x_dets = get_flippable_detectors(self.dem_x)
        for kz, k in gz_to_g.items():
            if k is None:
                continue

            dets_z = get_detectors(self.dem_z[kz])
            dets_z = frozenset(dets_z)  # to be hashable

            dets = get_detectors(self.dem[k])
            dets_x = set(dets).intersection(x_dets)
            dets_x = np.array(list(dets_x), dtype=int)
            self.zedges_to_xdets[dets_z] = dets_x

            logs = get_logicals(self.dem_z[kz])
            logs = np.array(list(logs), dtype=int)
            self.zedges_to_logs[dets_z] = logs

        # prepare MWPM decoders
        self.mwpm_z = Matching(dem_z)
        self.mwpm_x = Matching(dem_x)

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


def _sorted_dem_instr_without_p(dem_instr: stim.DemInstruction) -> stim.DemInstruction:
    """Returns the same output as ``sorted_dem_instr`` but with the associated
    probability set to 1, to avoid problems between DEMs that have different
    probabilities due to combination of errors.
    """
    output = sorted_dem_instr(dem_instr)
    return stim.DemInstruction("error", args=[1], targets=output.targets_copy())
