import math
import numpy as np
import numba as nb
import stim
from scipy.sparse import csc_matrix


@nb.jit("float64(float64[:])", nopython=True)
def comb_probs_to_w(probs: np.ndarray) -> float:
    p = probs[0]
    for q in probs[1:]:
        p = p * (1 - q) + (1 - p) * q
    # avoid numerical issues with math.log
    eps = 1e-14
    if p < eps:
        p = eps
    elif p > 1 - eps:
        p = 1 - eps
    return -math.log(p / (1 - p))


def dem_to_hplc(
    dem: stim.DetectorErrorModel,
) -> tuple[csc_matrix, np.ndarray, csc_matrix, np.ndarray]:
    """Returns the detector-error matrix, error probabilities, logicals-error matrix,
    and the detector coordinates given a ``stim.DetectorErrorModel``.
    It keeps the ordering of the errors in ``dem.flattened()`` and the order of
    the detectors when building the output.

    Parameters
    ----------
    dem
        Detector error model (DEM).

    Returns
    -------
    det_err_matrix : np.ndarray(D, E)
        Detector-error matrix which related the error mechanisms and the detectors
        they trigger. ``D`` is the number of detectors and ``E`` the number
        of error mechanisms.
    err_probs : np.ndarray(E)
        Probabilities for each error mechanism.
    log_err_matrix : np.ndarray(L, E)
        Logicals-error matrix which relates the error mechanisms and the logical
        observables that they flip. ``L`` is the number of logical observables.
    coords : np.ndarray(D, C)
        Coordinates associated with each detector, with ``C`` the number of coordinates.
        If no coordinates are present in ``dem``, an empty array of shape ``(D,)``
        is returned.
    """
    if not isinstance(dem, stim.DetectorErrorModel):
        raise TypeError(
            f"'dem' must be a stim.DetectorErrorModel, but {type(dem)} was given."
        )

    det_err_list = []
    err_probs_list = []
    log_err_list = []
    coords_dict = {}

    for instr in dem.flattened():
        if instr.type == "error":
            # get information
            p = instr.args_copy()[0]
            dets, logs = [], []
            for t in instr.targets_copy():
                if t.is_relative_detector_id():
                    dets.append(t.val)
                elif t.is_logical_observable_id():
                    logs.append(t.val)
                elif t.is_separator():
                    pass
                else:
                    raise ValueError(f"{t} is not implemented.")
            det_err_list.append(dets)
            err_probs_list.append(p)
            log_err_list.append(logs)
        elif instr.type == "detector":
            det = instr.targets_copy()[0].val
            coords_dict[det] = instr.args_copy()
        elif instr.type == "logical_observable":
            pass
        else:
            raise ValueError(f"{instr} is not implemented.")

    det_err_matrix = _list_to_csc_matrix(
        det_err_list, shape=(dem.num_detectors, len(det_err_list))
    )
    log_err_matrix = _list_to_csc_matrix(
        log_err_list, shape=(dem.num_observables, len(log_err_list))
    )
    err_probs = np.array(err_probs_list)
    coords = np.empty(shape=(dem.num_detectors))
    if coords_dict:
        if dem.num_detectors != len(coords_dict):
            raise ValueError(
                "Either all the detectors have coordinates or none,"
                " but not all of them have."
            )
        coords = np.array([coords_dict[i] for i in range(dem.num_detectors)])

    return det_err_matrix, err_probs, log_err_matrix, coords


def _list_to_csc_matrix(my_list: list[list[int]], shape: tuple[int, int]) -> csc_matrix:
    """Returns ``csc_matrix`` built form the given list.

    The output matrix has all elements zero except in each column ``i`` it has
    ones on the rows ``my_list[i]``.

    Parameters
    ----------
    my_list
        List of lists of integers containing the entries with ones in the csc_matrix.
    shape
        Shape of the ``csc_matrix``.

    Returns
    -------
    matrix
        The described ``csc_matrix`` with 0s and 1s.
    """
    if shape[1] < len(my_list):
        raise ValueError(
            "The shape of the csc_matrix is not large enough to accomodate all the data."
        )

    num_ones = sum(len(l) for l in my_list)
    data = np.ones(
        num_ones, dtype=np.uint8
    )  # smallest integer size (bool operations do not work)
    row_inds = np.empty(num_ones, dtype=int)
    col_inds = np.empty(num_ones, dtype=int)
    i = 0
    for c, det_inds in enumerate(my_list):
        for r in det_inds:
            row_inds[i] = r
            col_inds[i] = c
            i += 1

    return csc_matrix((data, (row_inds, col_inds)), shape=shape)


def dem_only_errors(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    dem_errors = stim.DetectorErrorModel()
    for instr in dem.flattened():
        if instr.type == "error":
            dem_errors.append(instr)
    return dem_errors


def valid_decomposition(
    h_instr: stim.DemInstruction,
    e_instrs: list[stim.DemInstruction],
) -> bool:
    h_dets = set(i.val for i in h_instr.targets_copy() if i.is_relative_detector_id())
    h_logs = set(i.val for i in h_instr.targets_copy() if i.is_logical_observable_id())
    e_dets = set()
    e_logs = set()
    for e_instr in e_instrs:
        curr_e_dets = set(
            i.val for i in e_instr.targets_copy() if i.is_relative_detector_id()
        )
        curr_e_logs = set(
            i.val for i in e_instr.targets_copy() if i.is_logical_observable_id()
        )
        e_dets.symmetric_difference_update(curr_e_dets)
        e_logs.symmetric_difference_update(curr_e_logs)

    return (h_dets == e_dets) and (h_logs == e_logs)


def get_edges_dict(
    primitive_dem: stim.DetectorErrorModel,
    ignore_decomposition_failures: bool = False,
) -> dict[tuple[int, int], int]:
    edges_dict = {}
    ind_to_logs_dict = {}
    for ind, instr in enumerate(primitive_dem):
        dets = [i.val for i in instr.targets_copy() if i.is_relative_detector_id()]
        logs = [i.val for i in instr.targets_copy() if i.is_logical_observable_id()]
        if len(dets) == 1:
            dets.append(-1)  # boundary node
        dets = tuple(sorted(dets))
        logs = tuple(sorted(logs))

        if dets in edges_dict:
            equiv_ind = edges_dict[dets]
            if (ind_to_logs_dict[equiv_ind] == logs) or ignore_decomposition_failures:
                # repeated instruction or ignore decomposition failure
                continue
            else:
                raise ValueError(
                    "Edges triggering same detector but with different logical effect have been found."
                )

        edges_dict[dets] = ind
        ind_to_logs_dict[ind] = logs

    return edges_dict


def standardize_edge(pymatching_edge: np.ndarray) -> tuple[int, int]:
    return tuple(sorted(pymatching_edge.tolist()))
