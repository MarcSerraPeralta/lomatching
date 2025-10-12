import pytest
import stim
from surface_sim.setups import CircuitNoiseSetup
from surface_sim.models import IncResMeasNoiseModel
from surface_sim import Detectors
from surface_sim.experiments import schedule_from_circuit, experiment_from_schedule
from surface_sim.circuit_blocks.unrot_surface_code_css import gate_to_iterator
from surface_sim.layouts import unrot_surface_codes

from lomatching.util import (
    get_observing_region,
    get_reliable_observables,
    get_subgraph,
    get_qubit_measurements,
    get_all_reset_paulistrings,
    commute,
    remove_obs_except,
)


def test_get_observing_region():
    circuit = stim.Circuit(
        """
        R 0
        TICK
        H 0 1
        TICK
        H 0
        TICK
        RX 1
        TICK
        CNOT 0 1
        H 0
        TICK
        MX 0 
        TICK
        X 1
        TICK
        M 1
        OBSERVABLE_INCLUDE(0) rec[-1]
        OBSERVABLE_INCLUDE(1) rec[-2] rec[-1]
        """
    )

    obs_region = get_observing_region(circuit, observable=[0])
    expected_obs_region = {
        0: stim.PauliString("+Z_"),
        1: stim.PauliString("+X_"),
        2: stim.PauliString("+Z_"),
        3: stim.PauliString("+ZZ"),
        4: stim.PauliString("+_Z"),
        5: stim.PauliString("+_Z"),
        6: stim.PauliString("+_Z"),
    }
    assert expected_obs_region == obs_region

    obs_region = get_observing_region(circuit, observable=[1])
    expected_obs_region = {
        3: stim.PauliString("+_Z"),
        4: stim.PauliString("+XZ"),
        5: stim.PauliString("+_Z"),
        6: stim.PauliString("+_Z"),
    }
    assert expected_obs_region == obs_region

    return


def test_get_reliable_observables():
    circuit = stim.Circuit(
        """
        RX 0
        R 1
        TICK
        CNOT 0 1
        H 0
        TICK
        MX 0
        M 1
        OBSERVABLE_INCLUDE(0) rec[-2]
        OBSERVABLE_INCLUDE(1) rec[-1]
        """
    )
    reliable_obs = get_reliable_observables(circuit)
    expected_reliable_obs = [set([0, 1])]
    assert reliable_obs == expected_reliable_obs

    circuit = stim.Circuit(
        """
        RX 0
        RZ 1 2
        TICK
        CNOT 1 2
        CNOT 0 1
        TICK
        MZ 0 1 2
        OBSERVABLE_INCLUDE(0) rec[-3]
        OBSERVABLE_INCLUDE(1) rec[-2]
        OBSERVABLE_INCLUDE(2) rec[-1]
        """
    )
    reliable_obs = get_reliable_observables(circuit)
    expected_reliable_obs = [set([0, 1]), set([2])]
    assert reliable_obs == expected_reliable_obs

    circuit = stim.Circuit(
        """
        RX 0
        TICK
        R 1 2
        TICK
        CNOT 0 1
        TICK
        H 0 1 2
        TICK
        H 0 1 2
        TICK
        CNOT 1 2
        TICK
        M 1 2
        TICK 
        M 0
        OBSERVABLE_INCLUDE(0) rec[-1]
        OBSERVABLE_INCLUDE(1) rec[-3]
        OBSERVABLE_INCLUDE(2) rec[-2]
        """
    )
    reliable_obs = get_reliable_observables(circuit)
    expected_reliable_obs = [set([0, 2]), set([1, 2])]
    assert reliable_obs == expected_reliable_obs

    circuit = stim.Circuit(
        """
        RX 0
        TICK
        R 1 2
        TICK
        M 1 2
        TICK
        M 0
        OBSERVABLE_INCLUDE(0) rec[-1]
        OBSERVABLE_INCLUDE(1) rec[-3]
        OBSERVABLE_INCLUDE(2) rec[-2]
        """
    )
    reliable_obs = get_reliable_observables(circuit)
    expected_reliable_obs = [set([1]), set([2])]
    assert reliable_obs == expected_reliable_obs

    return


def test_commute():
    region_a, region_b = {0: stim.PauliString("_X_")}, {1: stim.PauliString("_Z_")}
    assert commute(region_a, region_b)

    region_a, region_b = {10: stim.PauliString("_X_")}, {10: stim.PauliString("_Z_")}
    assert not commute(region_a, region_b)

    region_a, region_b = {1: stim.PauliString("XX_")}, {1: stim.PauliString("ZZ_")}
    assert commute(region_a, region_b)

    region_a, region_b = {0: stim.PauliString("_YZ_")}, {0: stim.PauliString("_XX_")}
    assert commute(region_a, region_b)

    region_a, region_b = {1: stim.PauliString("_XX")}, {1: stim.PauliString("_ZX")}
    assert not commute(region_a, region_b)

    return


def test_get_all_reset_paulistrings():
    circuit = stim.Circuit(
        """
        R 0
        TICK
        RX 2
        TICK
        R
        TICK
        H 0 1
        TICK
        RY 0 1
        """
    )
    resets = get_all_reset_paulistrings(circuit)
    expected_resets = {
        0: {0: stim.PauliString("Z__")},
        1: {1: stim.PauliString("__X")},
        2: {4: stim.PauliString("Y__")},
        3: {4: stim.PauliString("_Y_")},
    }

    assert resets == expected_resets

    circuit = stim.Circuit(
        """
        R 0
        RX 2
        TICK
        """
    )
    resets = get_all_reset_paulistrings(circuit)
    expected_resets = {
        0: {0: stim.PauliString("Z__")},
        1: {0: stim.PauliString("__X")},
    }

    assert resets == expected_resets

    return


def test_get_qubit_measurements():
    circuit = stim.Circuit(
        """
        R 0 1
        H 1
        TICK
        CNOT 0 1
        TICK 
        MX 1
        TICK
        R 0
        TICK
        M 0 1
        TICK
        """
    )

    measurements = get_qubit_measurements(circuit)

    expected_measurements = {
        1: {1: "X"},
        3: {0: "Z", 1: "Z"},
    }

    assert measurements == expected_measurements

    circuit = stim.Circuit("R 0\nTICK\nMY 0")
    with pytest.raises(TypeError):
        _ = get_qubit_measurements(circuit)

    return


def test_remove_obs_except():
    circuit = stim.Circuit(
        """
        R 0 1
        TICK
        M 0 1
        TICK
        CNOT 0 1
        OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(1) rec[-1]
        H 0
        TICK
        M 0 1
        """
    )

    new_circuit = remove_obs_except(circuit, [[0, 1]])

    expected_circuit = stim.Circuit(
        """
        R 0 1
        TICK
        M 0 1
        TICK
        CNOT 0 1
        H 0
        TICK
        M 0 1
        OBSERVABLE_INCLUDE(0) rec[-4]
        """
    )

    assert new_circuit == expected_circuit

    return


def test_get_subgraph():
    layouts = unrot_surface_codes(2, distance=3)
    setup = CircuitNoiseSetup()
    setup.set_var_param("prob", 1e-3)
    model = IncResMeasNoiseModel.from_layouts(setup, *layouts)
    detectors = Detectors.from_layouts("pre-gate", *layouts)
    stab_coords = [{} for _ in layouts]
    for l, layout in enumerate(layouts):
        coords = layout.anc_coords
        stab_coords[l][f"Z"] = [v for k, v in coords.items() if k[0] == "Z"]
        stab_coords[l][f"X"] = [v for k, v in coords.items() if k[0] == "X"]

    unencoded_circuit = stim.Circuit(
        """
        RX 0
        R 1
        TICK
        CNOT 0 1
        TICK
        H 0
        S 1
        TICK
        H 0
        S 1
        TICK
        MZ 0 1
        OBSERVABLE_INCLUDE(0) rec[-2]
        OBSERVABLE_INCLUDE(1) rec[-1]
        """
    )
    schedule = schedule_from_circuit(unencoded_circuit[:-2], layouts, gate_to_iterator)
    encoded_circuit = experiment_from_schedule(
        schedule, model, detectors, anc_reset=True, anc_detectors=None
    )

    dem_subgraph, det_inds = get_subgraph(
        unencoded_circuit, encoded_circuit, (0, 1), stab_coords
    )

    # check that the decoding subgraph is a graph
    for instr in dem_subgraph.flattened():
        if instr.type != "error":
            continue

        dets = [t for t in instr.targets_copy() if t.is_relative_detector_id()]
        contains_separator = any(
            [True for t in instr.targets_copy() if t.is_separator()]
        )
        # because the incoming noise model is applied both before and after the
        # logical gates, there are X and Z incoming errors that correspond to
        # hyperedges.
        if not contains_separator:
            assert len(dets) <= 2
        else:
            assert len(dets) == 4

    assert len(det_inds) == dem_subgraph.num_detectors

    return
