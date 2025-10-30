import numpy as np
import stim
from surface_sim.setups import CircuitNoiseSetup
from surface_sim.models import CircuitNoiseModel
from surface_sim import Detectors
from surface_sim.experiments import schedule_from_circuit, experiment_from_schedule
from surface_sim.circuit_blocks.unrot_surface_code_css import gate_to_iterator
from surface_sim.layouts import unrot_surface_codes

from lomatching import (
    MoMatching,
    BeliefMoMatching,
    get_reliable_observables,
    remove_obs_except,
)


def test_MoMatching():
    layouts = unrot_surface_codes(2, distance=3)
    setup = CircuitNoiseSetup()
    setup.set_var_param("prob", 1e-3)
    model = CircuitNoiseModel.from_layouts(setup, *layouts)
    detectors = Detectors.from_layouts("pre-gate", *layouts)
    stab_coords = [{} for _ in layouts]
    for l, layout in enumerate(layouts):
        coords = layout.anc_coords
        stab_coords[l][f"Z"] = [v for k, v in coords.items() if k[0] == "Z"]
        stab_coords[l][f"X"] = [v for k, v in coords.items() if k[0] == "X"]

    unencoded_circuit = stim.Circuit(
        """
        RX 0 1
        TICK
        CNOT 0 1
        TICK
        CNOT 1 0
        TICK
        CNOT 0 1
        TICK
        CNOT 1 0
        H 0 1
        TICK
        S 0
        TICK 
        S 0
        TICK
        MZ 0 1
        """
    )
    schedule = schedule_from_circuit(unencoded_circuit, layouts, gate_to_iterator)
    encoded_circuit = experiment_from_schedule(
        schedule, model, detectors, anc_reset=True, anc_detectors=None
    )

    decoder = MoMatching(encoded_circuit, stab_coords)

    dem = decoder.dem
    sampler = dem.compile_sampler()
    syndrome, _, _ = sampler.sample(shots=10)

    predictions = decoder.decode(syndrome[0])
    assert predictions.shape == (2,)

    predictions = decoder.decode_batch(syndrome)
    assert predictions.shape == (10, 2)

    return


def test_MoMatching_performance():
    unencoded_circuits = [
        stim.Circuit("R 0\nTICK\nTICK\nTICK\nM 0"),
        stim.Circuit("RX 0\nRZ 1\nTICK\nTICK\nTICK\nM 0 1"),
        stim.Circuit("RX 0\nRZ 1\nTICK\nCNOT 0 1\nTICK\nM 0 1"),
        stim.Circuit("R 0 1\nTICK\nCNOT 0 1\nTICK\nM 0 1"),
    ]
    for gate_frame in ["pre-gate", "post-gate"]:
        for unencoded_circuit in unencoded_circuits:
            log_prob_decode = []
            log_prob_decode_batch = []
            for distance in [3, 5, 7]:
                layouts = unrot_surface_codes(
                    unencoded_circuit.num_qubits, distance=distance
                )
                setup = CircuitNoiseSetup()
                setup.set_var_param("prob", 1e-3)
                model = CircuitNoiseModel.from_layouts(setup, *layouts)
                detectors = Detectors.from_layouts(gate_frame, *layouts)
                stab_coords = [{} for _ in layouts]
                for l, layout in enumerate(layouts):
                    coords = layout.anc_coords
                    stab_coords[l][f"Z"] = [v for k, v in coords.items() if k[0] == "Z"]
                    stab_coords[l][f"X"] = [v for k, v in coords.items() if k[0] == "X"]

                schedule = schedule_from_circuit(
                    unencoded_circuit, layouts, gate_to_iterator
                )
                encoded_circuit = experiment_from_schedule(
                    schedule, model, detectors, anc_reset=True, anc_detectors=None
                )
                encoded_circuit = remove_obs_except(
                    encoded_circuit, get_reliable_observables(encoded_circuit)
                )

                decoder = MoMatching(encoded_circuit, stab_coords)

                dem = decoder.dem
                sampler = dem.compile_sampler()
                syndromes, log_flips, _ = sampler.sample(shots=1_000)

                predictions = decoder.decode_batch(syndromes)
                log_prob_decode_batch.append(
                    (predictions != log_flips).any(axis=1).mean()
                )

                predictions = np.array([decoder.decode(s) for s in syndromes])
                log_prob_decode.append((predictions != log_flips).any(axis=1).mean())

            assert (
                log_prob_decode_batch[0]
                >= log_prob_decode_batch[1]
                >= log_prob_decode_batch[2]
            )
            assert log_prob_decode[0] >= log_prob_decode[1] >= log_prob_decode[2]

    return


def test_BeliefMoMatching():
    layouts = unrot_surface_codes(2, distance=3)
    setup = CircuitNoiseSetup()
    setup.set_var_param("prob", 1e-3)
    model = CircuitNoiseModel.from_layouts(setup, *layouts)
    detectors = Detectors.from_layouts("pre-gate", *layouts)
    stab_coords = [{} for _ in layouts]
    for l, layout in enumerate(layouts):
        coords = layout.anc_coords
        stab_coords[l][f"Z"] = [v for k, v in coords.items() if k[0] == "Z"]
        stab_coords[l][f"X"] = [v for k, v in coords.items() if k[0] == "X"]

    unencoded_circuit = stim.Circuit(
        """
        RX 0 1
        TICK
        CNOT 1 0
        H 0 1
        TICK
        S 0
        TICK
        MZ 0 1
        """
    )
    schedule = schedule_from_circuit(unencoded_circuit, layouts, gate_to_iterator)
    encoded_circuit = experiment_from_schedule(
        schedule, model, detectors, anc_reset=True, anc_detectors=None
    )

    decoder = BeliefMoMatching(encoded_circuit, stab_coords)

    dem = decoder.dem
    sampler = dem.compile_sampler()
    syndrome, _, _ = sampler.sample(shots=10)

    predictions = decoder.decode(syndrome[0])
    assert predictions.shape == (2,)

    predictions = decoder.decode_batch(syndrome)
    assert predictions.shape == (10, 2)

    return


def test_BeliefMoMatching_performance():
    unencoded_circuits = [
        stim.Circuit("R 0\nTICK\n\nM 0"),
        stim.Circuit("RX 0\nRZ 1\nTICK\nM 0 1"),
        stim.Circuit("RX 0\nRZ 1\nTICK\nCNOT 0 1\nTICK\nM 0 1"),
    ]
    for gate_frame in ["pre-gate"]:
        for unencoded_circuit in unencoded_circuits:
            log_prob_decode = []
            log_prob_decode_batch = []
            for distance in [3, 5]:
                layouts = unrot_surface_codes(
                    unencoded_circuit.num_qubits, distance=distance
                )
                setup = CircuitNoiseSetup()
                setup.set_var_param("prob", 1e-3)
                model = CircuitNoiseModel.from_layouts(setup, *layouts)
                detectors = Detectors.from_layouts(gate_frame, *layouts)
                stab_coords = [{} for _ in layouts]
                for l, layout in enumerate(layouts):
                    coords = layout.anc_coords
                    stab_coords[l][f"Z"] = [v for k, v in coords.items() if k[0] == "Z"]
                    stab_coords[l][f"X"] = [v for k, v in coords.items() if k[0] == "X"]

                schedule = schedule_from_circuit(
                    unencoded_circuit, layouts, gate_to_iterator
                )
                encoded_circuit = experiment_from_schedule(
                    schedule, model, detectors, anc_reset=True, anc_detectors=None
                )
                encoded_circuit = remove_obs_except(
                    encoded_circuit, get_reliable_observables(encoded_circuit)
                )

                decoder = BeliefMoMatching(encoded_circuit, stab_coords)

                dem = decoder.dem
                sampler = dem.compile_sampler()
                syndromes, log_flips, _ = sampler.sample(shots=1_000)

                predictions = decoder.decode_batch(syndromes)
                log_prob_decode_batch.append(
                    (predictions != log_flips).any(axis=1).mean()
                )

                predictions = np.array([decoder.decode(s) for s in syndromes])
                log_prob_decode.append((predictions != log_flips).any(axis=1).mean())

            assert log_prob_decode_batch[0] >= log_prob_decode_batch[1]
            assert log_prob_decode[0] >= log_prob_decode[1]

    return
