import stim
from surface_sim.setups import CircuitNoiseSetup
from surface_sim.models import CircuitNoiseModel
from surface_sim import Detectors
from surface_sim.experiments import schedule_from_circuit, experiment_from_schedule
from surface_sim.circuit_blocks.unrot_surface_code_css import gate_to_iterator
from surface_sim.layouts import unrot_surface_codes

from lomatching import MoMatching


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
        OBSERVABLE_INCLUDE(0) rec[-2]
        OBSERVABLE_INCLUDE(1) rec[-1]
        """
    )
    schedule = schedule_from_circuit(unencoded_circuit[:-2], layouts, gate_to_iterator)
    encoded_circuit = experiment_from_schedule(
        schedule, model, detectors, anc_reset=True, anc_detectors=None
    )

    decoder = MoMatching(unencoded_circuit, encoded_circuit, stab_coords)

    dem = decoder.encoded_circuit_with_only_reliable_observables.detector_error_model()
    sampler = dem.compile_sampler()
    syndrome, _, _ = sampler.sample(shots=10)

    predictions = decoder.decode(syndrome[0])
    assert predictions.shape == (2,)

    predictions = decoder.decode_batch(syndrome)
    assert predictions.shape == (10, 2)

    return
