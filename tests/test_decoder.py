import stim
from surface_sim.setup import CircuitNoiseSetup
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
    stab_coords = {}
    for l, layout in enumerate(layouts):
        coords = layout.anc_coords
        stab_coords[f"Z{l}"] = [v for k, v in coords.items() if k[0] == "Z"]
        stab_coords[f"X{l}"] = [v for k, v in coords.items() if k[0] == "X"]

    circuit = stim.Circuit(
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
        TICK
        MX 0 1
        """
    )
    schedule = schedule_from_circuit(circuit, layouts, gate_to_iterator)
    experiment = experiment_from_schedule(
        schedule, model, detectors, anc_reset=True, anc_detectors=None
    )

    dem = experiment.detector_error_model(allow_gauge_detectors=True)
    decoder = MoMatching(dem, circuit, [["X0"], ["X1"]], stab_coords, "pre-gate")

    sampler = dem.compile_sampler()
    syndrome, _, _ = sampler.sample(shots=10)

    predictions = decoder.decode(syndrome[0])
    assert predictions.shape == (2,)

    predictions = decoder.decode_batch(syndrome)
    assert predictions.shape == (10, 2)

    return
