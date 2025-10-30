import pickle
import stim
from surface_sim.setups import CircuitNoiseSetup
from surface_sim.models import CircuitNoiseModel
from surface_sim import Detectors
from surface_sim.experiments import schedule_from_circuit, experiment_from_schedule
from surface_sim.circuit_blocks.unrot_surface_code_css import gate_to_iterator
from surface_sim.layouts import unrot_surface_codes

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

encoded_circuit.to_file("encoded_circuit.stim")
with open("stab_coords.pkl", "wb") as outp:  # Overwrites any existing file.
    pickle.dump(stab_coords, outp, pickle.HIGHEST_PROTOCOL)
