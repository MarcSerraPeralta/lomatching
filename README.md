# lomatching

![example workflow](https://github.com/MarcSerraPeralta/lomatching/actions/workflows/actions.yaml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![PyPI](https://img.shields.io/pypi/v/lomatching?label=pypi%20package)

Decoder for (fold-)transversal logical gates in surface codes based on MWPM.

## Installation

This package is available in PyPI, thus it can be installed using
```
pip install lomatching
```
or alternatively, it can be installed from source using
```
git clone git@github.com:MarcSerraPeralta/lomatching.git
pip install lomatching/
```

## Example

```
import stim

from surface_sim.setup import CircuitNoiseSetup
from surface_sim.models import CircuitNoiseModel
from surface_sim import Detectors
from surface_sim.experiments import schedule_from_circuit, experiment_from_schedule
from surface_sim.circuit_blocks.unrot_surface_code_css import gate_to_iterator
from surface_sim.layouts import unrot_surface_codes

from lomatching import MoMatching

# circuit considered
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

# generate encoded circuit
layouts = unrot_surface_codes(2, distance=3)
qubit_inds = {}
anc_coords = {}
anc_qubits = []
for l, layout in enumerate(layouts):
    qubit_inds.update(layout.qubit_inds())
    anc_qubits += layout.get_qubits(role="anc")
    coords = layout.anc_coords()
    anc_coords.update(coords)

setup = CircuitNoiseSetup()
setup.set_var_param("prob", 1e-3)
model = CircuitNoiseModel(setup=setup, qubit_inds=qubit_inds)
detectors = Detectors(anc_qubits, frame="pre-gate", anc_coords=anc_coords)

schedule = schedule_from_circuit(circuit, layouts, gate_to_iterator)
experiment = experiment_from_schedule(
    schedule, model, detectors, anc_reset=True, anc_detectors=None
)

# prepare inputs for MoMatching
dem = experiment.detector_error_model(allow_gauge_detectors=True)
stab_coords = {}
for l, layout in enumerate(layouts):
    stab_coords[f"Z{l}"] = [v for k, v in coords.items() if k[0] == "Z"]
    stab_coords[f"X{l}"] = [v for k, v in coords.items() if k[0] == "X"]

decoder = MoMatching(dem, circuit, [["X0"], ["X1"]], stab_coords, "pre-gate")

# run MoMatching
sampler = dem.compile_sampler()
syndrome, log_flips, _ = sampler.sample(shots=10)

predictions = decoder.decode_batch(syndrome)
log_errors = (predictions != log_flips)
```


## How do I cite `lomatching`?

When using `lomatching` for research, please cite:


