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
import numpy as np
import stim

from surface_sim.setup import CircuitNoiseSetup
from surface_sim.models import CircuitNoiseModel
from surface_sim import Detectors
from surface_sim.experiments import schedule_from_circuit, experiment_from_schedule
from surface_sim.circuit_blocks.unrot_surface_code_css import gate_to_iterator
from surface_sim.layouts import unrot_surface_codes

from lomatching import MoMatching, get_reliable_observables, remove_obs_except

# circuit considered
circuit = stim.Circuit(
    """
    RX 0
    RZ 1
    TICK
    CNOT 0 1
    MX 0 1
    """
)

# generate encoded circuit
layouts = unrot_surface_codes(2, distance=3)
setup = CircuitNoiseSetup()
setup.set_var_param("prob", 1e-3)
model = CircuitNoiseModel.from_layouts(setup, *layouts)
detectors = Detectors.from_layouts("pre-gate", *layouts)

schedule = schedule_from_circuit(circuit, layouts, gate_to_iterator)
encoded_circuit = experiment_from_schedule(
    schedule, model, detectors, anc_reset=True, anc_detectors=None
)

# prepare inputs for MoMatching
stab_coords = [{} for _ in layouts]
for l, layout in enumerate(layouts):
    coords = layout.anc_coords
    stab_coords[l][f"Z"] = [v for k, v in coords.items() if k[0] == "Z"]
    stab_coords[l][f"X"] = [v for k, v in coords.items() if k[0] == "X"]

reliable_obs = get_reliable_observables(encoded_circuit)
encoded_circuit = remove_obs_except(encoded_circuit, reliable_obs)

decoder = MoMatching(encoded_circuit, stab_coords)

# run MoMatching
sampler = encoded_circuit.detector_error_model().compile_sampler()
syndromes, log_flips, _ = sampler.sample(shots=10)

predictions = decoder.decode_batch(syndromes)
log_errors = (predictions != log_flips).any(axis=1)

print("Logical error probability:", np.average(log_errors))
```


## How do I cite `lomatching`?

When using `lomatching` for research, please cite:
```
@misc{serraperalta2025decoding,
      title={Decoding across transversal Clifford gates in the surface code}, 
      author={Marc Serra-Peralta and Mackenzie H. Shaw and Barbara M. Terhal},
      year={2025},
      eprint={2505.13599},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2505.13599}, 
}
```

