import stim
import pickle
from lomatching import BeliefMoMatching

encoded_circuit = stim.Circuit.from_file(file="encoded_circuit.stim")
with open("stab_coords.pkl", "rb") as inp:
    stab_coords = pickle.load(inp)

decoder = BeliefMoMatching(encoded_circuit, stab_coords)
