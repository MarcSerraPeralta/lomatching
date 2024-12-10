from split_mwpm.clifford_circuit_sampler import sample_clifford_circuit
from split_mwpm.greedy_algorithm import get_ops


def test_sample_clifford_circuit():
    circuit = sample_clifford_circuit(num_qubits=3)

    # check that generates a valid circuit for processing
    _ = get_ops(circuit)

    return
