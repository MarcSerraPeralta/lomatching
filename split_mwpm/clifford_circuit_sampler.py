import numpy as np
import stim


def sample_clifford_circuit(num_qubits: int) -> stim.Circuit:
    """
    Samples a uniformly random Clifford operation and returns
    a corresponding circuit with H, S, CX.

    This function uses ``stim.Tableau.random`` and
    ``stim.Tableau.to_circuit``.

    Parameters
    ----------
    num_qubits
        Number of qubits the Clifford operation should act on.

    Returns
    -------
    circuit
        Stim circuit for the random Clifford operation which only
        includes a single operation per qubit between TICKS.
        The circuit includes resets and measurements for all qubits
        at the beginning and end of the circuit, respectively.
    """
    circuit = stim.Tableau.random(num_qubits).to_circuit(method="elimination")

    # split all gates in the circuit into pairs because can be e.g.
    # CX 0 2 0 1 or S 0 0
    # add a tick or the instructions are going to be compressed again
    circuit_tmp = stim.Circuit()
    for instr in circuit.flattened():
        if instr.name != "CX":
            for t in instr.targets_copy():
                circuit_tmp.append(stim.CircuitInstruction(instr.name, targets=[t]))
                circuit_tmp.append(stim.CircuitInstruction("TICK"))
            continue

        t = instr.targets_copy()
        pairs = zip(*[iter(t)] * 2)
        for pair in pairs:
            circuit_tmp.append(stim.CircuitInstruction("CX", targets=pair))
            circuit_tmp.append(stim.CircuitInstruction("TICK"))

    # split circuit into blocks that have only one operation per qubit
    blocks = []
    curr_block = []
    num_gates_qubits = np.zeros(num_qubits, dtype=int)
    for instr in circuit_tmp.flattened():
        if instr.name == "TICK":
            continue

        qubits = np.array([t.value for t in instr.targets_copy()])
        num_gates_qubits[qubits] += 1

        if (num_gates_qubits > 1).any():
            blocks.append(curr_block)
            curr_block = [instr]
            num_gates_qubits = np.zeros(num_qubits, dtype=int)
            num_gates_qubits[qubits] += 1
        else:
            curr_block.append(instr)
    blocks.append(curr_block)

    # merge blocks and separate them by TICKs
    circuit = stim.Circuit()
    circuit.append(stim.CircuitInstruction("R", targets=list(range(num_qubits))))
    circuit.append(stim.CircuitInstruction("TICK"))
    for block in blocks:
        for instr in block:
            circuit.append(instr)
        circuit.append(stim.CircuitInstruction("TICK"))
    circuit.append(stim.CircuitInstruction("M", targets=list(range(num_qubits))))

    return circuit
