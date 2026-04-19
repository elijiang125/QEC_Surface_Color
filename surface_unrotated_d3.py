"""
Unrotated d=3 Surface Code (25 Qubits)

This file builds a distance-3 unrotated surface code utilizing a total footprint
of 25 qubits (9 data qubits + 16 ancillas). 

Boundary Logic: 
The unrotated formulation sets top/bottom boundaries and right/left boundaries
using weight-2 parity checks spanning parallel to each edge. 
Logical Z path: Connects across the grid horizontally using Data Qubits 0, 1, 2.
"""

import stim
import pymatching
import numpy as np

def build_surface_code(rounds, p=0.001):
    circuit = stim.Circuit()

    # Data qubit layout (row, col) as logical elements in an unrotated 3x3 grid:
    data_coords = {
        0:[0,0], 1:[0,2], 2:[0,4],
        3:[2,0], 4:[2,2], 5:[2,4],
        6:[4,0], 7:[4,2], 8:[4,4],
    }
    
    # 16 ancilla qubits 
    ancilla_coords = {
         9:[0,1], 10:[0,3], 11:[1,0], 12:[1,2], 13:[1,4],
        14:[2,1], 15:[2,3], 16:[3,0], 17:[3,2], 18:[3,4],
        19:[4,1], 20:[4,3], 21:[1,1], 22:[1,3], 23:[3,1], 24:[3,3],
    }

    for q, coord in {**data_coords, **ancilla_coords}.items():
        circuit.append("QUBIT_COORDS", [q], coord)

    # active ancillas and their role
    x_ancillas = [9, 20, 22, 23]    # X-type stabilizers
    z_ancillas = [13, 16, 21, 24]   # Z-type stabilizers
    active_ancillas = sorted(x_ancillas + z_ancillas)

    # offset calculations relative to 16 total sweep records
    active_i = [a - 9 for a in active_ancillas]

    def extract_syndromes(ckt, p_curr):
        def add_cnot_step(cnot_targets):
            ckt.append("CNOT", cnot_targets)
            if p_curr > 0:
                ckt.append("DEPOLARIZE2", cnot_targets, p_curr)
                active_set = set(cnot_targets)
                idle = [q for q in range(25) if q not in active_set]
                if idle:
                    ckt.append("DEPOLARIZE1", idle, p_curr / 10)
            ckt.append("TICK")

        def add_h_step(h_targets):
            ckt.append("H", h_targets)
            if p_curr > 0:
                ckt.append("DEPOLARIZE1", range(25), p_curr / 10)
            ckt.append("TICK")

        # Z checks
        add_cnot_step([0,21,  4,24,  2,13]) # NW
        add_cnot_step([1,21,  5,24,  3,16]) # NE
        add_cnot_step([3,21,  7,24,  5,13]) # SW
        add_cnot_step([4,21,  8,24,  6,16]) # SE

        # X checks
        add_h_step(x_ancillas)
        add_cnot_step([22,1,  23,3,   9,0]) # NW
        add_cnot_step([22,2,  23,4,  20,7]) # NE
        add_cnot_step([22,4,  23,6,   9,1]) # SW
        add_cnot_step([22,5,  23,7,  20,8]) # SE
        add_h_step(x_ancillas)

    # Initial ideal preparation map
    circuit.append("R", range(25))
    circuit.append("TICK")

    z_active_i = [a - 9 for a in z_ancillas]
    extract_syndromes(circuit, 0)
    circuit.append("MR", range(9, 25))
    for i in z_active_i:
        circuit.append("DETECTOR", [stim.target_rec(-16 + i)], ancilla_coords[9 + i])
    circuit.append("TICK")

    # Cycle round execution
    cycle = stim.Circuit()
    extract_syndromes(cycle, p)
    cycle.append("MR", range(9, 25), p / 10)
    cycle.append("DEPOLARIZE1", range(9), p / 10)

    for i in active_i:
        cycle.append(
            "DETECTOR",
            [stim.target_rec(-16 + i), stim.target_rec(-32 + i)],
            ancilla_coords[9 + i],
        )
    cycle.append("TICK")

    # Combine circuit rounds
    circuit += cycle * (rounds - 1)
    circuit.append("M", range(9), p / 10)

    # Logical Z = Z_0 \otimes Z_1 \otimes Z_2
    circuit.append(
        "OBSERVABLE_INCLUDE",
        [stim.target_rec(-9), stim.target_rec(-8), stim.target_rec(-7)],
        0,
    )
    return circuit

if __name__ == "__main__":
    print("Running Unrotated $d=3$ Surface Code Simulation (25 qubits)...")
    c = build_surface_code(rounds=3, p=0.001)

    error_model = c.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(error_model)

    sampler = c.compile_detector_sampler()
    syndromes, observables = sampler.sample(shots=10000, separate_observables=True)

    predictions = matching.decode_batch(syndromes)
    logical_errors = np.any(predictions != observables, axis=1)
    error_amt = np.sum(logical_errors)

    print(f"There are {error_amt} errors out of 10000 shots.")
    print(f"Corrected Failure Rate: {(error_amt / 10000) * 100:.4f}%")
