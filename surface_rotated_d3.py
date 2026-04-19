"""
Rotated d=3 Surface Code (17 Qubits)

This file builds a distance-3 rotated surface code utilizing a total footprint
of 17 qubits (9 data qubits + 8 ancillas) resembling standard Nature 2023 models.

Boundary Logic: 
The rotated formulation sets the top/bottom as X boundaries and left/right as Z boundaries,
significantly cutting down on physical layout area without sacrificing logical distances.
Logical Z path: Connects across the grid horizontally using Data Qubits 0, 1, 2.
"""

import stim
import pymatching
import numpy as np

def build_surface_code(rounds, p=0.001):
    circuit = stim.Circuit()

    # Rotated d=3 mapping (17 qubits total)
    # data qubit layout (row, col) as logical elements in a 3x3 staggered grid:
    data_coords = {
        0:[1,5], 1:[3,5], 2:[5,5],
        3:[1,3], 4:[3,3], 5:[5,3],
        6:[1,1], 7:[3,1], 8:[5,1],
    }
    
    # 8 ancilla qubits
    ancilla_coords = {
         9:[0,4], 10:[4,4], 11:[2,2], 12:[6,2],  # Z-ancillas (Measure X parity)
        13:[4,6], 14:[2,4], 15:[4,2], 16:[2,0],  # X-ancillas (Measure Z parity)
    }

    for q, coord in {**data_coords, **ancilla_coords}.items():
        circuit.append("QUBIT_COORDS", [q], coord)

    # ancillas and their role
    z_ancillas = [9, 10, 11, 12]    # Z-type stabilizers
    x_ancillas = [13, 14, 15, 16]   # X-type stabilizers
    active_ancillas = sorted(x_ancillas + z_ancillas) 

    # i within the 8-ancilla window (ancilla_idx - 9)
    active_i = [a - 9 for a in active_ancillas]        

    def extract_syndromes(ckt, p_curr):
        def add_cnot_step(cnot_targets):
            ckt.append("CNOT", cnot_targets)
            if p_curr > 0:
                ckt.append("DEPOLARIZE2", cnot_targets, p_curr)
                # Find idle qubits: 17 total
                active_set = set(cnot_targets)
                idle = [q for q in range(17) if q not in active_set]
                if idle:
                    ckt.append("DEPOLARIZE1", idle, p_curr / 10)
            ckt.append("TICK")

        def add_h_step(h_targets):
            ckt.append("H", h_targets)
            if p_curr > 0:
                # Both H and idle qubits get p/10, so just apply to all 17
                ckt.append("DEPOLARIZE1", range(17), p_curr / 10)
            ckt.append("TICK")

        # Z checks
        add_cnot_step([1,10, 3,11, 5,12]) # Z step 1 NW
        add_cnot_step([0,9,  2,10, 4,11]) # Z step 2 NE
        add_cnot_step([4,10, 6,11, 8,12]) # Z step 3 SW
        add_cnot_step([3,9,  5,10, 7,11]) # Z step 4 SE

        # X checks
        add_h_step(x_ancillas)
        add_cnot_step([14,0, 15,4, 16,6]) # X step 1 NW
        add_cnot_step([14,1, 15,5, 16,7]) # X step 2 NE
        add_cnot_step([13,1, 14,3, 15,7]) # X step 3 SW
        add_cnot_step([13,2, 14,4, 15,8]) # X step 4 SE
        add_h_step(x_ancillas)

    # Initial ideal preparation map
    circuit.append("R", range(17))
    circuit.append("TICK")

    z_active_i = [a - 9 for a in z_ancillas]
    extract_syndromes(circuit, 0)
    circuit.append("MR", range(9, 17))
    for i in z_active_i:
        circuit.append("DETECTOR", [stim.target_rec(-8 + i)], ancilla_coords[9 + i])
    circuit.append("TICK")

    # Cycle round execution
    cycle = stim.Circuit()
    extract_syndromes(cycle, p)
    cycle.append("MR", range(9, 17), p / 10)
    cycle.append("DEPOLARIZE1", range(9), p / 10)

    for i in active_i:
        cycle.append(
            "DETECTOR",
            [stim.target_rec(-8 + i), stim.target_rec(-16 + i)],
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
    print("Running Rotated $d=3$ Surface Code Simulation (17 qubits)...")
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
