import stim
import pymatching
import numpy as np

# params we can change here: we're doing rounds + physical noise
def build_surface_code(rounds, p=0.001):
    circuit = stim.Circuit()

    # so we need tobuild a 5x5 grid.  
    # data qubits msut be at even coords, ancillas at odd/boundary coords
    #
    # data qubit layout (row, col):
    #   q_0(0,0)  q_1(0,2)  q_2(0,4)
    #   q_3(2,0)  q_4(2,2)  q_5(2,4)
    #   q_6(4,0)  q_7(4,2)  q_8(4,4)

    # ALL 16 X×Z stabiliser pairs share 0 or 2 data qubits
    
    # X stabilisers (ancilla->data, ancilla starts |+>):
    #   anc  9  (0,1) -> X_0X_1            (top boundary, weight-2)
    #   anc 20 (4,3) -> X_7X_8            (bottom boundary, weight-2)
    #   anc 22 (1,3) -> X_1X_2X_4X_5 
    #   anc 23 (3,1) -> X_3X_4X_6X_7 
    #
    # Z stabilisers (data->ancilla, ancilla starts |0>):
    #   anc 13 (1,4) -> Z_2Z_5            (right boundary, weight-2)
    #   anc 16 (3,0) -> Z_3Z_6            (left boundary, weight-2)
    #   anc 21 (1,1) -> Z_0Z_1Z_3Z_4       
    #   anc 24 (3,3) -> Z_4Z_5Z_7Z_8       
    #
    # ancillas 10,11,12,14,15,17,18,19 are physically present (included in the
    # MR sweep) but idle.
    # in other words, they always read 0, and we add NO detectors for them.
    #
    # Logical Z = Z_0Z_1Z_2  (top row, weight-3)  // anti-commutes with X
    # Logical X = X_0X_3X_6  (left col, weight-3) // commutes with all Z

    data_coords = {
        0:[0,0], 1:[0,2], 2:[0,4],
        3:[2,0], 4:[2,2], 5:[2,4],
        6:[4,0], 7:[4,2], 8:[4,4],
    }
    ancilla_coords = {
         9:[0,1], 10:[0,3], 11:[1,0], 12:[1,2], 13:[1,4],
        14:[2,1], 15:[2,3], 16:[3,0], 17:[3,2], 18:[3,4],
        19:[4,1], 20:[4,3], 21:[1,1], 22:[1,3], 23:[3,1], 24:[3,3],
    }

    for q, coord in {**data_coords, **ancilla_coords}.items():
        circuit.append("QUBIT_COORDS", [q], coord)

    # ancillas and their role
    x_ancillas = [9, 20, 22, 23]    # X-type stabilisers
    z_ancillas = [13, 16, 21, 24]   # Z-type stabilisers
    active_ancillas = sorted(x_ancillas + z_ancillas)  # [9,13,16,20,21,22,23,24]

    # i within the 16-ancilla window (ancilla_idx - 9)
    active_i = [a - 9 for a in active_ancillas]        # [0, 4, 7, 11, 12, 13, 14, 15]

    # okay this is the four step syndrome extraction
    # each TICK touches every data qubit AT MOST ONCE.
    #
    # Z-checks  (data qubit = CONTROL, ancilla = TARGET):
    #   Step 1: q_0->21, q_2->13, q_4->24      data set {0,2,4}
    #   Step 2: q_1->21, q_5->24, q_3->16      data set {1,5,3}
    #   Step 3: q_3->21, q_5->13, q_7->24      data set {3,5,7}
    #   Step 4: q_4->21, q_8->24, q_6->16      data set {4,8,6}
    #
    # X-checks  (ancilla = CONTROL, data = TARGET):
    #   H(x_ancillas)
    #   Step 1: 22->q_2, 23->q_3,  9->q_0     data set {2,3,0}
    #   Step 2: 22->q_1, 23->q_4, 20->q_7     data set {1,4,7}
    #   Step 3: 22->q_5, 23->q_6,  9->q_1     data set {5,6,1}
    #   Step 4: 22->q_4, 23->q_7, 20->q_8     data set {4,7,8}
    #   H(x_ancillas)
    #
    # cross-tick check: q_4 used in Z-steps 1&4 (different TICKs),
    #   q_3 in Z-steps 2&3, q_5 in Z-steps 2&3 ..they're all in different ticks

    def extract_syndromes(ckt):
        # we check z pairings first
        # Ordering: NW corner, NE corner, SW corner, SE corner of each face.
        # Boundary stabs interleaved so they share no data qubit per TICK.
        #
        #   anc_21 Z_0Z_1Z_3Z_4:  q_0(NW) q_1(NE) q_3(SW) q_4(SE)
        #   anc_24 Z_4Z_5Z_7Z_8:  q_4(NW) q_5(NE) q_7(SW) q_8(SE)
        #   anc_13 Z_2Z_5 :  q_2(N) q_5(S)
        #   anc_16 Z_3Z-6:  q_3(N) q_6(S)
        #
        #   Step 1 targets: q_0(21), q_4(24), q_2(13) -> {0,4,2} 
        #   Step 2 targets: q_1(21), q_5(24), q_3(16) -> {1,5,3} 
        #   Step 3 targets: q_3(21), q_7(24), q_5(13) -> {3,7,5} 
        #   Step 4 targets: q_4(21), q_8(24), q_6(16) -> {4,8,6} 
        ckt.append("TICK")
        ckt.append("CNOT", [0,21,  4,24,  2,13]) # Z step 1  (NW)
        ckt.append("TICK")
        ckt.append("CNOT", [1,21,  5,24,  3,16]) # Z step 2  (NE)
        ckt.append("TICK")
        ckt.append("CNOT", [3,21,  7,24,  5,13]) # Z step 3  (SW)
        ckt.append("TICK")
        ckt.append("CNOT", [4,21,  8,24,  6,16]) # Z step 4  (SE)
        ckt.append("TICK")

        # we check the X pairings
        #   anc_22 X_1X-2X_4X-5:  q_1(NW) q_2(NE) q_4(SW) q_5(SE)
        #   anc_23 X_3X_4X_6X_7:  q_3(NW) q_4(NE) q_6(SW) q_7(SE)
        #   anc 9 X_0X_1:  q_0(W) q_1(E)
        #   anc_20 X_7X_8:  q_7(W) q_8(E)
        #
        #   Step 1 targets: q_1(22), q_3(23), q_0(9) -> {1,3,0} 
        #   Step 2 targets: q_2(22), q_4(23), q_7(20) -> {2,4,7} 
        #   Step 3 targets: q_4(22), q_6(23), q_1(9) -> {4,6,1} 
        #   Step 4 targets: q_5(22), q_7(23), q_8(20) -> {5,7,8} 
        ckt.append("H", x_ancillas)
        ckt.append("TICK")
        ckt.append("CNOT", [22,1,  23,3,   9,0])          # X step 1  (NW)
        ckt.append("TICK")
        ckt.append("CNOT", [22,2,  23,4,  20,7])          # X step 2  (NE)
        ckt.append("TICK")
        ckt.append("CNOT", [22,4,  23,6,   9,1])          # X step 3  (SW)
        ckt.append("TICK")
        ckt.append("CNOT", [22,5,  23,7,  20,8])          # X step 4  (SE)
        ckt.append("TICK")
        ckt.append("H", x_ancillas)
        ckt.append("TICK")

    # so from here we build our circuit

    # reset everything to |0>
    # Z-ancillas start |0>  -> first Z-stabiliser measurement is deterministic

    # X-ancillas start |0>  -> extract_syndromes immediately does H on them before
    # any CNOT, putting them in |+>  -> first X-stabiliser measurement is also
    # deterministic.  DO NOT add an extra H here or it double-cancels that's why 
    # the cnots were all fucked up
    circuit.append("R", range(25))

    # so at initialization, we gotta stabilize everything
    # only detectors are added here (9..24 indices range)
    # X measurements are recorded by MR and used as reference in cycle,
    # but their round-0 values are kept as a reference baseline (no detector).
    z_active_i = [a - 9 for a in z_ancillas]   # [4, 7, 12, 15] = ancs 13,16,21,24
    extract_syndromes(circuit)
    circuit.append("MR", range(9, 25), p)
    for i in z_active_i:
        circuit.append("DETECTOR", [stim.target_rec(-16 + i)], ancilla_coords[9 + i])
    circuit.append("TICK")

    # this repeats the cycle
    cycle = stim.Circuit()
    cycle.append("DEPOLARIZE1", range(9), p)
    extract_syndromes(cycle)
    cycle.append("MR", range(9, 25), p)

    # XOR detectors for all 8 active ancillas.
    #  XOR (cycle-1 \otimes round-0) is well-defined because round-0
    # X measurements are deterministic
    #   ancillas start |0> then H, then CNOTs
    for i in active_i:
        cycle.append(
            "DETECTOR",
            [stim.target_rec(-16 + i), stim.target_rec(-32 + i)],
            ancilla_coords[9 + i],
        )
    cycle.append("TICK")

    # so build up our rounds
    circuit += cycle * (rounds - 1)
    circuit.append("M", range(9))
    # Logical Z = Z_0 \otimes Z_1 \otimes Z_2
    circuit.append(
        "OBSERVABLE_INCLUDE",
        [stim.target_rec(-9), stim.target_rec(-8), stim.target_rec(-7)],
        0,
    )
    print(f"Debugging: At this point, the surface code circuit is built for {rounds} rounds!")
    return circuit

# okay first try part 5
# so here, we use pymatching to run simulation
print("Running surface code (totally not going to fail whatsoever)")

c = build_surface_code(rounds=3, p=0.001)

# build error model + matching graph
error_model = c.detector_error_model(decompose_errors=True)
matching = pymatching.Matching.from_detector_error_model(error_model)

# sample syndrome data + logical observables
sampler = c.compile_detector_sampler()
syndromes, observables = sampler.sample(shots=10000, separate_observables=True)

# decode + get results
predictions = matching.decode_batch(syndromes)
logical_errors = np.any(predictions != observables, axis=1)
error_amt = np.sum(logical_errors)

print(f"There are {error_amt} errors out of 10000 shots.")
print(f"Corrected Failure Rate: {(error_amt / 10000) * 100:.4f}%")

# first try part #
# Test 1: 501/1024: was resetting data qubits each round
# Test 2: 540/1024: CNOTs in wrong positions
# Test 3: 509/1024: still ~50/50; decoder added but stabilisers invalid
# Test 4: ValueError all X-faces + all Z-edges anti-commute
# for some reason data qubits appeared 2 to 3 times per tick
# Test 5: YOOOO 1 ERROR LESGOO
# Test 6: 15/10000: expanded the # of shots
