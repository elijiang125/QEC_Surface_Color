"""stim_triangular_color_code.py

A first Stim-style version of the distance-3 triangular color code
(the 7-qubit Steane code).

Done so far:  
1. Builds the triangular color code as a Stim circuit with 7 data qubits, 
   3 Z-check ancillas, 3 X-check ancillas, repeated syndrome-extraction rounds.
2. Follows patter of current surface-code script: initialize once, 
   run repeated rounds, then do a final data-qubit readout.
3. Keeps a small exact decoder for the color code, and uses the final Z-check
   syndrome plus the final data readout to estimate a logical-Z memory failure
   rate.
4. Have option to insert Stim DETECTOR and OBSERVABLE_INCLUDE instructions so the
   circuit is ready for a later detector-based decoder.

Note: 
The decoder implemented here assumes the repeated syndrome extraction is present,
but the actual decoding uses only the final round syndrome,
so measurement noise should be kept at p_meas = 0 for now.

TO DO: 
1. use detector events from round-to-round syndrome changes
2. add a measurement-noise-aware decoder
3. possibly move to a projection/restriction decoder or another color-code
  decoder once the simple version is trusted
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


BitArray = np.ndarray
Syndrome = Tuple[int, int, int]


@dataclass
class ShotResults:
    final_z_syndrome: BitArray
    observed_logical_z: BitArray
    predicted_logical_z_after_correction: BitArray
    logical_failures: BitArray


class TriangularColorCode:
    """Distance-3 triangular color code in 7-qubit Steane form.

    Stabilizer supports are encoded by the 3x7 binary matrix H. The same H is
    used for X and Z stabilizers because this code is self-dual.

    The stabilizers are
        X0 X1 X2 X3
        X0 X1 X4 X5
        X0 X2 X4 X6

        Z0 Z1 Z2 Z3
        Z0 Z1 Z4 Z5
        Z0 Z2 Z4 Z6

    A convenient logical choice is
        X_L = X0 X1 X2 X3 X4 X5 X6
        Z_L = Z0 Z1 Z2 Z3 Z4 Z5 Z6
    """

    def __init__(self) -> None:
        self.n_data = 7
        self.H = np.array(
            [
                [1, 1, 1, 1, 0, 0, 0],
                [1, 1, 0, 0, 1, 1, 0],
                [1, 0, 1, 0, 1, 0, 1],
            ],
            dtype=np.uint8,
        )
        self.logical_z = np.ones(self.n_data, dtype=np.uint8)
        self.logical_x = np.ones(self.n_data, dtype=np.uint8)
        self.x_lookup = self._build_sector_lookup()
        self.z_lookup = self._build_sector_lookup()

        # Stabilizer support lists, convenient for circuit construction.
        self.supports: List[List[int]] = [
            [0, 1, 2, 3],
            [0, 1, 4, 5],
            [0, 2, 4, 6],
        ]

    @staticmethod
    def _bits_to_tuple(bits: BitArray) -> Tuple[int, ...]:
        return tuple(int(b) for b in bits.tolist())

    @staticmethod
    def _hamming_weight(bits: BitArray) -> int:
        return int(np.sum(bits))

    def syndrome(self, bits: BitArray) -> Syndrome:
        syn = (self.H @ bits) % 2
        return tuple(int(s) for s in syn.tolist())

    def _build_sector_lookup(self) -> Dict[Syndrome, BitArray]:
        """Minimum-weight representative for each 3-bit syndrome."""
        lookup: Dict[Syndrome, Tuple[int, BitArray]] = {}
        for pattern in itertools.product([0, 1], repeat=self.n_data):
            bits = np.array(pattern, dtype=np.uint8)
            syn = self.syndrome(bits)
            wt = self._hamming_weight(bits)
            prev = lookup.get(syn)
            if prev is None or wt < prev[0] or (
                wt == prev[0] and self._bits_to_tuple(bits) < self._bits_to_tuple(prev[1])
            ):
                lookup[syn] = (wt, bits.copy())
        return {syn: bits for syn, (_, bits) in lookup.items()}

    def x_correction_from_z_syndrome(self, syn: Sequence[int]) -> BitArray:
        key = tuple(int(x) for x in syn)
        return self.x_lookup[key].copy()


def build_color_code_circuit(
    rounds: int,
    p_data: float = 0.001,
    p_meas: float = 0.0,
    add_detectors: bool = True,
):
    """Build a Stim circuit for the simple triangular color-code memory test.

    - One initialization round, repeated noisy rounds, final data measurement.
    - The current exact decoder only uses the final-round syndrome, so in this
      script measurement noise should be left at p_meas = 0
    """
    try:
        import stim
    except ImportError as exc:
        raise ImportError(
            "Stim is not installed in this environment. Install `stim` in the "
            "project environment before running this script."
        ) from exc

    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    if not (0.0 <= p_data <= 1.0):
        raise ValueError("p_data must be between 0 and 1")
    if not (0.0 <= p_meas <= 1.0):
        raise ValueError("p_meas must be between 0 and 1")

    code = TriangularColorCode()
    circuit = stim.Circuit()

    data = list(range(7))
    z_ancillas = [7, 8, 9]
    x_ancillas = [10, 11, 12]
    ancillas = z_ancillas + x_ancillas

    # Very simple visual layout. These coordinates are only for plotting/debug.
    data_coords = {
        0: [0.0, 0.0],
        1: [2.0, 0.0],
        2: [4.0, 0.0],
        3: [1.0, 1.7],
        4: [3.0, 1.7],
        5: [2.0, 3.4],
        6: [2.0, 5.1],
    }
    z_ancilla_coords = {
        7: [1.0, -0.9],
        8: [3.0, -0.9],
        9: [2.0, 2.3],
    }
    x_ancilla_coords = {
        10: [1.0, 0.8],
        11: [3.0, 0.8],
        12: [2.0, 4.2],
    }

    for q, coord in {**data_coords, **z_ancilla_coords, **x_ancilla_coords}.items():
        circuit.append("QUBIT_COORDS", [q], coord)

    # A 4-step edge-coloring schedule so no data or ancilla is touched more than
    # once per TICK within the Z-check layer or within the X-check layer.
    #
    # Z-check ancillas:
    #   7 measures Z0 Z1 Z2 Z3
    #   8 measures Z0 Z1 Z4 Z5
    #   9 measures Z0 Z2 Z4 Z6
    #
    # X-check ancillas:
    #   10 measures X0 X1 X2 X3
    #   11 measures X0 X1 X4 X5
    #   12 measures X0 X2 X4 X6
    #
    z_schedule = [
        [(0, 7), (1, 8), (2, 9)],
        [(1, 7), (0, 8), (4, 9)],
        [(2, 7), (4, 8), (0, 9)],
        [(3, 7), (5, 8), (6, 9)],
    ]
    x_schedule = [
        [(10, 0), (11, 1), (12, 2)],
        [(10, 1), (11, 0), (12, 4)],
        [(10, 2), (11, 4), (12, 0)],
        [(10, 3), (11, 5), (12, 6)],
    ]

    def append_measure_round(ckt):
        # Measure Z stabilizers: data control goes to Z ancilla target
        for step in z_schedule:
            ckt.append("TICK")
            flat_targets: List[int] = []
            for d, a in step:
                flat_targets.extend([d, a])
            ckt.append("CNOT", flat_targets)
        ckt.append("TICK")

        # Measure X stabilizers: put X ancillas into |+>, then ancilla control goes to data target,
        # then rotate back before measurement
        ckt.append("H", x_ancillas)
        for step in x_schedule:
            ckt.append("TICK")
            flat_targets = []
            for a, d in step:
                flat_targets.extend([a, d])
            ckt.append("CNOT", flat_targets)
        ckt.append("TICK")
        ckt.append("H", x_ancillas)
        ckt.append("TICK")

    # Reset all qubits to |0>, this prepares the Z sector
    # The first X-stabilizer measurement projects the state into one X-syndrome sector,
    # later round-to-round changes can be tracked with detectors
    circuit.append("R", data + ancillas)

    # Initial projection round
    append_measure_round(circuit)
    if p_meas > 0:
        circuit.append("MR", ancillas, p_meas)
    else:
        circuit.append("MR", ancillas)

    if add_detectors:
        # Initially, Z stabilizers should be +1 because all data start in |0>
        # So we create absolute detectors for the first three measured ancillas (the Z checks),
        # but not for the X checks
        for i, anc in enumerate(z_ancillas):
            circuit.append("DETECTOR", [stim.target_rec(-6 + i)], z_ancilla_coords[anc])
    circuit.append("TICK")

    # Repeated memory cycles
    cycle = stim.Circuit()
    if p_data > 0:
        cycle.append("DEPOLARIZE1", data, p_data)
    append_measure_round(cycle)
    if p_meas > 0:
        cycle.append("MR", ancillas, p_meas)
    else:
        cycle.append("MR", ancillas)

    if add_detectors:
        # Use parity between successive rounds as future decoder inputs
        for i, anc in enumerate(ancillas):
            coords = z_ancilla_coords.get(anc, x_ancilla_coords.get(anc))
            cycle.append(
                "DETECTOR",
                [stim.target_rec(-6 + i), stim.target_rec(-12 + i)],
                coords,
            )
    cycle.append("TICK")

    circuit += cycle * (rounds - 1)

    # Final data readout in the Z basis.
    circuit.append("M", data)
    if add_detectors:
        logical_targets = [stim.target_rec(-(7 - i)) for i in range(7)]
        circuit.append("OBSERVABLE_INCLUDE", logical_targets, 0)

    return circuit


def parse_measurement_samples(samples: np.ndarray, rounds: int) -> Tuple[np.ndarray, np.ndarray]:
    """Parse raw Stim measurement samples.

    Returns ancilla_rounds: the six ancilla bits are ordered as
        [Z0, Z1, Z2, X0, X1, X2] for each round.
    and final_data: the Z-basis measurements on the data qubits.
    """
    shots, total_bits = samples.shape
    expected_bits = 6 * rounds + 7
    if total_bits != expected_bits:
        raise ValueError(
            f"Expected {expected_bits} measurement bits for rounds={rounds}, got {total_bits}."
        )

    ancilla_bits = samples[:, : 6 * rounds].reshape(shots, rounds, 6)
    final_data = samples[:, 6 * rounds :]
    return ancilla_bits, final_data


def decode_final_round_z_memory(
    ancilla_rounds: np.ndarray,
    final_data: np.ndarray,
    code: TriangularColorCode | None = None,
) -> ShotResults:
    """Decode logical-Z memory shots using the final Z-check syndrome only.

    Current simple decoder. It is valid when measurement noise is absent
    because the final round Z-check outcomes equal the syndrome of the
    accumulated X error.

    For each shot:
      1. read the final Z-check syndrome (3 bits)
      2. choose the min weight X correction for that syndrome
      3. compute the observed logical Z from the final data readout
      4. flip that logical value by the effect of the chosen X correction
      5. declare failure if the corrected logical Z is 1 instead of 0
    """
    if code is None:
        code = TriangularColorCode()

    final_z_syndrome = np.asarray(ancilla_rounds[:, -1, :3], dtype=np.uint8)
    final_data = np.asarray(final_data, dtype=np.uint8)

    observed_logical_z = (final_data @ code.logical_z) % 2

    x_corrections = np.zeros((final_z_syndrome.shape[0], code.n_data), dtype=np.uint8)
    for shot in range(final_z_syndrome.shape[0]):
        syn = tuple(int(x) for x in final_z_syndrome[shot].tolist())
        x_corrections[shot] = code.x_correction_from_z_syndrome(syn)

    logical_flip_from_correction = (x_corrections @ code.logical_z) % 2
    predicted_logical_z_after_correction = observed_logical_z ^ logical_flip_from_correction
    logical_failures = predicted_logical_z_after_correction.astype(bool)

    return ShotResults(
        final_z_syndrome=final_z_syndrome,
        observed_logical_z=observed_logical_z,
        predicted_logical_z_after_correction=predicted_logical_z_after_correction,
        logical_failures=logical_failures,
    )


def run_stim_z_memory_experiment(
    rounds: int = 3,
    p_data: float = 0.001,
    p_meas: float = 0.0,
    shots: int = 10_000,
    add_detectors: bool = True,
) -> Dict[str, object]:
    """Run the simple Stim-style logical-Z memory experiment.

    Compare against the current surface-code script.
    The decoder is still the simple exact color-code decoder, so set p_meas=0 for now.
    """
    try:
        import stim
    except ImportError as exc:
        raise ImportError(
            "Stim is not installed in this environment. Install `stim` first."
        ) from exc

    if p_meas != 0.0:
        print(
            "WARNING: the current decoder ignores measurement history and is "
            "not measurement-noise-aware. Use p_meas=0 for now."
        )

    code = TriangularColorCode()
    circuit = build_color_code_circuit(
        rounds=rounds,
        p_data=p_data,
        p_meas=p_meas,
        add_detectors=add_detectors,
    )

    sampler = circuit.compile_sampler()
    samples = sampler.sample(shots=shots)
    ancilla_rounds, final_data = parse_measurement_samples(samples, rounds)
    decoded = decode_final_round_z_memory(ancilla_rounds, final_data, code=code)

    num_failures = int(np.sum(decoded.logical_failures))
    failure_rate = num_failures / shots

    return {
        "circuit": circuit,
        "shots": shots,
        "rounds": rounds,
        "p_data": p_data,
        "p_meas": p_meas,
        "num_failures": num_failures,
        "failure_rate": failure_rate,
        "decoded": decoded,
    }


if __name__ == "__main__":
    print("Running simple Stim-style triangular color-code memory experiment...")
    results = run_stim_z_memory_experiment(
        rounds=3,
        p_data=0.01,
        p_meas=0.0,
        shots=100_000,
        add_detectors=True,
    )
    print(f"Rounds: {results['rounds']}")
    print(f"Data noise: {results['p_data']}")
    print(f"Measurement noise: {results['p_meas']}")
    print(f"Failures: {results['num_failures']} / {results['shots']}")
    print(f"Estimated logical-Z failure rate: {100 * results['failure_rate']:.4f}%")
