"""stim_triangular_color_code_history_decoder.py

Triangular color-code logical-Z memory experiment with a full round-to-round
history decoder.

This version upgrades the color-code decoder so that it can use the entire 
Z-syndrome measurement history across all rounds instead of only the final syndrome. 
This means it can now handle independent ancilla measurement errors.

Decoder model: 
Kdpg the same simplified phenomenological noise model already used in the
surface-code baseline:
    - after each completed round except the first, each data qubit undergoes Stim's DEPOLARIZE1(p_data)
    - each ancilla readout is flipped independently with probability p_meas

For logical-Z memory, only the X component of the data noise matters. Under
DEPOLARIZE1(p_data), a data qubit has an X component with probability

    q_x = 2 p_data / 3,

because X and Y both flip Z-basis outcomes.

Decoder here is an exact Viterbi style maximum-likelihood space time decoder over:
    - the hidden 7-bit accumulated X-error state of the data block
    - data-error transitions between rounds
    - and measurement flips on the three Z-check ancillas

This is practical because the distance-3 triangular code is tiny: there are
only 2^7 = 128 hidden X-error states.

Cleanest comparison against the current surface-code script is:
noise model: phenomenological memory noise
    - DEPOLARIZE1(p) on data between rounds
    - MR(..., p) on ancilla measurements
    - no noisy CNOT model yet

rounds: benchmark 3, 5, 7

shots: 100_000 per point for final plots (10_000-20_000 for debugging)


logical failure definition: initialize in |0_L>, decode the full Z-check
history, and count failure when the decoder-corrected logical-Z outcome is 1.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


BitArray = np.ndarray
Syndrome = Tuple[int, int, int]


@dataclass
class ShotResults:
    z_measurement_history: BitArray
    observed_logical_z: BitArray
    estimated_final_x_error: BitArray
    estimated_final_logical_flip: BitArray
    corrected_logical_z: BitArray
    logical_failures: BitArray


class TriangularColorCode:
    """Distance-3 triangular color code in 7-qubit Steane form.

    We use the self-dual 3x7 parity-check matrix

        H = [[1 1 1 1 0 0 0],
             [1 1 0 0 1 1 0],
             [1 0 1 0 1 0 1]]

    for both X- and Z-type stabilizers.
    """

    def __init__(self) -> None:
        self.n_data = 7
        self.n_checks = 3
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

        # supports convenient for building circuit
        self.supports: List[List[int]] = [
            [0, 1, 2, 3],
            [0, 1, 4, 5],
            [0, 2, 4, 6],
        ]

    def syndrome(self, x_bits: BitArray) -> Syndrome:
        syn = (self.H @ x_bits) % 2
        return tuple(int(s) for s in syn.tolist())

    def logical_z_flip(self, x_bits: BitArray) -> int:
        return int((x_bits @ self.logical_z) % 2)


class ExactSpacetimeZDecoder:
    """Exact full-history decoder for logical-Z memory.

    Hidden state:
    The hidden state at round t is the accumulated 7-bit X-error pattern on the
    data qubits. There are 128 possible states.

    Observation model:
    At each round, observe the three Z-check ancilla readouts. Those equal the
    true Z syndrome of the current X-error state, corrupted by indp measurement 
    flips with probability p_meas.

    Transition model:
    Between consecutive rounds, each data qubit independently acquires an X
    component with probability q_x = 2 p_data / 3.

    Decoding rule:
    Run exact Viterbi dynamic program over all hidden states to find the
    maximum likelihood fault history consistent with the measured syndrome
    history. The inferred final X-error state becomes the applied correction.
    """

    def __init__(self, code: TriangularColorCode, rounds: int, p_data: float, p_meas: float) -> None:
        if rounds < 1:
            raise ValueError("rounds must be at least 1")
        if not (0.0 <= p_data <= 1.0):
            raise ValueError("p_data must be between 0 and 1")
        if not (0.0 <= p_meas <= 1.0):
            raise ValueError("p_meas must be between 0 and 1")

        self.code = code
        self.rounds = rounds
        self.p_data = p_data
        self.p_meas = p_meas
        self.q_x = 2.0 * p_data / 3.0

        self.states = np.array(list(itertools.product([0, 1], repeat=code.n_data)), dtype=np.uint8)
        self.n_states = self.states.shape[0]
        self.zero_state_index = 0

        self.state_tuples: List[Tuple[int, ...]] = [tuple(int(b) for b in row.tolist()) for row in self.states]
        self.state_index: Dict[Tuple[int, ...], int] = {
            state: i for i, state in enumerate(self.state_tuples)
        }
        self.state_syndromes = np.array([code.syndrome(state) for state in self.states], dtype=np.uint8)
        self.state_logical_flip = np.array(
            [code.logical_z_flip(state) for state in self.states], dtype=np.uint8
        )

        self.transition_log_prob = self._build_transition_log_prob()

    @staticmethod
    def _log_bernoulli_vector_prob(num_bits: int, flips: int, p_flip: float) -> float:
        if not (0 <= flips <= num_bits):
            raise ValueError("flips must lie between 0 and num_bits")
        if p_flip == 0.0:
            return 0.0 if flips == 0 else -math.inf
        if p_flip == 1.0:
            return 0.0 if flips == num_bits else -math.inf
        return flips * math.log(p_flip) + (num_bits - flips) * math.log(1.0 - p_flip)

    def _build_transition_log_prob(self) -> np.ndarray:
        trans = np.full((self.n_states, self.n_states), -math.inf, dtype=float)
        for i in range(self.n_states):
            delta = self.states[i] ^ self.states
            weights = np.sum(delta, axis=1)
            trans[i, :] = np.array(
                [self._log_bernoulli_vector_prob(self.code.n_data, int(w), self.q_x) for w in weights],
                dtype=float,
            )
        return trans

    def _observation_log_prob_vector(self, observed_syndrome: BitArray) -> np.ndarray:
        observed = np.asarray(observed_syndrome, dtype=np.uint8)
        if observed.shape != (self.code.n_checks,):
            raise ValueError(
                f"Expected observed_syndrome shape ({self.code.n_checks},), got {observed.shape}."
            )
        hamming = np.sum(self.state_syndromes ^ observed[None, :], axis=1)
        return np.array(
            [
                self._log_bernoulli_vector_prob(self.code.n_checks, int(h), self.p_meas)
                for h in hamming
            ],
            dtype=float,
        )

    def decode_batch(self, z_measurement_history: np.ndarray, observed_logical_z: np.ndarray) -> ShotResults:
        history = np.asarray(z_measurement_history, dtype=np.uint8)
        logical = np.asarray(observed_logical_z, dtype=np.uint8)

        if history.ndim != 3 or history.shape[1:] != (self.rounds, self.code.n_checks):
            raise ValueError(
                f"Expected z_measurement_history shape (shots, {self.rounds}, {self.code.n_checks}), "
                f"got {history.shape}."
            )
        if logical.shape != (history.shape[0],):
            raise ValueError(
                f"Expected observed_logical_z shape ({history.shape[0]},), got {logical.shape}."
            )

        shots = history.shape[0]
        estimated_final_x_error = np.zeros((shots, self.code.n_data), dtype=np.uint8)
        estimated_final_logical_flip = np.zeros(shots, dtype=np.uint8)
        corrected_logical_z = np.zeros(shots, dtype=np.uint8)

        for shot in range(shots):
            # Round 1: no data noise has occurred yet, so the hidden state is
            # definitely the zero state. Only measurement error can affect the
            # first observed syndrome.
            obs0 = self._observation_log_prob_vector(history[shot, 0])
            alpha = np.full(self.n_states, -math.inf, dtype=float)
            alpha[self.zero_state_index] = obs0[self.zero_state_index]

            backpointers = np.full((self.rounds, self.n_states), -1, dtype=np.int16)

            for t in range(1, self.rounds):
                obs_t = self._observation_log_prob_vector(history[shot, t])
                new_alpha = np.full(self.n_states, -math.inf, dtype=float)
                for s_new in range(self.n_states):
                    scores = alpha + self.transition_log_prob[:, s_new]
                    best_prev = int(np.argmax(scores))
                    new_alpha[s_new] = scores[best_prev] + obs_t[s_new]
                    backpointers[t, s_new] = best_prev
                alpha = new_alpha

            best_final_state = int(np.argmax(alpha))
            state_path = [best_final_state]
            current = best_final_state
            for t in range(self.rounds - 1, 0, -1):
                current = int(backpointers[t, current])
                state_path.append(current)
            state_path.reverse()

            final_state_index = state_path[-1]
            estimated_final_x_error[shot] = self.states[final_state_index]
            estimated_final_logical_flip[shot] = self.state_logical_flip[final_state_index]
            corrected_logical_z[shot] = logical[shot] ^ estimated_final_logical_flip[shot]

        logical_failures = corrected_logical_z.astype(bool)
        return ShotResults(
            z_measurement_history=history,
            observed_logical_z=logical,
            estimated_final_x_error=estimated_final_x_error,
            estimated_final_logical_flip=estimated_final_logical_flip,
            corrected_logical_z=corrected_logical_z,
            logical_failures=logical_failures,
        )


def build_color_code_circuit(
    rounds: int,
    p_data: float = 0.001,
    p_meas: float = 0.001,
    add_detectors: bool = True,
):
    """Build the Stim circuit for the triangular color-code memory experiment.

    Keeps the same overall simulation style as the current surface-code script:
      - reset once
      - do one initial projection round
      - repeat noisy syndrome rounds
      - finish with a final data readout

    Decoder below uses the full Z-check measurement history. The X-check
    layer is still included in the circuit because it is part of the code's full
    stabilizer measurement cycle and it matches the surface code workflow.
    """
    try:
        import stim
    except ImportError as exc:
        raise ImportError(
            "Stim is not installed in this environment. Install `stim` in the project environment first."
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
        for step in z_schedule:
            ckt.append("TICK")
            flat_targets: List[int] = []
            for d, a in step:
                flat_targets.extend([d, a])
            ckt.append("CNOT", flat_targets)
        ckt.append("TICK")

        ckt.append("H", x_ancillas)
        for step in x_schedule:
            ckt.append("TICK")
            flat_targets: List[int] = []
            for a, d in step:
                flat_targets.extend([a, d])
            ckt.append("CNOT", flat_targets)
        ckt.append("TICK")
        ckt.append("H", x_ancillas)
        ckt.append("TICK")

    circuit.append("R", data + ancillas)

    append_measure_round(circuit)
    if p_meas > 0:
        circuit.append("MR", ancillas, p_meas)
    else:
        circuit.append("MR", ancillas)

    if add_detectors:
        for i, anc in enumerate(z_ancillas):
            circuit.append("DETECTOR", [stim.target_rec(-6 + i)], z_ancilla_coords[anc])
    circuit.append("TICK")

    cycle = stim.Circuit()
    if p_data > 0:
        cycle.append("DEPOLARIZE1", data, p_data)
    append_measure_round(cycle)
    if p_meas > 0:
        cycle.append("MR", ancillas, p_meas)
    else:
        cycle.append("MR", ancillas)

    if add_detectors:
        for i, anc in enumerate(ancillas):
            coords = z_ancilla_coords.get(anc, x_ancilla_coords.get(anc))
            cycle.append(
                "DETECTOR",
                [stim.target_rec(-6 + i), stim.target_rec(-12 + i)],
                coords,
            )
    cycle.append("TICK")

    circuit += cycle * (rounds - 1)

    circuit.append("M", data)
    if add_detectors:
        logical_targets = [stim.target_rec(-(7 - i)) for i in range(7)]
        circuit.append("OBSERVABLE_INCLUDE", logical_targets, 0)

    return circuit


def parse_measurement_samples(samples: np.ndarray, rounds: int) -> Tuple[np.ndarray, np.ndarray]:
    """Parse raw Stim measurement samples.

    Returns:
    ancilla_rounds:
        Shape (shots, rounds, 6), ordered as [Z0, Z1, Z2, X0, X1, X2].
    final_data:
        Shape (shots, 7), the final Z-basis measurements of the data qubits.
    """
    shots, total_bits = samples.shape
    expected_bits = 6 * rounds + 7
    if total_bits != expected_bits:
        raise ValueError(
            f"Expected {expected_bits} measurement bits for rounds={rounds}, got {total_bits}."
        )

    ancilla_bits = np.asarray(samples[:, : 6 * rounds], dtype=np.uint8).reshape(shots, rounds, 6)
    final_data = np.asarray(samples[:, 6 * rounds :], dtype=np.uint8)
    return ancilla_bits, final_data


def run_stim_z_memory_experiment(
    rounds: int = 3,
    p_data: float = 0.001,
    p_meas: float = 0.001,
    shots: int = 10_000,
    add_detectors: bool = True,
) -> Dict[str, object]:
    """Run the full-history logical-Z memory experiment.

    Updated main driver. Unlike the earlier simple version, this
    function uses nonzero measurement noise because decoding is based on
    entire Z-check syndrome history, not just the final round.
    """
    circuit = build_color_code_circuit(
        rounds=rounds,
        p_data=p_data,
        p_meas=p_meas,
        add_detectors=add_detectors,
    )

    sampler = circuit.compile_sampler()
    samples = sampler.sample(shots=shots)
    ancilla_rounds, final_data = parse_measurement_samples(samples, rounds)

    z_history = ancilla_rounds[:, :, :3]
    observed_logical_z = (final_data @ TriangularColorCode().logical_z) % 2

    code = TriangularColorCode()
    decoder = ExactSpacetimeZDecoder(code=code, rounds=rounds, p_data=p_data, p_meas=p_meas)
    decoded = decoder.decode_batch(z_history, observed_logical_z)

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
        "z_history": z_history,
        "final_data": final_data,
    }


if __name__ == "__main__":
    print("Running triangular color-code memory experiment with full-history decoder...")
    results = run_stim_z_memory_experiment(
        rounds=7,
        p_data=0.001,
        p_meas=0.001,
        shots=100_000,
        add_detectors=True,
    )
    print(f"Rounds: {results['rounds']}")
    print(f"Data noise: {results['p_data']}")
    print(f"Measurement noise: {results['p_meas']}")
    print(f"Failures: {results['num_failures']} / {results['shots']}")
    print(f"Estimated logical-Z failure rate: {100 * results['failure_rate']:.4f}%")
