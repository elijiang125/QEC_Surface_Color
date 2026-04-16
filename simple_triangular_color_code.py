
"""simple_triangular_color_code.py

First effort dist-3 triangular color-code (7-qubit Steane code) baseline

Done so far:
1. Define 7-qubit CSS color code using 3 X stabilizers and 3 Z stabilizers
2. Builds an exact min-weight lookup-table decoder for the X and Z sectors
3. Checks all single-qubit X, Y, Z errors are corrected successfully
4. Approxes the logical failure rate under "perfect syndrome extraction" with i.i.d. data-qubit depolarizing noise

TO DO:
-repeated syndrome extraction rounds
-measurement noise/detector events (im not super familiar on how to put these kinds of things in)
-MWPM decoder
-some kind of clear benchmark against the surface-code circuit (also open to ideas here lol)

TLDR: simple working baseline before moving to the full repeated-round color-code simulation.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


BitArray = np.ndarray
Syndrome = Tuple[int, int, int]


@dataclass
class DecodeResult:
    x_error: BitArray
    z_error: BitArray
    x_syndrome: Syndrome
    z_syndrome: Syndrome
    x_correction: BitArray
    z_correction: BitArray
    x_residual: BitArray
    z_residual: BitArray
    success: bool


class SimpleTriangularColorCode:
    """Distance-3 triangular color code in its 7-qubit Steane-code form."""

    def __init__(self) -> None:
        self.n = 7
        self.H = np.array(
            [
                [1, 1, 1, 1, 0, 0, 0],
                [1, 1, 0, 0, 1, 1, 0],
                [1, 0, 1, 0, 1, 0, 1],
            ],
            dtype=np.uint8,
        )

        self.x_stabilizers: List[BitArray] = [row.copy() for row in self.H]
        self.z_stabilizers: List[BitArray] = [row.copy() for row in self.H]
        self.logical_x = np.ones(self.n, dtype=np.uint8)
        self.logical_z = np.ones(self.n, dtype=np.uint8)

        self._stabilizer_span = self._row_span(self.H)
        self.x_lookup = self._build_sector_lookup()
        self.z_lookup = self._build_sector_lookup()


    ### Basic linear-algebra helpers over GF(2)

    @staticmethod
    def _bits_to_tuple(bits: BitArray) -> Tuple[int, ...]:
        return tuple(int(b) for b in bits.tolist())

    @staticmethod
    def _hamming_weight(bits: BitArray) -> int:
        return int(np.sum(bits))

    def _row_span(self, mat: BitArray) -> set[Tuple[int, ...]]:
        """Return the set of all binary linear combinations of the rows."""
        num_rows = mat.shape[0]
        span = set()
        for mask in range(1 << num_rows):
            vec = np.zeros(mat.shape[1], dtype=np.uint8)
            for i in range(num_rows):
                if (mask >> i) & 1:
                    vec ^= mat[i]
            span.add(self._bits_to_tuple(vec))
        return span

    def _syndrome(self, bits: BitArray) -> Syndrome:
        """Compute H * bits mod 2."""
        syn = (self.H @ bits) % 2
        return tuple(int(s) for s in syn.tolist())

    def _build_sector_lookup(self) -> Dict[Syndrome, BitArray]:
        """Exact minimum-weight lookup table for one CSS sector.

        For each of the 2^3 possible syndromes, loop over all 2^7 binary error
        patterns and keep one min-weight representative.
        """
        lookup: Dict[Syndrome, Tuple[int, BitArray]] = {}
        for pattern in itertools.product([0, 1], repeat=self.n):
            bits = np.array(pattern, dtype=np.uint8)
            syn = self._syndrome(bits)
            weight = self._hamming_weight(bits)

            prev = lookup.get(syn)
            if prev is None:
                lookup[syn] = (weight, bits.copy())
                continue

            prev_weight, prev_bits = prev
            if weight < prev_weight:
                lookup[syn] = (weight, bits.copy())
            elif weight == prev_weight and self._bits_to_tuple(bits) < self._bits_to_tuple(prev_bits):
                lookup[syn] = (weight, bits.copy())

        return {syn: bits for syn, (_, bits) in lookup.items()}

    ### Error representation and decoding

    def pauli_to_xz(self, pauli: str) -> Tuple[BitArray, BitArray]:
        """Convert a 7-character Pauli string into X/Z bit-vectors.

        Ex: 'IIXIZYI'
        """
        pauli = pauli.strip().upper()
        if len(pauli) != self.n:
            raise ValueError(f"Expected a Pauli string of length {self.n}, got {len(pauli)}.")

        x = np.zeros(self.n, dtype=np.uint8)
        z = np.zeros(self.n, dtype=np.uint8)
        for i, p in enumerate(pauli):
            if p not in {"I", "X", "Y", "Z"}:
                raise ValueError(f"Invalid Pauli {p!r} at position {i}.")
            if p in {"X", "Y"}:
                x[i] = 1
            if p in {"Z", "Y"}:
                z[i] = 1
        return x, z

    def decode_from_xz(self, x_error: BitArray, z_error: BitArray) -> DecodeResult:
        """Decode a Pauli error represented by X and Z bit-vectors."""
        x_error = np.array(x_error, dtype=np.uint8)
        z_error = np.array(z_error, dtype=np.uint8)

        # CSS decoding:
        # X errors use the Z stabilizer syndrome, so we have same H here
        # Z errors use the X stabilizer syndrome, again we have same H here
        x_syndrome = self._syndrome(x_error)
        z_syndrome = self._syndrome(z_error)

        x_correction = self.x_lookup[x_syndrome].copy()
        z_correction = self.z_lookup[z_syndrome].copy()

        x_residual = x_error ^ x_correction
        z_residual = z_error ^ z_correction

        # Ok so after ideal decoding, each residual should have 0 syndrome
        if self._syndrome(x_residual) != (0, 0, 0):
            raise RuntimeError("Decoded X residual still has nonzero syndrome.")
        if self._syndrome(z_residual) != (0, 0, 0):
            raise RuntimeError("Decoded Z residual still has nonzero syndrome.")

        # If we succeeded then the residual lies in the stabilizer span in each sector
        # If the residual has zero syndrome but its not in the stabilizer span,
        # then it differs from identity by a logical operator
        success = (
            self._bits_to_tuple(x_residual) in self._stabilizer_span
            and self._bits_to_tuple(z_residual) in self._stabilizer_span
        )

        return DecodeResult(
            x_error=x_error,
            z_error=z_error,
            x_syndrome=x_syndrome,
            z_syndrome=z_syndrome,
            x_correction=x_correction,
            z_correction=z_correction,
            x_residual=x_residual,
            z_residual=z_residual,
            success=success,
        )

    def decode_pauli_string(self, pauli: str) -> DecodeResult:
        x_error, z_error = self.pauli_to_xz(pauli)
        return self.decode_from_xz(x_error, z_error)

    ### Validation/Monte Carlo

    def exhaustive_single_qubit_test(self, verbose: bool = True) -> bool:
        """Check all 21 single-qubit X/Y/Z errors."""
        failures: List[str] = []
        for q in range(self.n):
            for p in ("X", "Y", "Z"):
                pauli = ["I"] * self.n
                pauli[q] = p
                pauli_str = "".join(pauli)
                result = self.decode_pauli_string(pauli_str)
                if not result.success:
                    failures.append(pauli_str)

        if verbose:
            if failures:
                print("Single-qubit validation FAILED for:")
                for pauli_str in failures:
                    print("  ", pauli_str)
            else:
                print("Single-qubit validation PASSED for all 21 X/Y/Z errors.")

        return len(failures) == 0

    def sample_depolarizing_error(self, p: float, rng: np.random.Generator) -> Tuple[BitArray, BitArray]:
        """Sample iid depolarizing noise on the 7 data qubits.

        On each qubit:
            I with probability 1 - p
            X, Y, Z each with probability p/3
        """
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must lie in [0, 1].")

        x = np.zeros(self.n, dtype=np.uint8)
        z = np.zeros(self.n, dtype=np.uint8)

        draws = rng.random(self.n)
        for i, r in enumerate(draws):
            if r >= p:
                continue
            local = rng.integers(0, 3)  # note: 0 is X, 1 is Y, 2 is Z here
            if local == 0:
                x[i] = 1
            elif local == 1:
                x[i] = 1
                z[i] = 1
            else:
                z[i] = 1
        return x, z

    def monte_carlo(self, p: float, shots: int = 10000, seed: int | None = None) -> float:
        """Estimate logical failure rate under perfect syndrome extraction."""
        if shots <= 0:
            raise ValueError("shots must be positive.")

        rng = np.random.default_rng(seed)
        logical_failures = 0
        for _ in range(shots):
            x_error, z_error = self.sample_depolarizing_error(p, rng)
            result = self.decode_from_xz(x_error, z_error)
            if not result.success:
                logical_failures += 1
        return logical_failures / shots

    ### Display helpers

    def print_code_summary(self) -> None:
        print("=== Simple distance-3 triangular color code (7-qubit Steane form) ===")
        print("X stabilizers:")
        for i, row in enumerate(self.x_stabilizers):
            support = [j for j, b in enumerate(row.tolist()) if b]
            print(f"  X{i}: support {support}")
        print("Z stabilizers:")
        for i, row in enumerate(self.z_stabilizers):
            support = [j for j, b in enumerate(row.tolist()) if b]
            print(f"  Z{i}: support {support}")
        print(f"Logical X support: {[i for i in range(self.n)]}")
        print(f"Logical Z support: {[i for i in range(self.n)]}")
        print()

    def print_lookup_table(self) -> None:
        print("=== Minimum-weight lookup table for one sector ===")
        for syn in sorted(self.x_lookup.keys()):
            corr = self.x_lookup[syn]
            support = [i for i, b in enumerate(corr.tolist()) if b]
            print(f"  syndrome {syn} -> correction support {support}")


def main() -> None:
    code = SimpleTriangularColorCode()
    code.print_code_summary()
    code.print_lookup_table()
    print()

    ok = code.exhaustive_single_qubit_test(verbose=True)
    if not ok:
        raise SystemExit("Single-qubit test failed; fix decoder before moving on.")

    print()
    for p in [1e-3, 3e-3, 1e-2]:
        failure_rate = code.monte_carlo(p=p, shots=20000, seed=1234)
        print(f"p = {p:.4g} -> estimated logical failure rate = {failure_rate:.6f}")


if __name__ == "__main__":
    main()
