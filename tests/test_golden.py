"""
Golden tests — fixed seed=42, exact floating-point match.

These tests pin the output of each generator so that any change to the
numerical recipe (formula, warmup length, integration scheme, subsample
factor, etc.) is immediately caught.  They are NOT tolerance tests; they
use assert_array_equal (exact bit match) on the scalar values below.

Golden values were recorded with seed=42 on 2026-04-28.
"""

import numpy as np
import pytest

from rc_bench.core.data_provider import (
    generate_lorenz63,
    generate_mackey_glass,
    generate_narma10,
    generate_narma30,
)

# ---------------------------------------------------------------------------
# Golden constants
# ---------------------------------------------------------------------------

NARMA10_GOLDEN = {
    # (index, component): expected value
    # X is input u, y is NARMA output
    ("X", 10,   0): 0.18539901211629062,
    ("y", 10,    ): 0.23071709530705925,
    ("X", 100,  0): 0.4542903453538035,
    ("y", 100,   ): 0.2662891121790265,
    ("X", 500,  0): 0.36835284424111014,
    ("y", 500,   ): 0.3988298668487473,
    ("X", 1999, 0): 0.26003595613074404,
    ("y", 1999,  ): 0.35044842508463836,
}

NARMA30_GOLDEN = {
    ("X", 30,   0): 0.37238107795390857,
    ("y", 30,    ): 0.2982437008081814,
    ("X", 100,  0): 0.4542903453538035,
    ("y", 100,   ): 0.28384563137881813,
    ("X", 500,  0): 0.36835284424111014,
    ("y", 500,   ): 0.34698762583516496,
    ("X", 2999, 0): 0.04903991354546622,
    ("y", 2999,  ): 0.3139383321614483,
}

MACKEY_GLASS_GOLDEN = {
    ("X", 0,    0): 1.050804808111234,
    ("y", 0,     ): 1.0341360035823108,
    ("X", 100,  0): 1.1498047732940186,
    ("y", 100,   ): 1.1612349079819086,
    ("X", 500,  0): 0.9034723200018729,
    ("y", 500,   ): 0.8874638119631225,
    ("X", 4999, 0): 0.6002719992635708,
    ("y", 4999,  ): 0.5624946856135634,
}

LORENZ63_GOLDEN = {
    ("X", 0,    0): -7.668100443662575,
    ("y", 0,     ): -4.244981761028991,
    ("X", 100,  0): 1.2636170097956647,
    ("y", 100,   ): 2.3167660742154883,
    ("X", 500,  0): 5.1605899290612935,
    ("y", 500,   ): 9.087715649279218,
    ("X", 4999, 0): -3.655785374195358,
    ("y", 4999,  ): -7.4644959649252955,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_golden(gen, T, golden_dict):
    X, y = gen(T, seed=42)
    for key, expected in golden_dict.items():
        kind, idx = key[0], key[1]
        if kind == "X":
            col = key[2]
            actual = X[idx, col]
        else:
            actual = y[idx]
        np.testing.assert_array_equal(
            actual, expected,
            err_msg=f"Golden mismatch at {key}: got {actual!r}, expected {expected!r}",
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGoldenNarma10:
    def test_golden_values(self):
        _check_golden(generate_narma10, T=2000, golden_dict=NARMA10_GOLDEN)

    def test_total_length(self):
        X, y = generate_narma10(2000, seed=42)
        assert X.shape == (2000, 1)
        assert y.shape == (2000,)


class TestGoldenNarma30:
    def test_golden_values(self):
        _check_golden(generate_narma30, T=3000, golden_dict=NARMA30_GOLDEN)

    def test_total_length(self):
        X, y = generate_narma30(3000, seed=42)
        assert X.shape == (3000, 1)
        assert y.shape == (3000,)


class TestGoldenMackeyGlass:
    def test_golden_values(self):
        _check_golden(generate_mackey_glass, T=5000, golden_dict=MACKEY_GLASS_GOLDEN)

    def test_total_length(self):
        X, y = generate_mackey_glass(5000, seed=42)
        assert X.shape == (5000, 1)
        assert y.shape == (5000,)


class TestGoldenLorenz63:
    def test_golden_values(self):
        _check_golden(generate_lorenz63, T=5000, golden_dict=LORENZ63_GOLDEN)

    def test_total_length(self):
        X, y = generate_lorenz63(5000, seed=42)
        assert X.shape == (5000, 1)
        assert y.shape == (5000,)
