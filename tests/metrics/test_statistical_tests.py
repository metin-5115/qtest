"""Tests for ``qtest.metrics.statistical_tests``."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from qtest.metrics import auto_tolerance, chi_square_test, kolmogorov_smirnov_test

# ---------------------------------------------------------------------------
# chi_square_test
# ---------------------------------------------------------------------------


class TestChiSquareTest:
    def test_perfect_fit_has_high_pvalue(self) -> None:
        observed = {"00": 500, "11": 500}
        expected = {"00": 0.5, "11": 0.5}
        statistic, p_value = chi_square_test(observed, expected, shots=1000)
        assert statistic == pytest.approx(0.0)
        assert p_value > 0.99

    def test_strong_mismatch_has_small_pvalue(self) -> None:
        observed = {"00": 900, "11": 100}
        expected = {"00": 0.5, "11": 0.5}
        _, p_value = chi_square_test(observed, expected, shots=1000)
        assert p_value < 1e-3

    @pytest.mark.parametrize("shots", [100, 1_000, 10_000])
    def test_returns_two_floats_in_unit_interval(self, shots: int) -> None:
        observed = {"0": shots // 2, "1": shots // 2}
        expected = {"0": 0.5, "1": 0.5}
        stat, p = chi_square_test(observed, expected, shots=shots)
        assert isinstance(stat, float)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0

    def test_both_zero_bins_dropped(self) -> None:
        # Bin "11" has zero on both sides; it should be silently dropped,
        # leaving two valid bins ("00", "01") so the chi-square has df > 0.
        observed = {"00": 500, "01": 500}
        expected = {"00": 0.5, "01": 0.5, "11": 0.0}
        stat, p = chi_square_test(observed, expected, shots=1000)
        assert stat == pytest.approx(0.0)
        assert p == pytest.approx(1.0)

    def test_zero_expected_nonzero_observed_raises(self) -> None:
        observed = {"00": 100, "11": 100}
        expected = {"00": 1.0, "11": 0.0}
        with pytest.raises(ValueError):
            chi_square_test(observed, expected, shots=200)

    def test_non_positive_shots_raises(self) -> None:
        with pytest.raises(ValueError):
            chi_square_test({"0": 0}, {"0": 1.0}, shots=0)

    def test_empty_observed_raises(self) -> None:
        with pytest.raises(ValueError):
            chi_square_test({}, {"0": 1.0}, shots=10)

    def test_empty_expected_raises(self) -> None:
        with pytest.raises(ValueError):
            chi_square_test({"0": 10}, {}, shots=10)

    def test_negative_expected_probability_raises(self) -> None:
        with pytest.raises(ValueError):
            chi_square_test({"0": 5, "1": 5}, {"0": -0.1, "1": 1.1}, shots=10)

    def test_unnormalized_expected_raises(self) -> None:
        with pytest.raises(ValueError):
            chi_square_test({"0": 5, "1": 5}, {"0": 0.5, "1": 0.4}, shots=10)


# ---------------------------------------------------------------------------
# kolmogorov_smirnov_test
# ---------------------------------------------------------------------------


class TestKolmogorovSmirnovTest:
    def test_uniform_samples_against_uniform_cdf(self) -> None:
        rng = np.random.default_rng(7)
        samples = rng.uniform(0.0, 1.0, size=2000).tolist()

        # scipy.stats.kstest passes arrays to the CDF — use a vectorised form.
        stat, p = kolmogorov_smirnov_test(samples, stats.uniform.cdf)
        assert stat < 0.05
        assert p > 0.01

    def test_normal_samples_against_normal_cdf(self) -> None:
        rng = np.random.default_rng(11)
        samples = rng.standard_normal(2000).tolist()
        stat, p = kolmogorov_smirnov_test(samples, stats.norm.cdf)
        assert isinstance(stat, float)
        assert p > 0.01

    def test_uniform_samples_against_normal_cdf_rejects(self) -> None:
        rng = np.random.default_rng(13)
        samples = rng.uniform(-1.0, 1.0, size=2000).tolist()
        _, p = kolmogorov_smirnov_test(samples, stats.norm.cdf)
        assert p < 1e-3

    def test_empty_observed_raises(self) -> None:
        with pytest.raises(ValueError):
            kolmogorov_smirnov_test([], stats.norm.cdf)

    def test_non_callable_cdf_raises(self) -> None:
        with pytest.raises(ValueError):
            # Intentional misuse for the test.
            kolmogorov_smirnov_test([0.1, 0.2], "not a function")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# auto_tolerance
# ---------------------------------------------------------------------------


class TestAutoTolerance:
    @pytest.mark.parametrize("shots", [100, 1_000, 10_000, 100_000])
    def test_monotonically_decreasing_in_shots(self, shots: int) -> None:
        higher = auto_tolerance(shots, confidence=0.99)
        lower = auto_tolerance(shots * 10, confidence=0.99)
        assert higher > lower

    def test_default_confidence_is_99_percent(self) -> None:
        # z_{0.995} ≈ 2.5758
        tol = auto_tolerance(10_000)
        expected_z = float(stats.norm.ppf(0.995))
        assert tol == pytest.approx(expected_z * math.sqrt(1.0 / 10_000), rel=1e-9)

    @pytest.mark.parametrize(
        ("confidence", "expected_z"),
        [
            (0.68, float(stats.norm.ppf(0.84))),
            (0.95, float(stats.norm.ppf(0.975))),
            (0.99, float(stats.norm.ppf(0.995))),
        ],
    )
    def test_known_z_scores(self, confidence: float, expected_z: float) -> None:
        shots = 1_000
        tol = auto_tolerance(shots, confidence=confidence)
        assert tol == pytest.approx(expected_z * math.sqrt(1.0 / shots), rel=1e-9)

    def test_zero_shots_raises(self) -> None:
        with pytest.raises(ValueError):
            auto_tolerance(0)

    def test_negative_shots_raises(self) -> None:
        with pytest.raises(ValueError):
            auto_tolerance(-100)

    @pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.5])
    def test_confidence_out_of_range_raises(self, confidence: float) -> None:
        with pytest.raises(ValueError):
            auto_tolerance(100, confidence=confidence)
