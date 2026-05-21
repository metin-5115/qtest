"""Frequentist hypothesis tests and shot-noise helpers used by qtest assertions.

All functions are pure: they validate their inputs and return numerical
results without mutating arguments. Tests follow the scipy.stats conventions
for ``(statistic, p_value)`` tuples.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from scipy import stats

_PROB_SUM_ATOL: float = 1e-6


def chi_square_test(
    observed: Mapping[str, float],
    expected: Mapping[str, float],
    shots: int,
) -> tuple[float, float]:
    r"""Pearson :math:`\chi^{2}` goodness-of-fit test for measurement counts.

    .. math::

        \chi^{2} \;=\; \sum_{x} \frac{ (O_{x} - E_{x})^{2} }{ E_{x} }

    where :math:`O_{x}` is the observed count for outcome ``x`` and
    :math:`E_{x} = N \cdot P_{\text{expected}}(x)` is the expected count
    under the null hypothesis at :math:`N` shots.

    Outcomes missing from either mapping are treated as zero. Bins with
    both observed and expected equal to zero are dropped. A bin with
    ``E = 0`` but ``O > 0`` is a hard rejection — that case raises rather
    than returning an infinite statistic.

    Parameters
    ----------
    observed : Mapping[str, float]
        Observed counts per outcome (integer-valued; floats are tolerated and
        passed through to scipy).
    expected : Mapping[str, float]
        Expected *probabilities* per outcome. Must form a probability
        distribution (non-negative, summing to one).
    shots : int
        Total number of measurement shots used to convert expected
        probabilities into expected counts.

    Returns
    -------
    statistic : float
        The :math:`\chi^{2}` test statistic.
    p_value : float
        Two-sided p-value under the null hypothesis.

    Raises
    ------
    ValueError
        If ``shots`` is non-positive, either mapping is empty, ``expected`` is
        not a valid probability distribution, or a non-empty observed bin has
        zero expected probability.
    """
    if shots <= 0:
        raise ValueError(f"shots must be positive; got {shots}.")
    if not observed:
        raise ValueError("observed must be a non-empty mapping.")
    if not expected:
        raise ValueError("expected must be a non-empty mapping.")

    for key, prob in expected.items():
        if not isinstance(prob, (int, float)) or np.isnan(prob) or np.isinf(prob):
            raise ValueError(f"expected[{key!r}] = {prob!r} is not a finite real number.")
        if prob < 0.0:
            raise ValueError(f"expected[{key!r}] = {prob} is negative.")
    expected_total = float(sum(expected.values()))
    if not np.isclose(expected_total, 1.0, atol=_PROB_SUM_ATOL):
        raise ValueError(
            f"expected probabilities must sum to 1 (within {_PROB_SUM_ATOL}); "
            f"got {expected_total}."
        )

    keys = sorted(set(observed.keys()) | set(expected.keys()))
    observed_counts = np.array([float(observed.get(k, 0.0)) for k in keys], dtype=float)
    expected_counts = np.array([float(expected.get(k, 0.0)) * shots for k in keys], dtype=float)

    # Reject the model immediately if it forbids an outcome we actually saw.
    impossible = [
        k for k, e, o in zip(keys, expected_counts, observed_counts) if e == 0.0 and o > 0.0
    ]
    if impossible:
        raise ValueError(
            f"expected probability is 0 for non-empty observed bins: {impossible}"
        )

    # Drop bins where both observed and expected are zero — they contribute
    # nothing and would make scipy unhappy (division by zero).
    keep = ~((observed_counts == 0.0) & (expected_counts == 0.0))
    observed_counts = observed_counts[keep]
    expected_counts = expected_counts[keep]

    # scipy.stats.chisquare requires the two arrays to share the same sum.
    obs_sum = float(observed_counts.sum())
    exp_sum = float(expected_counts.sum())
    if exp_sum > 0.0 and not np.isclose(obs_sum, exp_sum):
        expected_counts = expected_counts * (obs_sum / exp_sum)

    result = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)
    return float(result.statistic), float(result.pvalue)


def kolmogorov_smirnov_test(
    observed: Sequence[float],
    expected_cdf: Callable[[float], float],
) -> tuple[float, float]:
    r"""One-sample Kolmogorov–Smirnov test against a continuous CDF.

    .. math::

        D_{n} \;=\; \sup_{x} \left| F_{n}(x) - F(x) \right|

    where :math:`F_{n}` is the empirical CDF of ``observed`` and :math:`F`
    is ``expected_cdf``.

    Parameters
    ----------
    observed : Sequence[float]
        Samples drawn from the candidate distribution.
    expected_cdf : Callable[[float], float]
        Cumulative distribution function of the reference distribution.

    Returns
    -------
    statistic : float
        KS test statistic :math:`D_{n}`.
    p_value : float
        Two-sided p-value under the null hypothesis that the samples are
        drawn from the distribution described by ``expected_cdf``.

    Raises
    ------
    ValueError
        If ``observed`` is empty or ``expected_cdf`` is not callable.
    """
    if not observed:
        raise ValueError("observed must be a non-empty sequence.")
    if not callable(expected_cdf):
        raise ValueError("expected_cdf must be callable.")
    samples = np.asarray(list(observed), dtype=float)
    result = stats.kstest(samples, expected_cdf)
    return float(result.statistic), float(result.pvalue)


def auto_tolerance(shots: int, confidence: float = 0.99) -> float:
    r"""Shot-noise-aware default tolerance for distribution assertions.

    Uses the Wald-style normal approximation

    .. math::

        \tau \;=\; z_{1 - \alpha / 2} \cdot \sqrt{ \frac{1}{N} }

    where :math:`N` is the shot count, :math:`\alpha = 1 - \text{confidence}`
    is the two-sided significance level, and :math:`z_{1 - \alpha/2}` is the
    corresponding standard normal quantile.

    Parameters
    ----------
    shots : int
        Number of measurement shots. Must be positive.
    confidence : float, optional
        Two-sided confidence level in ``(0, 1)``. Defaults to ``0.99``.

    Returns
    -------
    float
        Suggested tolerance for ``assert_distribution_close``-style checks.

    Raises
    ------
    ValueError
        If ``shots`` is non-positive or ``confidence`` is outside ``(0, 1)``.
    """
    if shots <= 0:
        raise ValueError(f"shots must be positive; got {shots}.")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1); got {confidence}.")
    alpha = 1.0 - confidence
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return float(z * np.sqrt(1.0 / shots))
