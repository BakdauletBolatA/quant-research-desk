"""Covariance estimators, allocators and Black-Litterman.

Each allocator is tested against the property that *defines* it, not against a
stored output — a snapshot test would pass a subtly wrong risk-parity solver.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

import quantdesk.portfolio.covariance as cov_mod
import quantdesk.portfolio.optimizer as opt

# The package re-exports a `black_litterman` *function*, which shadows the
# submodule attribute, so the module is fetched explicitly.
bl_mod = importlib.import_module("quantdesk.portfolio.black_litterman")


@pytest.fixture(scope="module")
def cov(synthetic_returns) -> pd.DataFrame:
    return cov_mod.estimate_covariance(synthetic_returns, "ledoit_wolf", 252)


@pytest.fixture(scope="module")
def expected_returns(cov) -> pd.Series:
    return pd.Series([0.08, 0.06, 0.045, 0.03], index=cov.index)


# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", cov_mod.COVARIANCE_METHODS)
def test_covariance_is_symmetric_and_psd(synthetic_returns, method):
    matrix = cov_mod.estimate_covariance(synthetic_returns, method, 252).to_numpy()
    assert np.allclose(matrix, matrix.T, atol=1e-14)
    assert np.linalg.eigvalsh(matrix).min() > 0


def test_sample_covariance_annualises_by_the_period_count(synthetic_returns):
    daily = synthetic_returns.cov(ddof=1).to_numpy()
    annual = cov_mod.sample_covariance(synthetic_returns, 252).to_numpy()
    assert np.allclose(annual, daily * 252, atol=1e-12)


def test_shrinkage_is_a_valid_intensity(synthetic_returns):
    assert 0.0 <= cov_mod.ledoit_wolf_shrinkage(synthetic_returns) <= 1.0


def test_shrinkage_improves_conditioning_on_a_short_window(synthetic_returns):
    """The case that matters: fewer observations than the optimiser needs."""
    short = synthetic_returns.head(60)
    sample = cov_mod.condition_number(cov_mod.sample_covariance(short, 252))
    shrunk = cov_mod.condition_number(cov_mod.ledoit_wolf_covariance(short, 252))
    assert shrunk < sample


def test_nearest_psd_repairs_an_indefinite_matrix():
    broken = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, 0.9], [0.9, 0.9, -0.5]])
    repaired = cov_mod.nearest_psd(broken)
    assert np.linalg.eigvalsh(repaired).min() >= -1e-12
    assert np.allclose(repaired, repaired.T)


def test_correlation_has_unit_diagonal_and_is_bounded(cov):
    corr = cov_mod.correlation_from_covariance(cov).to_numpy()
    assert np.allclose(np.diag(corr), 1.0)
    assert corr.min() >= -1.0 and corr.max() <= 1.0


def test_ewma_covariance_weights_recent_data_more():
    """A late volatility spike must lift the EWMA estimate above the sample one."""
    index = pd.bdate_range("2020-01-01", periods=600)
    generator = np.random.default_rng(6)
    calm = generator.normal(0, 0.005, (500, 2))
    stormy = generator.normal(0, 0.030, (100, 2))
    frame = pd.DataFrame(np.vstack([calm, stormy]), index=index, columns=["A", "B"])
    ewma = cov_mod.ewma_covariance(frame, 0.94, 252).to_numpy()
    sample = cov_mod.sample_covariance(frame, 252).to_numpy()
    assert ewma[0, 0] > sample[0, 0]


# ---------------------------------------------------------------------------
# Allocators — shared invariants
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", opt.METHODS)
def test_weights_are_a_valid_long_only_portfolio(cov, expected_returns, method):
    constraints = opt.Constraints(long_only=True, min_weight=0.0, max_weight=0.60)
    weights = opt.optimise(method, cov, expected_returns, constraints)
    assert list(weights.index) == list(cov.index)
    assert weights.sum() == pytest.approx(1.0, abs=1e-8)
    assert (weights >= -1e-9).all()
    assert weights.max() <= 0.60 + 1e-6, method


@pytest.mark.parametrize("method", opt.METHODS)
def test_allocators_are_deterministic(cov, expected_returns, method):
    first = opt.optimise(method, cov, expected_returns)
    second = opt.optimise(method, cov, expected_returns)
    assert np.allclose(first.to_numpy(), second.to_numpy(), atol=1e-10)


def test_constraints_reject_an_infeasible_cap(cov):
    with pytest.raises(ValueError):
        opt.min_variance(cov, opt.Constraints(max_weight=0.10))  # 4 assets * 0.10 < 1


# ---------------------------------------------------------------------------
# Allocators — defining properties
# ---------------------------------------------------------------------------
def test_min_variance_actually_minimises_variance(cov):
    matrix = cov.to_numpy()
    optimal = opt.min_variance(cov).to_numpy()
    equal = np.full(len(optimal), 1.0 / len(optimal))
    assert optimal @ matrix @ optimal <= equal @ matrix @ equal + 1e-12

    generator = np.random.default_rng(13)
    for _ in range(200):
        candidate = generator.dirichlet(np.ones(len(optimal)))
        assert optimal @ matrix @ optimal <= candidate @ matrix @ candidate + 1e-9


def test_risk_parity_equalises_risk_contributions(cov):
    weights = opt.risk_parity(cov).to_numpy()
    shares = opt.risk_contributions(weights, cov.to_numpy())
    shares = shares / shares.sum()
    assert shares.std(ddof=0) < 1e-9
    assert shares.mean() == pytest.approx(1.0 / len(weights))


def test_risk_parity_tilts_toward_the_low_volatility_asset(cov):
    weights = opt.risk_parity(cov)
    volatilities = pd.Series(np.sqrt(np.diag(cov.to_numpy())), index=cov.index)
    assert weights.idxmax() == volatilities.idxmin()


def test_max_sharpe_beats_equal_weight_on_ex_ante_sharpe(cov, expected_returns):
    matrix = cov.to_numpy()
    mu = expected_returns.to_numpy()

    def sharpe(w: np.ndarray) -> float:
        return float(w @ mu / np.sqrt(w @ matrix @ w))

    optimal = opt.max_sharpe(cov, expected_returns).to_numpy()
    equal = np.full(len(mu), 1.0 / len(mu))
    assert sharpe(optimal) >= sharpe(equal) - 1e-9


def test_max_diversification_maximises_the_diversification_ratio(cov):
    matrix = cov.to_numpy()
    optimal = opt.max_diversification(cov).to_numpy()
    generator = np.random.default_rng(17)
    best = opt.diversification_ratio(optimal, matrix)
    for _ in range(200):
        candidate = generator.dirichlet(np.ones(len(optimal)))
        assert best >= opt.diversification_ratio(candidate, matrix) - 1e-9


def test_hrp_never_inverts_the_covariance_matrix(cov):
    """HRP must survive a singular matrix, which is the point of using it."""
    singular = cov.copy()
    singular.iloc[:, 1] = singular.iloc[:, 0].to_numpy()
    singular.iloc[1, :] = singular.iloc[0, :].to_numpy()
    weights = opt.hrp(singular)
    assert weights.sum() == pytest.approx(1.0, abs=1e-8)
    assert (weights > 0).all()


def test_equal_weight_is_exactly_one_over_n(cov):
    weights = opt.equal_weight(cov)
    assert np.allclose(weights.to_numpy(), 1.0 / cov.shape[0])


def test_effective_number_of_bets_endpoints():
    assert opt.effective_number_of_bets(np.array([0.25] * 4)) == pytest.approx(4.0)
    assert opt.effective_number_of_bets(np.array([1.0, 0.0, 0.0, 0.0])) == pytest.approx(1.0)


def test_risk_contributions_sum_to_portfolio_volatility(cov):
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    matrix = cov.to_numpy()
    assert opt.risk_contributions(weights, matrix).sum() == pytest.approx(
        opt.portfolio_volatility(weights, matrix)
    )


def test_efficient_frontier_is_monotone_in_risk(cov, expected_returns):
    frontier = opt.efficient_frontier(cov, expected_returns, n_points=25)
    assert len(frontier) >= 5
    assert frontier["volatility"].is_monotonic_increasing
    assert frontier["expected_return"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# Black-Litterman
# ---------------------------------------------------------------------------
def test_equilibrium_returns_are_the_reverse_optimisation_identity(cov):
    weights = pd.Series(0.25, index=cov.index)
    delta = 2.5
    pi = bl_mod.implied_equilibrium_returns(cov, weights, delta)
    expected = delta * cov.to_numpy() @ weights.to_numpy()
    assert np.allclose(pi.to_numpy(), expected)


def test_equilibrium_weights_must_sum_to_one(cov):
    with pytest.raises(ValueError):
        bl_mod.implied_equilibrium_returns(cov, pd.Series(0.5, index=cov.index), 2.5)


def test_derived_risk_aversion_reproduces_the_stated_erp(cov):
    weights = pd.Series(0.25, index=cov.index)
    erp = 0.045
    delta = bl_mod.derive_risk_aversion(cov, weights, erp)
    pi = bl_mod.implied_equilibrium_returns(cov, weights, delta)
    assert float(pi @ weights) == pytest.approx(erp, rel=1e-10)


def test_no_views_leaves_the_prior_untouched(cov):
    weights = pd.Series(0.25, index=cov.index)
    result = bl_mod.black_litterman(cov, weights, [], risk_aversion=2.5)
    assert np.allclose(result.posterior_returns.to_numpy(), result.prior_returns.to_numpy())


def test_a_near_certain_view_is_honoured(cov):
    weights = pd.Series(0.25, index=cov.index)
    view = [{"type": "absolute", "assets": ["HIGH"], "value": 0.25, "confidence": 0.999}]
    result = bl_mod.black_litterman(cov, weights, view, risk_aversion=2.5, tau=0.05)
    assert result.posterior_returns["HIGH"] == pytest.approx(0.25, abs=0.01)


def test_a_worthless_view_is_ignored(cov):
    weights = pd.Series(0.25, index=cov.index)
    view = [{"type": "absolute", "assets": ["HIGH"], "value": 0.99, "confidence": 1e-4}]
    result = bl_mod.black_litterman(cov, weights, view, risk_aversion=2.5, tau=0.05)
    assert result.posterior_returns["HIGH"] == pytest.approx(
        result.prior_returns["HIGH"], abs=0.01
    )


def test_relative_view_rows_are_zero_sum(cov):
    view = [{"type": "relative", "assets": ["HIGH", "LOW"], "value": 0.04, "confidence": 0.5}]
    P, Q, _ = bl_mod.build_views(view, list(cov.index))
    assert P.to_numpy().sum() == pytest.approx(0.0)
    assert P.iloc[0]["HIGH"] == pytest.approx(1.0)
    assert P.iloc[0]["LOW"] == pytest.approx(-1.0)
    assert Q.iloc[0] == pytest.approx(0.04)


def test_relative_view_moves_the_pair_in_opposite_directions(cov):
    weights = pd.Series(0.25, index=cov.index)
    view = [{"type": "relative", "assets": ["HIGH", "LOW"], "value": 0.10, "confidence": 0.9}]
    result = bl_mod.black_litterman(cov, weights, view, risk_aversion=2.5, tau=0.05)
    impact = result.posterior_returns - result.prior_returns
    assert impact["HIGH"] > 0 > impact["LOW"]


def test_unknown_view_type_raises(cov):
    with pytest.raises(ValueError):
        bl_mod.build_views([{"type": "sideways", "assets": ["HIGH"], "value": 0.1}],
                           list(cov.index))


def test_posterior_covariance_is_at_least_the_prior(cov):
    weights = pd.Series(0.25, index=cov.index)
    view = [{"type": "absolute", "assets": ["MID"], "value": 0.05, "confidence": 0.4}]
    result = bl_mod.black_litterman(cov, weights, view, risk_aversion=2.5)
    assert (np.diag(result.posterior_covariance.to_numpy()) >= np.diag(cov.to_numpy())).all()


def test_market_cap_weights_are_price_times_shares(cov):
    prices = pd.Series({"HIGH": 100.0, "MID": 50.0, "LOW": 20.0, "DEFENSIVE": 10.0})
    shares = {"HIGH": 10.0, "MID": 10.0, "LOW": 10.0, "DEFENSIVE": 10.0}
    weights = bl_mod.market_cap_weights(prices, shares)
    assert weights.sum() == pytest.approx(1.0)
    assert weights["HIGH"] == pytest.approx(100.0 / 180.0)
