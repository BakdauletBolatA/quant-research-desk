"""Shared fixtures.

Tests run on synthetic data wherever the property under test is mathematical,
and on the committed data cache only where the behaviour under test is about
real-world alignment. That keeps the suite fast, deterministic and runnable
with no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(20260909)


@pytest.fixture(scope="session")
def dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2015-01-01", periods=1500)


TRUE_BETAS = {"HIGH": 1.20, "MID": 1.00, "LOW": 0.70, "DEFENSIVE": 0.40}


@pytest.fixture(scope="session")
def synthetic_market(dates: pd.DatetimeIndex) -> pd.Series:
    """The latent market factor the synthetic assets are built from."""
    generator = np.random.default_rng(7)
    return pd.Series(generator.normal(0.0004, 0.010, len(dates)), index=dates, name="MKT")


@pytest.fixture(scope="session")
def synthetic_returns(dates: pd.DatetimeIndex, synthetic_market: pd.Series) -> pd.DataFrame:
    """Four assets with known betas to ``synthetic_market`` and unequal idio vol."""
    generator = np.random.default_rng(19)
    betas = np.array(list(TRUE_BETAS.values()))
    idio_vol = np.array([0.012, 0.009, 0.007, 0.006])
    idio = generator.normal(0.0, 1.0, (len(dates), len(betas))) * idio_vol
    data = synthetic_market.to_numpy()[:, None] * betas[None, :] + idio
    return pd.DataFrame(data, index=dates, columns=list(TRUE_BETAS))


@pytest.fixture(scope="session")
def synthetic_series(synthetic_returns: pd.DataFrame) -> pd.Series:
    return synthetic_returns.mean(axis=1).rename("portfolio")


@pytest.fixture(scope="session")
def dcf_inputs() -> dict:
    """A deliberately simple, hand-checkable company."""
    return {
        "revenue": 1000.0,
        "ebit": 200.0,
        "d_and_a": 50.0,
        "capex": 50.0,
        "delta_nwc": 0.0,
        "tax_rate": 0.25,
        "total_debt": 400.0,
        "cash_and_investments": 100.0,
        "shares_diluted": 100.0,
        "beta": 1.0,
        "cost_of_debt": 0.05,
        "revenue_growth": [0.05, 0.05, 0.05, 0.05, 0.05],
        "ebit_margin": [0.20, 0.20, 0.20, 0.20, 0.20],
        "terminal_growth": 0.02,
        "terminal_roic": 0.20,
    }


@pytest.fixture(scope="session")
def market_assumptions() -> dict:
    return {"risk_free_rate": 0.04, "equity_risk_premium": 0.05, "marginal_tax_rate": 0.25}


@pytest.fixture(scope="session")
def mc_config() -> dict:
    return {
        "revenue_growth_shock_sd": 0.02,
        "ebit_margin_shock_sd": 0.02,
        "wacc_shock_sd": 0.0075,
        "terminal_growth_shock_sd": 0.004,
        "terminal_growth_cap_spread": 0.01,
    }


@pytest.fixture(scope="session")
def wide_returns(dates: pd.DatetimeIndex, synthetic_market: pd.Series) -> pd.DataFrame:
    """Eight assets with dispersed betas and drifts.

    Needed wherever a rule selects a fixed number of names: with only four
    assets a "top five" screen holds everything and degenerates into 1/N.
    """
    generator = np.random.default_rng(101)
    n = len(dates)
    betas = np.linspace(0.4, 1.6, 8)
    drifts = np.linspace(-0.0002, 0.0006, 8)
    idio = generator.normal(0.0, 1.0, (n, 8)) * np.linspace(0.006, 0.014, 8)
    data = synthetic_market.to_numpy()[:, None] * betas[None, :] + idio + drifts[None, :]
    return pd.DataFrame(data, index=dates, columns=[f"A{i + 1}" for i in range(8)])
