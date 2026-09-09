"""Black-Litterman: blending analyst views with market equilibrium.

Plain mean-variance optimisation on historical means is unusable in production.
Feed it five years of sample returns and it will put 40% of the book in
whichever name happened to run hardest, because a 1% error in an expected
return moves the optimal weight far more than a 1% error in a covariance.

Black-Litterman fixes the input rather than patching the output. It starts from
the expected returns that would make the *market portfolio* optimal — reverse
optimisation, so the neutral answer is "hold the market" — and then moves away
from that prior only as far as the stated confidence in each view justifies.

    Π      = δ · Σ · w_market                       (implied equilibrium returns)
    μ_BL   = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹Π + P'Ω⁻¹Q]  (posterior mean)
    Σ_BL   = Σ + [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹                 (posterior covariance)

A view with zero confidence changes nothing; a view with full confidence is
imposed exactly. Every intermediate case is a weighted average — which is the
property that makes the model safe to hand to a discretionary PM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BlackLittermanResult:
    prior_returns: pd.Series           # Π, annualised
    posterior_returns: pd.Series       # μ_BL, annualised
    posterior_covariance: pd.DataFrame
    market_weights: pd.Series
    view_matrix: pd.DataFrame          # P
    view_returns: pd.Series            # Q
    view_uncertainty: pd.Series        # diag(Ω)
    tau: float
    risk_aversion: float

    def view_impact(self) -> pd.Series:
        """How far each name's expected return moved away from equilibrium."""
        return (self.posterior_returns - self.prior_returns).sort_values(ascending=False)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Market weight": self.market_weights,
                "Equilibrium return (Π)": self.prior_returns,
                "Posterior return (μ_BL)": self.posterior_returns,
                "View impact": self.posterior_returns - self.prior_returns,
            }
        )


def implied_equilibrium_returns(
    cov: pd.DataFrame, market_weights: pd.Series, risk_aversion: float
) -> pd.Series:
    """Reverse-optimise the market portfolio into expected excess returns."""
    w = market_weights.reindex(cov.index).to_numpy(dtype=float)
    if not np.isclose(w.sum(), 1.0, atol=1e-6):
        raise ValueError(f"market_weights must sum to 1, got {w.sum():.6f}")
    return pd.Series(risk_aversion * (cov.to_numpy() @ w), index=cov.index, name="pi")


def implied_risk_aversion(
    market_excess_return: float, market_variance: float
) -> float:
    """δ = E[R_m − r_f] / σ²_m, the market's revealed risk aversion."""
    if market_variance <= 0:
        raise ValueError("market_variance must be positive")
    return float(market_excess_return / market_variance)


def derive_risk_aversion(
    cov: pd.DataFrame, market_weights: pd.Series, equity_risk_premium: float
) -> float:
    """Calibrate δ so the equilibrium prior reproduces the stated ERP.

    Textbook δ = 2.5-3.0 is calibrated to a broad index. Applied to a 16-name
    mega-cap portfolio with ~22% volatility it implies an equilibrium market
    return north of 14%, which quietly overwrites the risk premium assumption
    used everywhere else in the platform. Deriving δ from the same ERP keeps
    the valuation chapter and the allocation chapter on one set of numbers.
    """
    w = market_weights.reindex(cov.index).to_numpy(dtype=float)
    market_variance = float(w @ cov.to_numpy() @ w)
    return implied_risk_aversion(equity_risk_premium, market_variance)


def build_views(
    views: list[dict], assets: list[str]
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Turn the YAML view list into the (P, Q, confidence) triple.

    Absolute view: "NVDA returns 6% over the risk-free rate."
        one +1 entry.
    Relative view: "MSFT outperforms XOM by 4%."
        +1 on the first asset, −1 spread evenly across the rest.
    """
    rows, q_values, confidences, labels = [], [], [], []

    for i, view in enumerate(views):
        kind = view.get("type", "absolute")
        names = [a for a in view["assets"] if a in assets]
        if not names:
            continue

        row = pd.Series(0.0, index=assets)
        if kind == "absolute":
            row[names] = 1.0 / len(names)
            label = f"{'+'.join(names)} = {view['value']:+.2%}"
        elif kind == "relative":
            if len(names) < 2:
                continue
            row[names[0]] = 1.0
            row[names[1:]] = -1.0 / (len(names) - 1)
            label = f"{names[0]} − {'/'.join(names[1:])} = {view['value']:+.2%}"
        else:
            raise ValueError(f"Unknown view type {kind!r}")

        rows.append(row)
        q_values.append(float(view["value"]))
        confidences.append(float(view.get("confidence", 0.5)))
        labels.append(f"V{i + 1}: {label}")

    if not rows:
        empty = pd.DataFrame(columns=assets)
        return empty, pd.Series(dtype=float), pd.Series(dtype=float)

    P = pd.DataFrame(rows, index=labels)
    Q = pd.Series(q_values, index=labels, name="Q")
    C = pd.Series(confidences, index=labels, name="confidence").clip(1e-4, 0.999)
    return P, Q, C


def black_litterman(
    cov: pd.DataFrame,
    market_weights: pd.Series,
    views: list[dict],
    *,
    risk_aversion: float = 3.0,
    tau: float = 0.05,
) -> BlackLittermanResult:
    """Posterior expected returns and covariance given equilibrium and views."""
    assets = list(cov.index)
    sigma = cov.to_numpy(dtype=float)
    pi = implied_equilibrium_returns(cov, market_weights, risk_aversion)

    P, Q, confidence = build_views(views, assets)
    if P.empty:
        return BlackLittermanResult(
            prior_returns=pi, posterior_returns=pi.rename("mu_bl"),
            posterior_covariance=cov, market_weights=market_weights.reindex(assets),
            view_matrix=P, view_returns=Q, view_uncertainty=pd.Series(dtype=float),
            tau=tau, risk_aversion=risk_aversion,
        )

    P_mat = P.to_numpy(dtype=float)
    tau_sigma = tau * sigma

    # Idzorek-style confidence: Ω → 0 as confidence → 1 (the view is imposed),
    # Ω → ∞ as confidence → 0 (the view is ignored).
    view_variance = np.diag(P_mat @ tau_sigma @ P_mat.T)
    c = confidence.to_numpy()
    omega_diag = np.maximum(view_variance * (1.0 - c) / c, 1e-12)
    omega_inv = np.diag(1.0 / omega_diag)

    tau_sigma_inv = np.linalg.inv(tau_sigma)
    posterior_precision = tau_sigma_inv + P_mat.T @ omega_inv @ P_mat
    posterior_cov_of_mean = np.linalg.inv(posterior_precision)

    mu_bl = posterior_cov_of_mean @ (
        tau_sigma_inv @ pi.to_numpy() + P_mat.T @ omega_inv @ Q.to_numpy()
    )

    return BlackLittermanResult(
        prior_returns=pi,
        posterior_returns=pd.Series(mu_bl, index=assets, name="mu_bl"),
        posterior_covariance=pd.DataFrame(
            sigma + posterior_cov_of_mean, index=assets, columns=assets
        ),
        market_weights=market_weights.reindex(assets),
        view_matrix=P,
        view_returns=Q,
        view_uncertainty=pd.Series(omega_diag, index=P.index, name="omega"),
        tau=tau,
        risk_aversion=risk_aversion,
    )


def market_cap_weights(
    prices: pd.Series, shares_outstanding: dict[str, float]
) -> pd.Series:
    """Cap weights for the investable universe from the last price and share count."""
    caps = {
        ticker: float(price) * float(shares_outstanding[ticker])
        for ticker, price in prices.items()
        if ticker in shares_outstanding
    }
    if not caps:
        raise ValueError("No overlap between prices and shares_outstanding")
    series = pd.Series(caps)
    return (series / series.sum()).rename("market_weight")
