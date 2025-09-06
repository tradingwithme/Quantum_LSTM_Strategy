import pandas as pd
import numpy as np
import os
import json

# --- Performance and Risk Metrics ---
def calculate_max_drawdown(equity_curve):
    """
    Calculates Max Drawdown from an equity curve (list or array of portfolio values).
    Represents the largest peak-to-trough decline during the historical period.
    Returns: the maximum drawdown as a percentage (negative or zero).
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0
    arr = np.asarray(equity_curve, dtype=float)
    if arr.size == 0:
        return 0.0
    peak = np.maximum.accumulate(arr)
    drawdown = np.where(peak != 0, (arr - peak) / peak, 0.0)
    return float(drawdown.min())


def calculate_sharpe_ratio(equity_curve, annualization_factor=252, risk_free_rate=0.02):
    """
    Calculates the annualized Sharpe Ratio from an equity curve.
    Assumes daily data for annualization.
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0

    equity_series = pd.Series(equity_curve)
    returns = equity_series.pct_change().dropna()

    if returns.empty:
        return 0.0

    avg_daily_return = returns.mean()
    std_daily_return = returns.std()

    if std_daily_return == 0:
        return 0.0

    annualized_return = avg_daily_return * annualization_factor
    annualized_volatility = std_daily_return * np.sqrt(annualization_factor)

    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

    return float(sharpe_ratio)


def calculate_cvar(returns: pd.Series, alpha=0.95):
    """
    Calculates Conditional Value-at-Risk (CVaR) at confidence level alpha.
    Returns: the CVaR as a positive number (expected loss).
    """
    if returns is None or len(returns) == 0:
        return 0.0
    losses = -pd.Series(returns).dropna().values
    if len(losses) == 0:
        return 0.0

    var = np.quantile(losses, alpha)

    tail_losses = losses[losses >= var]

    if len(tail_losses) == 0:
        return -float(var) if var > 0 else 0.0

    return float(tail_losses.mean())


def portfolio_performance(weights, expected_returns, covariance_matrix,
                          annualization_factor=252, rf=0.02):
    """
    Calculate portfolio return, volatility, and Sharpe ratio.
    Inputs: daily expected returns and covariance matrix.
    Outputs: annualized return, annualized volatility, annualized Sharpe ratio.
    """
    w = np.asarray(weights, dtype=float)
    if np.sum(w) != 0:
      w = w / np.sum(w)
    else:
      w = np.ones(len(weights)) / len(weights)

    mu = np.asarray(expected_returns, dtype=float)
    Sigma = np.asarray(covariance_matrix, dtype=float)

    if mu.shape[0] != w.shape[0] or Sigma.shape[0] != w.shape[0] or Sigma.shape[1] != w.shape[0]:
        print("Warning: Dimension mismatch in portfolio_performance. Check inputs.")
        if mu.shape[0] == w.shape[0]:
             p_ret = float(np.dot(w, mu) * annualization_factor)
        else:
             p_ret = 0.0

        if Sigma.shape[0] == w.shape[0] and Sigma.shape[1] == w.shape[0]:
             p_vol = float(np.sqrt(w @ Sigma @ w) * np.sqrt(annualization_factor))
        else:
             p_vol = 0.0

        p_sharpe = (p_ret - rf) / p_vol if p_vol > 0 else 0.0
        return p_ret, p_vol, p_sharpe


    p_ret = float(np.dot(w, mu) * annualization_factor)
    p_vol = float(np.sqrt(w @ Sigma @ w) * np.sqrt(annualization_factor))
    p_sharpe = (p_ret - rf) / p_vol if p_vol > 0 else 0.0
    return p_ret, p_vol, p_sharpe