import numpy as np
import pandas as pd
from scipy.optimize import minimize
import scipy.stats as stats
import cvxpy as cp
import os
import json 

# Import core metric functions from metrics.py
from metrics import portfolio_performance, calculate_max_drawdown, calculate_cvar


# --- Constants ---
RISK_FREE_RATE = 0.02
ANNUALIZATION_FACTOR = 252
HAS_CVXPY = True

# --- Portfolio Optimization ---
def project_to_simplex(v):
    """
    Project vector v onto the probability simplex: w >= 0, sum w = 1
    (Duchi et al. 2008)
    """
    v = np.asarray(v, dtype=float)
    if v.ndim != 1:
        v = v.ravel()
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0]
    rho = rho[-1] if rho.size > 0 else 0
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    return w

def calculate_expected_returns_covariance(returns_df: pd.DataFrame):
    """
    Calculates expected returns (mean daily returns) and covariance matrix.
    Input: DataFrame where each column is a security's returns.
    Returns: (expected_returns, covariance_matrix) - both daily.
    """
    if returns_df is None or returns_df.empty:
        print("Input returns_df is empty, cannot calculate expected returns and covariance.")
        return pd.Series(dtype=float), pd.DataFrame()
    expected_returns = returns_df.mean()
    covariance_matrix = returns_df.cov()
    return expected_returns, covariance_matrix


# ---------- Classical Optimizers (SciPy, CVXPY) ----------
def mean_variance_optimization_scipy(expected_returns, covariance_matrix, objective='max_sharpe',
                                     allow_short=False, rf=RISK_FREE_RATE, save_path_prefix='config/scipy_'):
    """
    Performs classical Mean-Variance Optimization using SciPy's minimize.
    Supports maximizing Sharpe Ratio or minimizing Volatility.
    Allows long-only or long/short portfolios.
    Saves results to JSON files in the specified path prefix.
    Uses portfolio_performance imported from metrics.py.
    """
    n = len(expected_returns)
    if n == 0:
        print("No assets for SciPy optimization.")
        return np.array([]), (0.0, 0.0, 0.0), None

    bounds = tuple((0.0, 1.0) for _ in range(n)) if not allow_short else tuple((-1.0, 1.0) for _ in range(n))
    x0 = np.ones(n)/n
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)

    def neg_sharpe(w):
        if not np.isfinite(w).all() or np.sum(np.abs(w)) == 0:
            return 1e10
        _, v, s = portfolio_performance(w, expected_returns, covariance_matrix, annualization_factor=ANNUALIZATION_FACTOR, rf=rf) # Use imported function
        return -s if v > 0 else 1e10

    def vol_obj(w):
        if not np.isfinite(w).all() or np.sum(np.abs(w)) == 0:
            return 1e10
        return portfolio_performance(w, expected_returns, covariance_matrix, annualization_factor=ANNUALIZATION_FACTOR, rf=rf)[1] # Use imported function

    obj = neg_sharpe if objective == 'max_sharpe' else vol_obj
    res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons)

    w = res.x if res.success else project_to_simplex(x0)
    w = project_to_simplex(w) # Ensure weights sum to 1 and are non-negative

    perf = portfolio_performance(w, expected_returns, covariance_matrix, annualization_factor=ANNUALIZATION_FACTOR, rf=rf) # Use imported function

    # Save results
    results = {
        'objective': objective,
        'allow_short': allow_short,
        'optimal_weights': w.tolist(),
        'annualized_return': perf[0],
        'annualized_volatility': perf[1],
        'sharpe_ratio': perf[2],
        'optimization_status': res.status,
        'optimization_message': res.message
    }
    try:
        # Ensure the directory exists. os.path.dirname gets the directory part of the path.
        os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)
        file_path = f"{save_path_prefix}{objective}_results.json"
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"SciPy optimization results saved to {file_path}")
    except Exception as e:
        print(f"Warning: Could not save SciPy optimization results: {e}")

    return w, perf, res

def mean_variance_optimization_cvxpy(expected_returns, covariance_matrix, objective='max_sharpe',
                                     allow_short=False, rf=RISK_FREE_RATE, save_path_prefix='config/cvxpy_'):
    """
    Performs Mean-Variance Optimization using CVXPY (if installed).
    Saves results to JSON files in the specified path prefix.
    Uses portfolio_performance imported from metrics.py.
    """
    try:
        import cvxpy as cp
        HAS_CVXPY_RUNTIME = True
    except ImportError:
        HAS_CVXPY_RUNTIME = False
        print("CVXPY not installed. Falling back to SciPy optimization.")
        return mean_variance_optimization_scipy(expected_returns, covariance_matrix, objective, allow_short, rf, save_path_prefix.replace('cvxpy_', 'scipy_fallback_'))


    if not HAS_CVXPY_RUNTIME:
         print("CVXPY not available at runtime. Falling back to SciPy optimization.")
         return mean_variance_optimization_scipy(expected_returns, covariance_matrix, objective, allow_short, rf, save_path_prefix.replace('cvxpy_', 'scipy_fallback_'))


    mu = np.asarray(expected_returns, dtype=float)
    Sigma = np.asarray(covariance_matrix, dtype=float)
    n = len(mu)
    if n == 0:
        print("No assets for CVXPY optimization.")
        return np.array([]), (0.0, 0.0, 0.0), None

    w = cp.Variable(n)

    cons = [cp.sum(w) == 1]
    if not allow_short:
        cons += [w >= 0]

    port_ret_ann = ANNUALIZATION_FACTOR * mu @ w
    port_var_ann = cp.quad_form(w, Sigma) * ANNUALIZATION_FACTOR
    port_vol_ann = cp.sqrt(port_var_ann)

    if objective == 'max_sharpe':
         cons += [port_vol_ann <= 1]
         prob = cp.Problem(cp.Maximize(port_ret_ann - rf), cons)

    elif objective == 'min_volatility':
        prob = cp.Problem(cp.Minimize(port_vol_ann), cons)
    else:
         raise ValueError("Objective must be 'max_sharpe' or 'min_volatility'")


    try:
        solvers = []
        if 'ECOS' in cp.installed_solvers():
            solvers.append('ECOS')
        if 'SCS' in cp.installed_solvers():
            solvers.append('SCS')
        if not solvers:
             print("No suitable CVXPY solvers installed (ECOS, SCS). Falling back to SciPy.")
             return mean_variance_optimization_scipy(expected_returns, covariance_matrix, objective, allow_short, rf, save_path_prefix.replace('cvxpy_', 'scipy_fallback_'))

        solver_to_use = solvers[0]
        print(f"Attempting to solve with CVXPY using solver: {solver_to_use}")

        prob.solve(solver=solver_to_use, verbose=False)

        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            weights = np.array(w.value).ravel()
            if not allow_short:
                 weights[weights < 1e-9] = 0
            weights = weights / np.sum(weights) if np.sum(weights) != 0 else np.ones(n)/n

            perf = portfolio_performance(weights, expected_returns, covariance_matrix, annualization_factor=ANNUALIZATION_FACTOR, rf=rf) # Use imported function

            # Save results
            results = {
                'objective': objective,
                'allow_short': allow_short,
                'optimal_weights': weights.tolist(),
                'annualized_return': perf[0],
                'annualized_volatility': perf[1],
                'sharpe_ratio': perf[2],
                'optimization_status': prob.status,
                'solver': solver_to_use
            }
            try:
                os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True) # Ensure directory exists
                file_path = f"{save_path_prefix}{objective}_results.json"
                with open(file_path, 'w') as f:
                    json.dump(results, f, indent=4)
                print(f"CVXPY optimization results saved to {file_path}")
            except Exception as e:
                print(f"Warning: Could not save CVXPY optimization results: {e}")


            print(f"CVXPY optimization successful (status: {prob.status}).")
            return weights, perf, prob
        else:
            print(f"CVXPY optimization failed (status: {prob.status}). Falling back to SciPy.")
            return mean_variance_optimization_scipy(expected_returns, covariance_matrix, objective, allow_short, rf, save_path_prefix.replace('cvxpy_', 'scipy_fallback_'))
    except cp.SolverError as e:
        print(f"CVXPY Solver Error: {e}. Falling back to SciPy.")
        return mean_variance_optimization_scipy(expected_returns, covariance_matrix, objective, allow_short, rf, save_path_prefix.replace('cvxpy_', 'scipy_fallback_'))
    except Exception as e:
        print(f"An unexpected error occurred during CVXPY optimization: {e}. Falling back to SciPy.")
        return mean_variance_optimization_scipy(expected_returns, covariance_matrix, objective, allow_short, rf, save_path_prefix.replace('cvxpy_', 'scipy_fallback_'))


def generate_efficient_frontier(expected_returns, covariance_matrix, num_portfolios=80, allow_short=False, rf=RISK_FREE_RATE, save_path='config/efficient_frontier.json'):
    """
    Generates data points along the efficient frontier by finding portfolios
    with the minimum volatility for a range of target returns.
    Uses SciPy's minimize.
    Returns: DataFrame with 'Return', 'Volatility', 'Sharpe Ratio', and 'Weights' for each portfolio on the frontier.
    Saves the frontier data to a JSON file.
    Uses portfolio_performance imported from metrics.py.
    """
    n = len(expected_returns)
    if n == 0:
        print("No assets to generate efficient frontier.")
        return pd.DataFrame()

    mu = np.asarray(expected_returns, dtype=float)
    Sigma = np.asarray(covariance_matrix, dtype=float)

    min_daily_ret = mu.min()
    max_daily_ret = mu.max()

    if np.isclose(min_daily_ret, max_daily_ret) or (np.isclose(min_daily_ret, 0) and np.isclose(max_daily_ret, 0)):
        print("Expected returns are all the same or zero. Efficient frontier is a single point.")
        if n > 0:
             w = np.ones(n) / n
             ret, vol, sharpe = portfolio_performance(w, mu, Sigma, annualization_factor=ANNUALIZATION_FACTOR, rf=rf) # Use imported function
             frontier_df = pd.DataFrame([{'Return': ret, 'Volatility': vol, 'Sharpe Ratio': sharpe, 'Weights': w.tolist()}])
             # Save single point frontier
             try:
                 os.makedirs(os.path.dirname(save_path), exist_ok=True)
                 frontier_df.to_json(save_path, orient='records', indent=4)
                 print(f"Efficient frontier (single point) saved to {save_path}")
             except Exception as e:
                 print(f"Warning: Could not save efficient frontier data: {e}")
             return frontier_df
        else:
             return pd.DataFrame()


    target_annualized_returns = np.linspace(min_daily_ret * ANNUALIZATION_FACTOR, max_daily_ret * ANNUALIZATION_FACTOR * 1.2, num_portfolios)


    records = []

    for target_ann_ret in target_annualized_returns:
        def vol_obj(w):
            if not np.isfinite(w).all() or np.sum(np.abs(w)) == 0:
                return 1e10
            return portfolio_performance(w, mu, Sigma, annualization_factor=ANNUALIZATION_FACTOR, rf=rf)[1] # Use imported function

        cons = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'eq', 'fun': lambda w, t=target_ann_ret: portfolio_performance(w, mu, Sigma, annualization_factor=ANNUALIZATION_FACTOR, rf=rf)[0] - t}, # Use imported function
        )

        bounds = tuple((0.0, 1.0) for _ in range(n)) if not allow_short else tuple((-1.0, 1.0) for _ in range(n))

        x0 = np.ones(n)/n

        res = minimize(vol_obj, x0, method='SLSQP', bounds=bounds, constraints=cons)

        if res.success:
            w = res.x
            if not allow_short:
                 w[w < 1e-9] = 0
            w = w / np.sum(w) if np.sum(w) != 0 else np.ones(n)/n

            r, v, s = portfolio_performance(w, mu, Sigma, annualization_factor=ANNUALIZATION_FACTOR, rf=rf) # Use imported function
            records.append({'Return': r, 'Volatility': v, 'Sharpe Ratio': s, 'Weights': w.tolist()})


    frontier_df = pd.DataFrame(records)
    frontier_df = frontier_df.sort_values('Volatility').reset_index(drop=True)

    # Save frontier data
    if not frontier_df.empty:
         try:
             os.makedirs(os.path.dirname(save_path), exist_ok=True)
             frontier_df.to_json(save_path, orient='records', indent=4)
             print(f"Efficient frontier data saved to {save_path}")
         except Exception as e:
             print(f"Warning: Could not save efficient frontier data: {e}")


    return frontier_df