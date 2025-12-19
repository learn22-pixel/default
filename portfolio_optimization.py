"""
Portfolio Optimization Module
Implements mean-variance portfolio optimization using scipy.

Constraints:
- No short selling (weights >= 0)
- Maximum weight per asset: 10%
- Portfolio volatility <= 10%
- Excluded tickers have weight = 0
- Objective: Maximize portfolio return
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Optional


def portfolio_return(weights: np.ndarray, mean_returns: np.ndarray) -> float:
    """Calculate portfolio expected return."""
    return np.sum(weights * mean_returns)


def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Calculate portfolio volatility (standard deviation)."""
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


def negative_portfolio_return(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray
) -> float:
    """Negative portfolio return for minimization."""
    return -portfolio_return(weights, mean_returns)


def optimize_portfolio(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    max_volatility: float = 0.10,
    max_weight: float = 0.10,
    excluded_tickers: Optional[List[str]] = None,
    risk_free_rate: float = 0.02
) -> Dict:
    """
    Optimize portfolio to maximize return subject to constraints.

    Constraints:
    - No short selling (weights >= 0)
    - Sum of weights = 1
    - Each weight <= max_weight (default 10%)
    - Portfolio volatility <= max_volatility (default 10%)
    - Excluded tickers have weight = 0
    """
    if excluded_tickers is None:
        excluded_tickers = []

    n_assets = len(mean_returns)
    tickers = mean_returns.index.tolist()

    mean_ret_arr = mean_returns.values
    cov_mat_arr = cov_matrix.values

    # Initial guess: equal weights for non-excluded assets
    n_included = n_assets - len([t for t in excluded_tickers if t in tickers])
    init_weights = np.array([
        0.0 if t in excluded_tickers else 1.0 / n_included
        for t in tickers
    ])

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'ineq', 'fun': lambda w: max_volatility - portfolio_volatility(w, cov_mat_arr)}
    ]

    # Add exclusion constraints for excluded tickers
    for i, ticker in enumerate(tickers):
        if ticker in excluded_tickers:
            constraints.append({'type': 'eq', 'fun': lambda w, idx=i: w[idx]})

    # Bounds: no short selling and max weight
    bounds = []
    for ticker in tickers:
        if ticker in excluded_tickers:
            bounds.append((0, 0))  # Excluded: weight must be 0
        else:
            bounds.append((0, max_weight))
    bounds = tuple(bounds)

    objective = lambda w: negative_portfolio_return(w, mean_ret_arr, cov_mat_arr)

    result = minimize(
        objective,
        init_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-10}
    )

    if not result.success:
        print(f"Warning: Optimization did not converge. Message: {result.message}")
        print("Retrying with slightly relaxed constraints...")
        constraints[1] = {'type': 'ineq', 'fun': lambda w: (max_volatility * 1.05) - portfolio_volatility(w, cov_mat_arr)}
        result = minimize(
            objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 2000, 'ftol': 1e-9}
        )

    optimal_weights = result.x
    optimal_weights[optimal_weights < 0.001] = 0
    optimal_weights = optimal_weights / optimal_weights.sum()

    # Ensure excluded tickers have exactly 0 weight
    for i, ticker in enumerate(tickers):
        if ticker in excluded_tickers:
            optimal_weights[i] = 0.0

    # Renormalize if needed
    if optimal_weights.sum() > 0:
        optimal_weights = optimal_weights / optimal_weights.sum()

    port_return = portfolio_return(optimal_weights, mean_ret_arr)
    port_volatility = portfolio_volatility(optimal_weights, cov_mat_arr)
    port_sharpe = (port_return - risk_free_rate) / port_volatility if port_volatility > 0 else 0

    return {
        'weights': dict(zip(tickers, optimal_weights)),
        'portfolio_return': port_return,
        'portfolio_volatility': port_volatility,
        'sharpe_ratio': port_sharpe,
        'optimization_success': result.success,
        'optimization_message': result.message,
        'excluded_tickers': excluded_tickers,
        'constraints': {
            'max_volatility': max_volatility,
            'max_weight': max_weight,
            'no_shorting': True
        }
    }


def backtest_portfolio(
    weights: Dict[str, float],
    returns: pd.DataFrame,
    initial_value: float = 100.0
) -> Dict:
    """Backtest portfolio with given weights on historical returns."""
    tickers = [t for t in weights.keys() if t in returns.columns]
    weight_arr = np.array([weights[t] for t in tickers])
    returns_aligned = returns[tickers]

    portfolio_returns = (returns_aligned * weight_arr).sum(axis=1)
    portfolio_value = initial_value * (1 + portfolio_returns).cumprod()

    individual_values = {}
    individual_returns_data = {}
    for ticker in tickers:
        if weights[ticker] > 0.001:
            stock_value = initial_value * (1 + returns[ticker]).cumprod()
            individual_values[ticker] = stock_value.tolist()
            individual_returns_data[ticker] = returns[ticker].tolist()

    total_return = (portfolio_value.iloc[-1] / initial_value) - 1
    annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
    annual_volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0

    rolling_max = portfolio_value.cummax()
    drawdown = (portfolio_value - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    return {
        'portfolio_returns': portfolio_returns,
        'portfolio_value': portfolio_value,
        'individual_values': individual_values,
        'individual_returns': individual_returns_data,
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'daily_returns': portfolio_returns.tolist(),
        'dates': [str(d.date()) for d in returns.index],
        'start_date': str(returns.index[0].date()),
        'end_date': str(returns.index[-1].date()),
        'num_days': len(portfolio_returns)
    }


def backtest_benchmark(
    benchmark_returns: pd.DataFrame,
    initial_value: float = 100.0
) -> Dict:
    """Calculate benchmark performance."""
    if benchmark_returns is None:
        return None

    returns = benchmark_returns.iloc[:, 0]
    benchmark_value = initial_value * (1 + returns).cumprod()

    total_return = (benchmark_value.iloc[-1] / initial_value) - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    annual_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0

    rolling_max = benchmark_value.cummax()
    drawdown = (benchmark_value - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    return {
        'benchmark_value': benchmark_value,
        'daily_returns': returns.tolist(),
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'dates': [str(d.date()) for d in benchmark_returns.index]
    }


def get_portfolio_allocation_summary(weights: Dict[str, float]) -> Dict:
    """Get a summary of portfolio allocation by category."""
    categories = {
        'thai_export': ['DELTA.BK', 'HANA.BK', 'STA.BK', 'IVL.BK', 'PTTGC.BK'],
        'thai_domestic': ['CPALL.BK', 'AOT.BK', 'BDMS.BK', 'SCB.BK', 'CPN.BK', 'MINT.BK'],
        'global': ['WDC', 'THD'],
        'fixed_income': ['LEMB', 'VWOB', 'EMLC'],
        'fx': ['THB=X']
    }

    allocation = {}
    for category, tickers in categories.items():
        allocation[category] = sum(weights.get(t, 0) for t in tickers)

    return allocation
