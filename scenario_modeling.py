"""
Scenario Modeling Module
Implements stress testing with Geometric Brownian Motion (GBM) simulation.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class StressScenario:
    """Configuration for a stress scenario."""
    name: str
    affected_tickers: List[str]
    expected_drop_pct: float
    drop_std_pct: float
    volatility_increase_pct: float
    start_date: datetime
    duration_days: int = 252


def create_trump_tariff_scenario(
    thai_tickers: List[str],
    start_date: Optional[datetime] = None,
    duration_days: int = 252,
    expected_drop_pct: float = -0.10,
    drop_std_pct: float = 0.02,
    volatility_increase_pct: float = 0.10
) -> StressScenario:
    """Create a Trump tariffs stress scenario affecting Thai assets."""
    if start_date is None:
        start_date = datetime.now()

    return StressScenario(
        name="Trump Tariffs Impact on Thai Assets",
        affected_tickers=thai_tickers,
        expected_drop_pct=expected_drop_pct,
        drop_std_pct=drop_std_pct,
        volatility_increase_pct=volatility_increase_pct,
        start_date=start_date,
        duration_days=duration_days
    )


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    dt: float,
    n_simulations: int = 1000
) -> np.ndarray:
    """Simulate asset prices using Geometric Brownian Motion."""
    n_steps = int(T / dt)
    Z = np.random.standard_normal((n_steps, n_simulations))

    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt) * Z

    log_returns = drift + diffusion
    log_prices = np.vstack([np.zeros(n_simulations), np.cumsum(log_returns, axis=0)])
    prices = S0 * np.exp(log_prices)

    return prices


def apply_initial_shock(
    prices: np.ndarray,
    shock_mean: float,
    shock_std: float,
    n_simulations: int
) -> np.ndarray:
    """Apply an initial price shock to simulation paths."""
    shocks = 1 + np.random.normal(shock_mean, shock_std, n_simulations)
    shocks = np.maximum(shocks, 0.01)
    shocked_prices = prices * shocks
    return shocked_prices


def simulate_stress_scenario(
    scenario: StressScenario,
    statistics: Dict,
    last_prices: pd.Series,
    n_simulations: int = 1000,
    random_seed: Optional[int] = 42
) -> Dict:
    """Simulate asset prices under a stress scenario."""
    if random_seed is not None:
        np.random.seed(random_seed)

    mean_returns = statistics['mean_returns']
    std_returns = statistics['std_returns']
    tickers = mean_returns.index.tolist()

    T = scenario.duration_days / 252
    dt = 1 / 252
    n_steps = scenario.duration_days

    simulated_prices = {}
    simulated_returns = {}

    for ticker in tickers:
        S0 = last_prices[ticker]
        mu = mean_returns[ticker]
        sigma = std_returns[ticker]

        is_affected = ticker in scenario.affected_tickers

        if is_affected:
            stressed_sigma = sigma * (1 + scenario.volatility_increase_pct)
            prices = simulate_gbm(S0=S0, mu=mu, sigma=stressed_sigma, T=T, dt=dt, n_simulations=n_simulations)
            prices = apply_initial_shock(prices, scenario.expected_drop_pct, scenario.drop_std_pct, n_simulations)
        else:
            prices = simulate_gbm(S0=S0, mu=mu, sigma=sigma, T=T, dt=dt, n_simulations=n_simulations)

        simulated_prices[ticker] = prices
        returns = np.diff(prices, axis=0) / prices[:-1]
        simulated_returns[ticker] = returns

    dates = pd.date_range(start=scenario.start_date, periods=n_steps + 1, freq='B')

    return {
        'scenario': scenario,
        'simulated_prices': simulated_prices,
        'simulated_returns': simulated_returns,
        'dates': dates,
        'n_simulations': n_simulations,
        'n_steps': n_steps,
        'tickers': tickers
    }


def calculate_portfolio_simulation(
    simulation_results: Dict,
    weights: Dict[str, float],
    initial_value: float = 100.0
) -> Dict:
    """Calculate portfolio performance under the stress scenario."""
    tickers = simulation_results['tickers']
    n_simulations = simulation_results['n_simulations']
    n_steps = simulation_results['n_steps']
    dates = simulation_results['dates']

    portfolio_returns = np.zeros((n_steps, n_simulations))

    for ticker in tickers:
        if ticker in weights and weights[ticker] > 0:
            weight = weights[ticker]
            asset_returns = simulation_results['simulated_returns'][ticker]
            portfolio_returns += weight * asset_returns

    portfolio_values = initial_value * np.cumprod(1 + portfolio_returns, axis=0)
    portfolio_values = np.vstack([np.full(n_simulations, initial_value), portfolio_values])

    individual_simulations = {}
    for ticker in tickers:
        if ticker in weights and weights[ticker] > 0.001:
            prices = simulation_results['simulated_prices'][ticker]
            normalized = 100 * (prices / prices[0, :])
            individual_simulations[ticker] = {
                'mean': np.mean(normalized, axis=1).tolist(),
                'median': np.median(normalized, axis=1).tolist(),
                'percentile_5': np.percentile(normalized, 5, axis=1).tolist(),
                'percentile_95': np.percentile(normalized, 95, axis=1).tolist(),
                'all_final_returns': ((prices[-1, :] / prices[0, :]) - 1).tolist()
            }

    final_values = portfolio_values[-1, :]
    total_returns = (final_values / initial_value) - 1

    percentiles = {
        '5th': np.percentile(total_returns, 5),
        '25th': np.percentile(total_returns, 25),
        '50th': np.percentile(total_returns, 50),
        '75th': np.percentile(total_returns, 75),
        '95th': np.percentile(total_returns, 95)
    }

    mean_return = np.mean(total_returns)
    std_return = np.std(total_returns)
    var_95 = np.percentile(total_returns, 5)
    cvar_95 = np.mean(total_returns[total_returns <= var_95])
    prob_loss = np.mean(total_returns < 0)

    return {
        'portfolio_values': portfolio_values,
        'portfolio_returns': portfolio_returns,
        'total_returns': total_returns.tolist(),
        'dates': [d.strftime('%Y-%m-%d') for d in dates],
        'statistics': {
            'mean_return': mean_return,
            'std_return': std_return,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'prob_loss': prob_loss,
            'percentiles': percentiles
        },
        'paths': {
            'mean': np.mean(portfolio_values, axis=1).tolist(),
            'median': np.median(portfolio_values, axis=1).tolist(),
            'percentile_5': np.percentile(portfolio_values, 5, axis=1).tolist(),
            'percentile_95': np.percentile(portfolio_values, 95, axis=1).tolist()
        },
        'individual_simulations': individual_simulations,
        'scenario_name': simulation_results['scenario'].name,
        'n_simulations': n_simulations
    }


def run_stress_test(
    statistics: Dict,
    last_prices: pd.Series,
    weights: Dict[str, float],
    thai_tickers: List[str],
    start_date: Optional[datetime] = None,
    duration_days: int = 252,
    expected_drop_pct: float = -0.10,
    drop_std_pct: float = 0.02,
    volatility_increase_pct: float = 0.10,
    n_simulations: int = 1000,
    random_seed: Optional[int] = 42,
    initial_value: float = 100.0
) -> Dict:
    """Main function to run a stress test on a portfolio."""
    scenario = create_trump_tariff_scenario(
        thai_tickers=thai_tickers,
        start_date=start_date,
        duration_days=duration_days,
        expected_drop_pct=expected_drop_pct,
        drop_std_pct=drop_std_pct,
        volatility_increase_pct=volatility_increase_pct
    )

    print(f"\n=== Running Stress Test: {scenario.name} ===")
    print(f"Affected tickers: {scenario.affected_tickers}")
    print(f"Expected drop: {scenario.expected_drop_pct:.1%} (+/-{scenario.drop_std_pct:.1%})")
    print(f"Volatility increase: {scenario.volatility_increase_pct:.1%}")
    print(f"Duration: {scenario.duration_days} trading days")
    print(f"Simulations: {n_simulations}")

    simulation_results = simulate_stress_scenario(
        scenario=scenario,
        statistics=statistics,
        last_prices=last_prices,
        n_simulations=n_simulations,
        random_seed=random_seed
    )

    portfolio_results = calculate_portfolio_simulation(
        simulation_results=simulation_results,
        weights=weights,
        initial_value=initial_value
    )

    print(f"\n=== Stress Test Results ===")
    print(f"Mean Return: {portfolio_results['statistics']['mean_return']:.2%}")
    print(f"Std Return: {portfolio_results['statistics']['std_return']:.2%}")
    print(f"95% VaR: {portfolio_results['statistics']['var_95']:.2%}")
    print(f"95% CVaR: {portfolio_results['statistics']['cvar_95']:.2%}")
    print(f"Probability of Loss: {portfolio_results['statistics']['prob_loss']:.1%}")

    return {
        'scenario_config': {
            'name': scenario.name,
            'affected_tickers': scenario.affected_tickers,
            'expected_drop_pct': scenario.expected_drop_pct,
            'drop_std_pct': scenario.drop_std_pct,
            'volatility_increase_pct': scenario.volatility_increase_pct,
            'start_date': scenario.start_date.strftime('%Y-%m-%d'),
            'duration_days': scenario.duration_days
        },
        'portfolio_results': portfolio_results
    }
