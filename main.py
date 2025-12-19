"""
Portfolio Optimization with Stress Testing
Main orchestration file that runs the complete analysis pipeline.

Requirements:
- Download 10 years of data from Yahoo Finance
- Clean data (NAs only, no outlier removal)
- Optimize portfolio: Max return, volatility <= 10%, max weight 10%, no shorting
- Exclude DELTA.BK from portfolio
- Compare to S&P 500 benchmark
- Run stress test: Trump tariffs scenario on Thai assets
- Output JSON and visualizations to /reports subfolder
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from data_loader import load_and_prepare_data, get_thai_tickers
from portfolio_optimization import (
    optimize_portfolio,
    backtest_portfolio,
    backtest_benchmark,
    get_portfolio_allocation_summary
)
from scenario_modeling import run_stress_test
from visualization import generate_all_visualizations


# Default ticker list
DEFAULT_TICKERS = [
    # Thai stocks (export-oriented)
    'DELTA.BK', 'HANA.BK', 'STA.BK', 'IVL.BK', 'PTTGC.BK',
    # Thai stocks (domestic)
    'CPALL.BK', 'AOT.BK', 'BDMS.BK', 'SCB.BK', 'CPN.BK', 'MINT.BK',
    # Global exposure
    'WDC', 'THD',
    # Fixed income
    'LEMB', 'VWOB', 'EMLC',
    # FX
    'THB=X'
]

# Tickers to exclude from portfolio
EXCLUDED_TICKERS = ['DELTA.BK']


def run_portfolio_analysis(
    tickers: list = None,
    excluded_tickers: list = None,
    years: int = 10,
    max_volatility: float = 0.10,
    max_weight: float = 0.10,
    risk_free_rate: float = 0.02,
    stress_drop_pct: float = -0.10,
    stress_drop_std: float = 0.02,
    stress_vol_increase: float = 0.10,
    stress_duration_days: int = 252,
    n_simulations: int = 1000,
    output_dir: str = 'result'
) -> dict:
    """Run the complete portfolio analysis pipeline."""

    if tickers is None:
        tickers = DEFAULT_TICKERS

    if excluded_tickers is None:
        excluded_tickers = EXCLUDED_TICKERS

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("PORTFOLIO OPTIMIZATION WITH STRESS TESTING")
    print("=" * 60)
    print(f"\nExcluded tickers: {excluded_tickers}")
    print(f"Output directory: {output_dir}")

    # Step 1: Load and prepare data
    print("\n" + "=" * 60)
    print("STEP 1: LOADING DATA")
    print("=" * 60)

    data = load_and_prepare_data(
        tickers=tickers,
        years=years,
        include_benchmark=True
    )

    # Step 2: Optimize portfolio
    print("\n" + "=" * 60)
    print("STEP 2: PORTFOLIO OPTIMIZATION")
    print("=" * 60)

    optimization_result = optimize_portfolio(
        mean_returns=data['statistics']['mean_returns'],
        cov_matrix=data['statistics']['cov_matrix'],
        max_volatility=max_volatility,
        max_weight=max_weight,
        excluded_tickers=excluded_tickers,
        risk_free_rate=risk_free_rate
    )

    print(f"\nOptimization {'succeeded' if optimization_result['optimization_success'] else 'failed'}!")
    print(f"Expected Return: {optimization_result['portfolio_return']:.2%}")
    print(f"Portfolio Volatility: {optimization_result['portfolio_volatility']:.2%}")
    print(f"Sharpe Ratio: {optimization_result['sharpe_ratio']:.4f}")

    print("\nOptimal Weights (non-zero):")
    for ticker, weight in sorted(optimization_result['weights'].items(), key=lambda x: -x[1]):
        if weight > 0.001:
            print(f"  {ticker}: {weight:.2%}")

    # Step 3: Backtest portfolio
    print("\n" + "=" * 60)
    print("STEP 3: BACKTESTING")
    print("=" * 60)

    backtest_result = backtest_portfolio(
        weights=optimization_result['weights'],
        returns=data['returns']
    )

    print(f"\nBacktest Results ({backtest_result['start_date']} to {backtest_result['end_date']}):")
    print(f"Total Return: {backtest_result['total_return']:.2%}")
    print(f"Annual Return: {backtest_result['annual_return']:.2%}")
    print(f"Annual Volatility: {backtest_result['annual_volatility']:.2%}")
    print(f"Sharpe Ratio: {backtest_result['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {backtest_result['max_drawdown']:.2%}")

    # Step 4: Backtest benchmark
    benchmark_result = None
    if data['benchmark_returns'] is not None:
        benchmark_result = backtest_benchmark(data['benchmark_returns'])
        print(f"\nBenchmark (S&P 500) Results:")
        print(f"Total Return: {benchmark_result['total_return']:.2%}")
        print(f"Annual Return: {benchmark_result['annual_return']:.2%}")
        print(f"Annual Volatility: {benchmark_result['annual_volatility']:.2%}")
        print(f"Sharpe Ratio: {benchmark_result['sharpe_ratio']:.4f}")
        print(f"Max Drawdown: {benchmark_result['max_drawdown']:.2%}")

    # Step 5: Run stress test
    print("\n" + "=" * 60)
    print("STEP 4: STRESS TESTING")
    print("=" * 60)

    last_prices = data['prices'].iloc[-1]

    stress_result = run_stress_test(
        statistics=data['statistics'],
        last_prices=last_prices,
        weights=optimization_result['weights'],
        thai_tickers=data['thai_tickers'],
        expected_drop_pct=stress_drop_pct,
        drop_std_pct=stress_drop_std,
        volatility_increase_pct=stress_vol_increase,
        duration_days=stress_duration_days,
        n_simulations=n_simulations
    )

    # Step 6: Compile results structure for visualization
    # Get allocation summary
    allocation_summary = get_portfolio_allocation_summary(optimization_result['weights'])

    # Build visualization-compatible structure
    viz_results = {
        'optimal_weights': optimization_result['weights'],
        'portfolio_statistics': {
            'expected_annual_return': optimization_result['portfolio_return'],
            'annual_volatility': optimization_result['portfolio_volatility'],
            'sharpe_ratio': optimization_result['sharpe_ratio']
        },
        'excluded_tickers': excluded_tickers,
        'backtest_results': {
            'dates': backtest_result['dates'],
            'daily_returns': backtest_result['daily_returns'],
            'total_return': backtest_result['total_return'],
            'annual_return': backtest_result['annual_return'],
            'annual_volatility': backtest_result['annual_volatility'],
            'sharpe_ratio': backtest_result['sharpe_ratio'],
            'max_drawdown': backtest_result['max_drawdown'],
            'start_date': backtest_result['start_date'],
            'end_date': backtest_result['end_date'],
            'individual_values': backtest_result['individual_values'],
            'individual_returns': backtest_result['individual_returns']
        },
        'benchmark_results': None,
        'stress_test_results': {
            'scenario_config': stress_result['scenario_config'],
            'portfolio_results': stress_result['portfolio_results']
        }
    }

    if benchmark_result is not None:
        viz_results['benchmark_results'] = {
            'dates': benchmark_result['dates'],
            'daily_returns': benchmark_result['daily_returns'],
            'total_return': benchmark_result['total_return'],
            'annual_return': benchmark_result['annual_return'],
            'annual_volatility': benchmark_result['annual_volatility'],
            'sharpe_ratio': benchmark_result['sharpe_ratio'],
            'max_drawdown': benchmark_result['max_drawdown']
        }

    # Step 7: Generate visualizations
    print("\n" + "=" * 60)
    print("STEP 5: GENERATING VISUALIZATIONS")
    print("=" * 60)

    generate_all_visualizations(
        results=viz_results,
        output_dir=output_dir
    )

    # Step 8: Compile and save results
    print("\n" + "=" * 60)
    print("STEP 6: SAVING RESULTS")
    print("=" * 60)

    # Compile full results
    results = {
        'run_timestamp': datetime.now().isoformat(),
        'parameters': {
            'years_of_data': years,
            'max_volatility': max_volatility,
            'max_weight': max_weight,
            'risk_free_rate': risk_free_rate,
            'excluded_tickers': excluded_tickers,
            'n_simulations': n_simulations,
            'stress_scenario': {
                'expected_drop_pct': stress_drop_pct,
                'drop_std_pct': stress_drop_std,
                'volatility_increase_pct': stress_vol_increase,
                'duration_days': stress_duration_days
            }
        },
        'data_summary': {
            'tickers_requested': tickers,
            'tickers_available': data['tickers'],
            'tickers_removed': data['removed_tickers'],
            'thai_tickers': data['thai_tickers'],
            'date_range': {
                'start': str(data['prices'].index[0].date()),
                'end': str(data['prices'].index[-1].date())
            },
            'num_observations': len(data['returns'])
        },
        'optimization': {
            'optimal_weights': optimization_result['weights'],
            'expected_return': optimization_result['portfolio_return'],
            'expected_volatility': optimization_result['portfolio_volatility'],
            'sharpe_ratio': optimization_result['sharpe_ratio'],
            'optimization_success': optimization_result['optimization_success'],
            'constraints': optimization_result['constraints'],
            'allocation_by_category': allocation_summary
        },
        'backtest': {
            'total_return': backtest_result['total_return'],
            'annual_return': backtest_result['annual_return'],
            'annual_volatility': backtest_result['annual_volatility'],
            'sharpe_ratio': backtest_result['sharpe_ratio'],
            'max_drawdown': backtest_result['max_drawdown'],
            'start_date': backtest_result['start_date'],
            'end_date': backtest_result['end_date'],
            'num_days': backtest_result['num_days'],
            'daily_returns': backtest_result['daily_returns'],
            'dates': backtest_result['dates']
        },
        'benchmark': None,
        'stress_test': {
            'scenario_config': stress_result['scenario_config'],
            'statistics': stress_result['portfolio_results']['statistics'],
            'paths': stress_result['portfolio_results']['paths'],
            'total_returns': stress_result['portfolio_results']['total_returns'],
            'dates': stress_result['portfolio_results']['dates'],
            'individual_simulations': stress_result['portfolio_results']['individual_simulations']
        }
    }

    # Add benchmark if available
    if benchmark_result is not None:
        results['benchmark'] = {
            'ticker': '^GSPC',
            'name': 'S&P 500',
            'total_return': benchmark_result['total_return'],
            'annual_return': benchmark_result['annual_return'],
            'annual_volatility': benchmark_result['annual_volatility'],
            'sharpe_ratio': benchmark_result['sharpe_ratio'],
            'max_drawdown': benchmark_result['max_drawdown'],
            'daily_returns': benchmark_result['daily_returns'],
            'dates': benchmark_result['dates']
        }

    # Save JSON results
    json_path = os.path.join(output_dir, 'portfolio_analysis_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {json_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nFiles generated in '{output_dir}/':")
    print("  - portfolio_analysis_results.json")
    print("  - portfolio_allocation.png")
    print("  - backtest_performance.png")
    print("  - stress_simulation.png")
    print("  - stress_returns_distribution.png")
    print("  - individual_asset_stress.png")
    print("  - correlation_matrix.png")
    print("  - drawdown_analysis.png")
    print("  - summary_dashboard.png")

    return results


if __name__ == '__main__':
    results = run_portfolio_analysis()
