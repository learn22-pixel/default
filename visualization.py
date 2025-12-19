"""
Visualization Module
Creates comprehensive charts and plots from portfolio optimization results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from typing import Dict
import os


def set_style():
    """Set consistent plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['font.size'] = 10


def load_results(filepath: str) -> dict:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_optimal_weights_pie(results: dict, output_path: str):
    """Plot optimal portfolio weights as a pie chart."""
    set_style()
    weights = results['optimal_weights']
    non_zero = {k: v for k, v in weights.items() if v > 0.001}
    sorted_weights = dict(sorted(non_zero.items(), key=lambda x: -x[1]))

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = plt.cm.Set3(np.linspace(0, 1, len(sorted_weights)))

    wedges, texts, autotexts = ax.pie(
        [v * 100 for v in sorted_weights.values()],
        labels=list(sorted_weights.keys()),
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        explode=[0.02] * len(sorted_weights),
        textprops={'fontsize': 10}
    )
    for autotext in autotexts:
        autotext.set_fontweight('bold')

    stats = results['portfolio_statistics']
    excluded = results.get('excluded_tickers', [])
    textstr = f"Expected Return: {stats['expected_annual_return']*100:.1f}%\nVolatility: {stats['annual_volatility']*100:.1f}%\nSharpe Ratio: {stats['sharpe_ratio']:.2f}"
    if excluded:
        textstr += f"\n\nExcluded: {', '.join(excluded)}"
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text(-1.5, -1.3, textstr, fontsize=10, bbox=props)

    ax.set_title('Optimal Portfolio Weights\n(Max Return with Vol≤10%, Max Weight 10%)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_backtest_total_portfolio(results: dict, output_path: str):
    """Plot total portfolio backtest performance vs S&P 500 benchmark."""
    set_style()
    backtest = results['backtest_results']
    benchmark = results.get('benchmark_results')

    dates = [datetime.strptime(d, '%Y-%m-%d') for d in backtest['dates']]
    portfolio_value = [100]
    for r in backtest['daily_returns']:
        portfolio_value.append(portfolio_value[-1] * (1 + r))

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(dates, portfolio_value[1:], color='#2980b9', linewidth=2, label='Optimal Portfolio')

    if benchmark:
        bench_value = [100]
        for r in benchmark['daily_returns']:
            bench_value.append(bench_value[-1] * (1 + r))
        ax.plot(dates, bench_value[1:], color='#e74c3c', linewidth=2, linestyle='--', label='S&P 500')

    ax.axhline(y=100, color='gray', linestyle=':', linewidth=1, alpha=0.7)

    textstr = f"Portfolio:\n  Total: {backtest['total_return']*100:.1f}%\n  Annual: {backtest['annual_return']*100:.1f}%\n  Sharpe: {backtest['sharpe_ratio']:.2f}\n  Max DD: {backtest['max_drawdown']*100:.1f}%"
    if benchmark:
        textstr += f"\n\nS&P 500:\n  Total: {benchmark['total_return']*100:.1f}%\n  Annual: {benchmark['annual_return']*100:.1f}%"
    props = dict(boxstyle='round', facecolor='white', alpha=0.9)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props, family='monospace')

    ax.set_ylabel('Portfolio Value (Starting = 100)')
    ax.set_xlabel('Date')
    ax.set_title('Historical Backtest: Portfolio vs S&P 500 Benchmark', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.65))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_backtest_individual_stocks(results: dict, output_path: str):
    """Plot individual stock performance during backtest period."""
    set_style()
    backtest = results['backtest_results']
    individual = backtest.get('individual_values', {})
    if not individual:
        return

    dates = [datetime.strptime(d, '%Y-%m-%d') for d in backtest['dates']]
    n_stocks = len(individual)
    n_cols = 3
    n_rows = (n_stocks + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_stocks > 1 else [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, n_stocks))

    for idx, (ticker, values) in enumerate(individual.items()):
        ax = axes[idx]
        ax.plot(dates, values, color=colors[idx], linewidth=1.5)
        ax.axhline(y=100, color='gray', linestyle=':', linewidth=0.8)
        final_return = (values[-1] / 100 - 1) * 100
        ax.set_title(f'{ticker}\nReturn: {final_return:+.1f}%', fontsize=11, fontweight='bold')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.tick_params(axis='x', rotation=45)

    for idx in range(n_stocks, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Individual Stock Performance (Historical Backtest)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_stress_total_portfolio(results: dict, output_path: str):
    """Plot total portfolio stress test simulation paths."""
    set_style()
    stress = results['stress_test_results']
    paths = stress['portfolio_results']['paths']
    dates = [datetime.strptime(d, '%Y-%m-%d') for d in stress['portfolio_results']['dates']]
    stats = stress['portfolio_results']['statistics']

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.fill_between(dates, paths['percentile_5'], paths['percentile_95'], alpha=0.3, color='#3498db', label='5th-95th Percentile')
    ax.plot(dates, paths['median'], color='#2980b9', linewidth=2, label='Median')
    ax.plot(dates, paths['mean'], color='#e74c3c', linewidth=2, linestyle='--', label='Mean')
    ax.axhline(y=100, color='gray', linestyle=':', linewidth=1)

    config = stress['scenario_config']
    scenario_text = f"Scenario: {config['name']}\nDrop: {config['expected_drop_pct']*100:.0f}% (+/-{config['drop_std_pct']*100:.0f}%)\nVol Increase: {config['volatility_increase_pct']*100:.0f}%"
    ax.text(0.02, 0.02, scenario_text, transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    stats_text = f"Mean: {stats['mean_return']*100:.1f}%\nStd: {stats['std_return']*100:.1f}%\n95% VaR: {stats['var_95']*100:.1f}%\nP(Loss): {stats['prob_loss']*100:.1f}%"
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_ylabel('Portfolio Value (Starting = 100)')
    ax.set_xlabel('Date')
    ax.set_title('Stress Test Simulation: Total Portfolio (1 Year Forward)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_stress_individual_stocks(results: dict, output_path: str):
    """Plot individual stock stress test simulations."""
    set_style()
    stress = results['stress_test_results']
    individual = stress['portfolio_results'].get('individual_simulations', {})
    if not individual:
        return

    dates = [datetime.strptime(d, '%Y-%m-%d') for d in stress['portfolio_results']['dates']]
    affected = stress['scenario_config']['affected_tickers']
    n_stocks = len(individual)
    n_cols = 3
    n_rows = (n_stocks + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_stocks > 1 else [axes]

    for idx, (ticker, sim_data) in enumerate(individual.items()):
        ax = axes[idx]
        ax.fill_between(dates, sim_data['percentile_5'], sim_data['percentile_95'], alpha=0.3, color='#3498db')
        ax.plot(dates, sim_data['median'], color='#2980b9', linewidth=1.5)
        ax.plot(dates, sim_data['mean'], color='#e74c3c', linewidth=1, linestyle='--')
        ax.axhline(y=100, color='gray', linestyle=':', linewidth=0.8)

        mean_return = np.mean(sim_data['all_final_returns']) * 100
        title_color = 'red' if ticker in affected else 'black'
        marker = ' [STRESSED]' if ticker in affected else ''
        ax.set_title(f'{ticker}{marker}\nMean: {mean_return:+.1f}%', fontsize=10, fontweight='bold', color=title_color)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.tick_params(axis='x', rotation=45)

    for idx in range(n_stocks, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Individual Stock Stress Test Simulations', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_returns_distribution_historical(results: dict, output_path: str):
    """Plot historical returns distribution."""
    set_style()
    backtest = results['backtest_results']
    portfolio_returns = np.array(backtest['daily_returns']) * 100
    individual = backtest.get('individual_returns', {})

    n_plots = 1 + len(individual)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    ax = axes[0]
    ax.hist(portfolio_returns, bins=50, density=True, alpha=0.7, color='#2980b9', edgecolor='black', linewidth=0.3)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
    ax.axvline(x=np.mean(portfolio_returns), color='green', linestyle='-', linewidth=2)
    ax.set_title('Portfolio Daily Returns', fontweight='bold')
    ax.set_xlabel('Daily Return (%)')

    for idx, (ticker, returns) in enumerate(individual.items(), start=1):
        if idx >= len(axes):
            break
        ax = axes[idx]
        returns_pct = np.array(returns) * 100
        ax.hist(returns_pct, bins=40, density=True, alpha=0.7, color='#3498db', edgecolor='black', linewidth=0.3)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=0.8)
        ax.set_title(f'{ticker}\nMean: {np.mean(returns_pct):.2f}%', fontsize=10)

    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Historical Daily Returns Distribution', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_returns_distribution_stress(results: dict, output_path: str):
    """Plot stress test returns distribution."""
    set_style()
    stress = results['stress_test_results']
    portfolio_returns = np.array(stress['portfolio_results']['total_returns']) * 100
    individual = stress['portfolio_results'].get('individual_simulations', {})
    affected = stress['scenario_config']['affected_tickers']

    n_plots = 1 + len(individual)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    ax = axes[0]
    n, bins, patches = ax.hist(portfolio_returns, bins=50, density=True, alpha=0.7, edgecolor='black', linewidth=0.3)
    for patch, left_edge in zip(patches, bins[:-1]):
        patch.set_facecolor('#e74c3c' if left_edge < 0 else '#2ecc71')
    ax.axvline(x=np.mean(portfolio_returns), color='blue', linewidth=2)
    ax.axvline(x=np.percentile(portfolio_returns, 5), color='darkred', linestyle='--', linewidth=2)
    ax.set_title('Portfolio 1-Year Return (Stress)', fontweight='bold')

    for idx, (ticker, sim_data) in enumerate(individual.items(), start=1):
        if idx >= len(axes):
            break
        ax = axes[idx]
        returns_pct = np.array(sim_data['all_final_returns']) * 100
        n, bins, patches = ax.hist(returns_pct, bins=40, density=True, alpha=0.7, edgecolor='black', linewidth=0.3)
        for patch, left_edge in zip(patches, bins[:-1]):
            patch.set_facecolor('#e74c3c' if left_edge < 0 else '#2ecc71')
        title_color = 'red' if ticker in affected else 'black'
        marker = ' [STRESSED]' if ticker in affected else ''
        ax.set_title(f'{ticker}{marker}\nMean: {np.mean(returns_pct):.1f}%', fontsize=10, color=title_color)

    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Stress Test: 1-Year Returns Distribution', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_summary_dashboard(results: dict, output_path: str):
    """Create a comprehensive summary dashboard."""
    set_style()
    fig = plt.figure(figsize=(20, 14))

    # Weights pie
    ax1 = fig.add_subplot(2, 3, 1)
    weights = results['optimal_weights']
    non_zero = {k: v for k, v in weights.items() if v > 0.001}
    sorted_weights = dict(sorted(non_zero.items(), key=lambda x: -x[1]))
    colors = plt.cm.Set3(np.linspace(0, 1, len(sorted_weights)))
    ax1.pie([v * 100 for v in sorted_weights.values()], labels=list(sorted_weights.keys()), autopct='%1.1f%%', colors=colors, startangle=90, textprops={'fontsize': 8})
    ax1.set_title('Optimal Weights', fontweight='bold')

    # Backtest
    ax2 = fig.add_subplot(2, 3, 2)
    backtest = results['backtest_results']
    benchmark = results.get('benchmark_results')
    portfolio_value = [100]
    for r in backtest['daily_returns']:
        portfolio_value.append(portfolio_value[-1] * (1 + r))
    ax2.plot(portfolio_value, color='#2980b9', linewidth=1.5, label='Portfolio')
    if benchmark:
        bench_value = [100]
        for r in benchmark['daily_returns']:
            bench_value.append(bench_value[-1] * (1 + r))
        ax2.plot(bench_value, color='#e74c3c', linewidth=1.5, linestyle='--', label='S&P 500')
    ax2.axhline(y=100, color='gray', linestyle=':', linewidth=0.8)
    ax2.set_title(f"Backtest: {backtest['total_return']*100:.0f}% Return", fontweight='bold')
    ax2.legend(fontsize=8)

    # Stress paths
    ax3 = fig.add_subplot(2, 3, 3)
    stress = results['stress_test_results']
    paths = stress['portfolio_results']['paths']
    ax3.fill_between(range(len(paths['median'])), paths['percentile_5'], paths['percentile_95'], alpha=0.3, color='#3498db')
    ax3.plot(paths['median'], color='#2980b9', linewidth=1.5)
    ax3.axhline(y=100, color='gray', linestyle=':', linewidth=0.8)
    ax3.set_title('Stress Test Simulation', fontweight='bold')

    # Historical returns
    ax4 = fig.add_subplot(2, 3, 4)
    hist_returns = np.array(backtest['daily_returns']) * 100
    ax4.hist(hist_returns, bins=50, density=True, alpha=0.7, color='#3498db')
    ax4.axvline(x=np.mean(hist_returns), color='red', linewidth=2)
    ax4.set_title('Historical Daily Returns', fontweight='bold')

    # Stress returns
    ax5 = fig.add_subplot(2, 3, 5)
    stress_returns = np.array(stress['portfolio_results']['total_returns']) * 100
    n, bins, patches = ax5.hist(stress_returns, bins=50, density=True, alpha=0.7)
    for patch, left_edge in zip(patches, bins[:-1]):
        patch.set_facecolor('#e74c3c' if left_edge < 0 else '#2ecc71')
    ax5.set_title('Stress 1-Year Returns', fontweight='bold')

    # Metrics
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    stats = stress['portfolio_results']['statistics']
    port_stats = results['portfolio_statistics']
    excluded = results.get('excluded_tickers', [])

    metrics_text = f"""
    EXCLUDED TICKERS: {', '.join(excluded) if excluded else 'None'}

    PORTFOLIO METRICS
    {'='*40}
    Expected Return:  {port_stats['expected_annual_return']*100:>8.1f}%
    Volatility:       {port_stats['annual_volatility']*100:>8.1f}%
    Sharpe Ratio:     {port_stats['sharpe_ratio']:>8.2f}

    BACKTEST ({backtest['start_date']} to {backtest['end_date']})
    {'='*40}
    Total Return:     {backtest['total_return']*100:>8.1f}%
    Annual Return:    {backtest['annual_return']*100:>8.1f}%
    Max Drawdown:     {backtest['max_drawdown']*100:>8.1f}%

    STRESS TEST
    {'='*40}
    Mean Return:      {stats['mean_return']*100:>8.1f}%
    95% VaR:          {stats['var_95']*100:>8.1f}%
    P(Loss):          {stats['prob_loss']*100:>8.1f}%
    """
    ax6.text(0.1, 0.95, metrics_text, transform=ax6.transAxes, fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('Portfolio Optimization & Stress Testing Summary', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_all_visualizations(results: dict, output_dir: str):
    """Generate all visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    plot_optimal_weights_pie(results, f"{output_dir}/01_optimal_weights_pie.png")
    plot_backtest_total_portfolio(results, f"{output_dir}/02_backtest_total_portfolio.png")
    plot_backtest_individual_stocks(results, f"{output_dir}/03_backtest_individual_stocks.png")
    plot_stress_total_portfolio(results, f"{output_dir}/04_stress_total_portfolio.png")
    plot_stress_individual_stocks(results, f"{output_dir}/05_stress_individual_stocks.png")
    plot_returns_distribution_historical(results, f"{output_dir}/06_returns_distribution_historical.png")
    plot_returns_distribution_stress(results, f"{output_dir}/07_returns_distribution_stress.png")
    plot_summary_dashboard(results, f"{output_dir}/08_summary_dashboard.png")

    print("\n" + "=" * 60)
    print("ALL VISUALIZATIONS COMPLETE")
    print("=" * 60)
