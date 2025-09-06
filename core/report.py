import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import numpy as np

# --- Reporting and Analysis ---
def calculate_max_drawdown(equity_curve):
    """
    Calculates Max Drawdown from an equity curve (list or array of portfolio values).
    Represents the largest peak-to-trough decline during the historical period.
    Returns: the maximum drawdown as a percentage (negative or zero).
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0
    arr = np.asarray(equity_curve, dtype=float)
    peak = np.maximum.accumulate(arr)
    drawdown = np.where(peak > 0, (arr - peak) / peak, 0.0)
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


def generate_simulation_report(simulation_trades_path='results/simulation_trades.csv',
                               equity_curve_path='results/simulation_equity_curve.csv',
                               output_dir='results',
                               report_filename='simulation_report.txt',
                               metrics_filename='simulation_metrics.json', # Added JSON filename
                               plot_filename='equity_curve.png'):
    """
    Generates a text report, saves metrics to JSON, and saves the equity curve plot.
    """
    print("--- Generating Simulation Report ---")
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, report_filename)
    metrics_path = os.path.join(output_dir, metrics_filename) # Path for JSON metrics
    plot_path = os.path.join(output_dir, plot_filename) # Path for the plot

    report_metrics = {} # Dictionary to store metrics for JSON and text report

    try:
        trades_df = pd.read_csv(simulation_trades_path) if os.path.exists(simulation_trades_path) else pd.DataFrame()
        equity_df = pd.read_csv(equity_curve_path, index_col='Date', parse_dates=True) if os.path.exists(equity_curve_path) else pd.DataFrame()

        # --- Calculate Metrics ---
        if not equity_df.empty:
            equity_curve = equity_df['Equity'].tolist()
            simulation_dates = equity_df.index.tolist()

            initial_equity = equity_curve[0] if equity_curve else 0
            final_equity = equity_curve[-1] if equity_curve else initial_equity
            total_return = (final_equity - initial_equity) / initial_equity if initial_equity > 0 else 0.0
            max_drawdown = calculate_max_drawdown(equity_curve)
            sharpe_ratio = calculate_sharpe_ratio(equity_curve)

            report_metrics['initial_equity'] = initial_equity
            report_metrics['final_equity'] = final_equity
            report_metrics['total_return'] = total_return
            report_metrics['max_drawdown'] = max_drawdown
            report_metrics['sharpe_ratio'] = sharpe_ratio

            # --- Generate and Save Plot ---
            if equity_curve and simulation_dates and len(simulation_dates) == len(equity_curve):
                plt.figure(figsize=(12, 6))
                plt.plot(simulation_dates, equity_curve, label='Equity Curve', color='blue')
                plt.title('Simulation Equity Curve Over Time')
                plt.xlabel('Date')
                plt.ylabel('Portfolio Value')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                print(f"Equity curve plot saved to: {plot_path}")
                report_metrics['equity_curve_plot_path'] = plot_path # Add plot path to metrics
            else:
                print("Could not generate equity curve plot due to data mismatch or missing data.")
                report_metrics['equity_curve_plot_status'] = 'Failed: Data mismatch or missing'


        else:
            print("No equity curve data found to calculate metrics or plot.")
            report_metrics['status'] = 'No equity curve data'


        # --- Add Trades Summary to Metrics ---
        if not trades_df.empty:
            report_metrics['total_trades'] = len(trades_df)
            report_metrics['buy_trades'] = len(trades_df[trades_df['type'] == 'buy'])
            report_metrics['sell_trades'] = len(trades_df[trades_df['type'] == 'sell'])
            # Optional: add more trade details if needed
        else:
            report_metrics['total_trades'] = 0
            report_metrics['buy_trades'] = 0
            report_metrics['sell_trades'] = 0
            print("No trade data found.")


        # --- Save Metrics to JSON ---
        try:
            # Use indent for pretty printing the JSON
            with open(metrics_path, 'w') as f:
                json.dump(report_metrics, f, indent=4)
            print(f"Simulation metrics saved to JSON: {metrics_path}")
        except Exception as e:
            print(f"Warning: Could not save simulation metrics to JSON: {e}")


        # --- Generate Text Report (optional, can use the metrics_dict) ---
        try:
            with open(report_path, 'w') as f:
                f.write("--- Trading Simulation Report ---\n\n")
                if 'status' in report_metrics and report_metrics['status'] == 'No equity curve data':
                    f.write("No simulation data available to generate a detailed report.\n")
                else:
                    f.write(f"Initial Equity: ${report_metrics.get('initial_equity', 0.0):.2f}\n")
                    f.write(f"Final Equity: ${report_metrics.get('final_equity', 0.0):.2f}\n")
                    f.write(f"Total Return: {report_metrics.get('total_return', 0.0):.2%}\n")
                    f.write(f"Max Drawdown: {report_metrics.get('max_drawdown', 0.0):.2%}\n")
                    f.write(f"Sharpe Ratio: {report_metrics.get('sharpe_ratio', 0.0):.4f}\n")
                    f.write(f"\nEquity curve plot saved to: {report_metrics.get('equity_curve_plot_path', 'N/A')}\n")

                f.write("\n--- Trades Summary ---\n")
                f.write(f"Total Number of Trades: {report_metrics.get('total_trades', 0)}\n")
                f.write(f"Number of Buy Trades: {report_metrics.get('buy_trades', 0)}\n")
                f.write(f"Number of Sell Trades: {report_metrics.get('sell_trades', 0)}\n")

                f.write("\n--- End of Report ---\n")

            print(f"Text report generated and saved to {report_path}")
        except Exception as e:
            print(f"Warning: Could not generate text report: {e}")


    except FileNotFoundError as e:
        print(f"Error: Required simulation file not found: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during report generation: {e}")