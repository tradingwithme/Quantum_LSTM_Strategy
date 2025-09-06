import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
from functools import reduce
from core.preprocess_data import preprocess_data, create_sequences 
from core.modeling import LSTMModel, train_model 
from core.evaluate_model import evaluate_model as evaluate_model_standalone 
from core.simulation import run_simulation_strategy 
from core.report import generate_simulation_report 
from core.optimizer import (calculate_expected_returns_covariance, 
mean_variance_optimization_cvxpy, 
mean_variance_optimization_scipy, 
generate_efficient_frontier)
from core.metrics import (calculate_max_drawdown, 
calculate_cvar, 
portfolio_performance)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stock trading strategy workflow.")
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT'], help='List of tickers to process.')
    parser.add_argument('--sequence_length', type=int, default=20, help='Number of time steps in each sequence.')
    parser.add_argument('--initial_cash', type=float, default=10000, help='Initial cash for simulation.')
    parser.add_argument('--trading_ticker', type=str, default='AAPL', help='Ticker for trading simulation.')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Number of units in LSTM hidden layers.')
    parser.add_argument('--dropout_prob', type=float, default=0.3, help='Dropout rate for regularization.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth', help='Path to save/load the best model state dictionary.')
    parser.add_argument('--config_path', type=str, default='models/model_config.json', help='Path to save/load the model configuration.')
    parser.add_argument('--skip_training', action='store_true', help='Skip model training and only run simulation with existing model.')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save output files (e.g., equity curve plot).')
    parser.add_argument('--core_dir', type=str, default='core', help='Directory to save core metrics.')
    parser.add_argument('--log_evaluation_details', action='store_true', help='Log detailed evaluation results.')
    # Add arguments for other optional components
    parser.add_argument('--run_portfolio_optimization', action='store_true', help='Run classical portfolio optimization after simulation.')
    parser.add_argument('--enable_risk_control_analysis', action='store_true', help='Perform risk control analysis after simulation (e.g., calculate drawdown/CVaR).')
    parser.add_argument('--quantum_enabled', action='store_true', help='Enable quantum-inspired/quantum features if available.')

    args = parser.parse_args()

    # Create output and core directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.core_dir, exist_ok=True)


    # --- 1. Preprocess the data ---
    # preprocess_data is imported from preprocess_data.py
    X_scaled, y, scaler, df_processed, sequence_dates, processed_tickers = preprocess_data(args.tickers, args.sequence_length)

    if X_scaled is not None and y is not None and not df_processed.empty and processed_tickers:
        print("\nData Preprocessing Successful.")

        input_dim = X_scaled.shape[-1] # Determine input_dim from preprocessed data
        print(f"Automatically set input_dim based on data shape: {input_dim}")

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # Determine the date range for the "current dataset" (evaluation data)
        eval_size = int(0.2 * len(X_tensor)) # Assuming 20% for evaluation
        train_size = len(X_tensor) - eval_size

        X_train_eval = X_tensor[:train_size] # Data used for main training/evaluation split
        y_train_eval = y_tensor[:train_size]
        sequence_dates_train_eval = sequence_dates[:train_size] # Dates corresponding to the end of each sequence

        X_eval = X_tensor[train_size:] # Data used for final evaluation/simulation
        y_eval = y_tensor[train_size:]
        sequence_dates_eval = sequence_dates[train_size:]


        # --- Prepare Fine-tuning Data ---
        X_finetune, y_finetune = None, None
        if len(sequence_dates_train_eval) > args.sequence_length:
             # Find the end date of the main training/evaluation data
             end_date_train_eval = sequence_dates_train_eval[-1]

             # Define the fine-tuning date range: 2 months to a year before the current dataset (end of X_eval)
             # The "current dataset" is interpreted as the final evaluation set here.
             # So, the range is 2 months to 1 year before the end of X_eval.
             if sequence_dates_eval: # Ensure eval dates exist
                 end_date_eval = sequence_dates_eval[-1]
                 fine_tune_end_date = end_date_eval - pd.DateOffset(months=2)
                 fine_tune_start_date = end_date_eval - pd.DateOffset(years=1)

                 print(f"\nPreparing fine-tuning data between {fine_tune_start_date.strftime('%Y-%m-%d')} and {fine_tune_end_date.strftime('%Y-%m-%d')}")

                 # Filter the original processed dataframe to this date range
                 # Need to be careful with date indexing and timezone if present
                 try:
                     df_finetune_range = df_processed.loc[fine_tune_start_date:fine_tune_end_date]
                     print(f"Filtered dataframe for fine-tuning shape: {df_finetune_range.shape}")

                     if not df_finetune_range.empty:
                         # Re-create sequences specifically for the fine-tuning date range
                         # Need the common features used during the initial preprocessing
                         # Assumes create_sequences is also imported from preprocess_data.py
                         inner_cols = [df_processed[t].columns.tolist() for t in df_processed.columns.levels[0]]
                         common_features = list(reduce(set.intersection, map(set, inner_cols)))
                         if 'Target' in common_features:
                             common_features.remove('Target')

                         X_finetune_np, y_finetune_np, sequence_dates_finetune = create_sequences(
                             df_finetune_range, common_features, args.sequence_length
                         )

                         if X_finetune_np.size > 0 and y_finetune_np.size > 0:
                              # Scale the fine-tuning data using the same scaler fitted on the main dataset
                              # Reshape X_finetune_np for scaling
                              X_finetune_reshaped_for_scaling = X_finetune_np.reshape(-1, X_finetune_np.shape[-1])

                            # Ensure the number of features matches the scaler's expected input
                              if hasattr(scaler, 'n_features_in_') and X_finetune_reshaped_for_scaling.shape[1] == scaler.n_features_in_:
                                   scaled_X_finetune_reshaped = scaler.transform(X_finetune_reshaped_for_scaling)
                                   # Reshape back to (samples, timesteps, features)
                                   X_finetune = torch.tensor(scaled_X_finetune_reshaped.reshape(X_finetune_np.shape[0], args.sequence_length, X_finetune_np.shape[-1]), dtype=torch.float32)
                                   y_finetune = torch.tensor(y_finetune_np, dtype=torch.float32)
                                   print(f"Fine-tuning data prepared. Shape of X_finetune: {X_finetune.shape}, Shape of y_finetune: {y_finetune.shape}")
                              elif not hasattr(scaler, 'n_features_in_'):
                                   print("Warning: scaler.n_features_in_ not available. Cannot verify feature count for fine-tuning scaling. Attempting scaling.")
                                   scaled_X_finetune_reshaped = scaler.transform(X_finetune_reshaped_for_scaling)
                                   X_finetune = torch.tensor(scaled_X_finetune_reshaped.reshape(X_finetune_np.shape[0], args.sequence_length, X_finetune_np.shape[-1]), dtype=torch.float32)
                                   y_finetune = torch.tensor(y_finetune_np, dtype=torch.float32)
                                   print(f"Fine-tuning data prepared (scaling not verified). Shape of X_finetune: {X_finetune.shape}, Shape of y_finetune: {y_finetune.shape}")
                              else:
                                   print(f"Warning: Feature count mismatch for fine-tuning data scaling. Expected {scaler.n_features_in_}, got {X_finetune_reshaped_for_scaling.shape[1]}. Skipping fine-tuning data preparation.")
                                   X_finetune, y_finetune = None, None # Reset to None if scaling fails

                         else: print("Warning: No sequences created for the fine-tuning date range.")

                     else: print("Warning: Filtered dataframe for fine-tuning date range is empty.")

                 except KeyError:
                     print(f"Warning: Could not filter dataframe for fine-tuning range using dates {fine_tune_start_date} to {fine_tune_end_date}. Check date index.")
                 except Exception as e:
                     print(f"An error occurred during fine-tuning data preparation: {e}")
             else:
                 print("Warning: Evaluation dates are not available. Cannot determine date range for fine-tuning.")
        print(f"\nTraining set size (main): {len(X_train_eval)}") # Note: This is the data *before* final eval split
        print(f"Evaluation set size (final): {len(X_eval)}")
        # --- 2. Train or Load the Model ---
        trained_model = None # Initialize model variable
        if not args.skip_training:
             print("\n--- Model Training ---")
             # train_model is imported from modeling.py
             trained_model, training_losses = train_model(
X_train_eval, y_train_eval, # Main training/evaluation split data
X_eval, y_eval, # Final evaluation data
X_finetune, y_finetune, # Fine-tuning data (can be None)
input_dim, args.hidden_dim, args.output_dim, args.dropout_prob,
args.lr, args.epochs, args.batch_size, args.model_path, args.config_path
)
             print("\nModel Training Complete.")
             # Optional: Plot training loss from script (requires matplotlib and a plotting function)
             if 'plot_training_loss' in globals(): plot_training_loss(training_losses, os.path.join(args.output_dir, 'training_loss.png'))

        elif os.path.exists(args.model_path) and os.path.exists(args.config_path):
            print(f"\n--- Loading Pre-trained Model ---")
            try:
                # Load config to get model dimensions
                with open(args.config_path, 'r') as f:
                    config = json.load(f)
                # Use data-derived input_dim if not in config, fallback to argument default
                loaded_input_dim = config.get('input_dim', input_dim if input_dim is not None else args.hidden_dim) # Fallback to hidden_dim is likely wrong, should be input_dim
                loaded_hidden_dim = config.get('hidden_dim', args.hidden_dim)
                loaded_output_dim = config.get('output_dim', args.output_dim)
                loaded_dropout_prob = config.get('dropout_prob', args.dropout_prob)

                # LSTMModel is imported from modeling.py
                trained_model = LSTMModel(loaded_input_dim, loaded_hidden_dim, loaded_output_dim, loaded_dropout_prob)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                trained_model.load_state_dict(torch.load(args.model_path, map_location=device))
                trained_model.to(device)
                print(f"Successfully loaded model from {args.model_path} and config from {args.config_path}")
                # Evaluate the loaded model using the standalone evaluation script
                # evaluate_model_standalone is imported from evaluate_model.py
                eval_accuracy = evaluate_model_standalone(trained_model, X_eval, y_eval, args.core_dir, args.output_dir, args.log_evaluation_details)
                print(f"Loaded Model Evaluation Accuracy (on final eval set): {eval_accuracy:.4f}")

            except Exception as e:
                print(f"Error loading model or config: {e}. Cannot run simulation.")
                trained_model = None # Ensure trained_model is None if loading fails
        else:
            print(f"\n--- Skipping Training ---")
            print(f"Model file '{args.model_path}' or config file '{args.config_path}' not found. Cannot run simulation without training or a loaded model.")


        # --- 3. Run Simulation ---
        if trained_model is not None:
             print("\n--- Trading Simulation ---")
             # Select a ticker for simulation from the successfully processed tickers
             trading_ticker_for_simulation = args.trading_ticker if args.trading_ticker in processed_tickers else (processed_tickers[0] if processed_tickers else None)

             if trading_ticker_for_simulation:
                 if trading_ticker_for_simulation != args.trading_ticker:
                      print(f"Warning: Specified trading ticker '{args.trading_ticker}' not in processed tickers. Using '{trading_ticker_for_simulation}' for simulation.")

                 # Need to get the common_features used during preprocessing to pass to run_simulation_strategy
                 # Assuming df_processed structure is maintained from preprocess_data call
                 inner_cols = [df_processed[t].columns.tolist() for t in df_processed.columns.levels[0]]
                 simulation_common_features = list(reduce(set.intersection, map(set, inner_cols)))
                 if 'Target' in simulation_common_features:
                     simulation_common_features.remove('Target')
                 # run_simulation_strategy is imported from simulation.py
                 final_equity, trades, equity_curve, simulation_dates_sim = run_simulation_strategy( # Renamed to avoid conflict
                     df_processed, trained_model, scaler, simulation_common_features,
                     args.sequence_length, args.initial_cash, trading_ticker_for_simulation, sequence_dates # Pass original sequence_dates for simulation start index logic
                 )
                 print(f"\nSimulation Final Equity: {final_equity:.2f}")
                 print(f"Number of Trades: {len(trades)}")

                 # Save trades and equity curve data (handled within run_simulation_strategy now)
                 # --- 4. Generate Report ---
                 print("\n--- Generating Simulation Report ---")
                 # generate_simulation_report is imported from report.py
                 generate_simulation_report(
simulation_trades_path=os.path.join(args.output_dir, 'simulation_trades.csv'),
equity_curve_path=os.path.join(args.output_dir, 'simulation_equity_curve.csv'),
output_dir=args.output_dir
)

                 # --- 5. Portfolio Optimization (Optional) ---
                 if args.run_portfolio_optimization:
                      print("\n--- Running Classical Portfolio Optimization ---")
                      # Assuming you want to optimize based on the returns of the tickers used in preprocessing
                      try:
                          returns_df = df_processed.xs('Returns', level=1, axis=1)
                          if not returns_df.empty:
                              # calculate_expected_returns_covariance is imported from config.optimizer
                              expected_returns, covariance_matrix = calculate_expected_returns_covariance(returns_df)
                              if not expected_returns.empty and not covariance_matrix.empty:
                                  # Example: Max Sharpe Ratio
                                  # mean_variance_optimization_cvxpy is imported from config.optimizer
                                  # Save results to core/optimizer_results
                                  optimal_weights_sharpe, perf_sharpe, _ = mean_variance_optimization_cvxpy(expected_returns, covariance_matrix, objective='max_sharpe', save_path_prefix=os.path.join(args.core_dir, 'optimizer_results/cvxpy_'))
                                  print(f"\nMax Sharpe Portfolio (CVXPY/SciPy): Return={perf_sharpe[0]:.2f}%, Volatility={perf_sharpe[1]:.2f}%, Sharpe={perf_sharpe[2]:.4f}")
                                  # Optional: Print optimal weights
                                  print("Optimal Weights:", dict(zip(expected_returns.index, optimal_weights_sharpe)))

                                  # Minimize Volatility using SciPy
                                  # mean_variance_optimization_scipy is imported from config.optimizer
                                  # Save results to core/optimizer_results
                                  optimal_weights_min_vol, perf_min_vol, _ = mean_variance_optimization_scipy(expected_returns, covariance_matrix, objective='min_volatility', save_path_prefix=os.path.join(args.core_dir, 'optimizer_results/scipy_'))
                                  print(f"\nMin Volatility Portfolio (SciPy): Return={perf_min_vol[0]:.2f}%, Volatility={perf_min_vol[1]:.2f}%, Sharpe={perf_min_vol[2]:.4f}")

                                  # Generate and plot Efficient Frontier
                                  # generate_efficient_frontier is imported from config.optimizer
                                  # Save frontier to core/optimizer_results
                                  frontier_df = generate_efficient_frontier(expected_returns, covariance_matrix, save_path=os.path.join(args.core_dir, 'optimizer_results/efficient_frontier.json'))
                                  if not frontier_df.empty:
                                       # Calculate performance of equal weight portfolio for plotting reference
                                       equal_weights = np.ones(len(expected_returns)) / len(expected_returns)
                                       # portfolio_performance is imported from metrics.py
                                       perf_equal = portfolio_performance(equal_weights, expected_returns, covariance_matrix)

                                       # You could add plotting for the frontier here if you have a plotting function
                                       # plot_efficient_frontier(frontier_df, perf_sharpe, perf_min_vol, perf_equal)

                                  else:
                                       print("Could not generate efficient frontier.")

                              else:
                                  print("Could not calculate expected returns or covariance matrix for optimization.")
                          else:
                              print("Returns data is empty. Cannot perform portfolio optimization.")
                      except Exception as e:
                          print(f"An error occurred during portfolio optimization: {e}")


                 # --- 6. Risk Control Analysis (Optional) ---
                 # This is for analysis *after* simulation, not during the simulation loop itself.
                 if args.enable_risk_control_analysis:
                      print("\n--- Running Risk Control Analysis ---")
                      # Example: Calculate Max Drawdown and CVaR using metrics.py
                      if equity_curve: # Check if simulation generated an equity curve
                           # calculate_max_drawdown and calculate_cvar are imported from metrics.py
                           max_dd = calculate_max_drawdown(equity_curve)
                           # Need returns for CVaR
                           equity_returns = pd.Series(equity_curve).pct_change().dropna()
                           cvar_val = calculate_cvar(equity_returns)

                           print(f"\nSimulation Max Drawdown: {max_dd:.2%}")
                           print(f"Simulation CVaR (95%): {cvar_val:.4f}")

                           # You could also add checks for stop-loss trigger points or cash limits
                           # based on the historical simulation data here, but this is less
                           # about *applying* risk control and more about *analyzing* the simulated
                           # strategy's risk profile.
                      else: print("No equity curve available for risk control analysis.")
             else: print("Error: No valid ticker available for simulation.")
        else: print("\nSkipping simulation as model training failed or model could not be loaded.")
    else: print("\nData preprocessing failed or no tickers were processed. Cannot proceed with model training and simulation.")