import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt 

# --- Simulation ---
def run_simulation_strategy(df, model, scaler, common_features, sequence_length, initial_cash, trading_ticker, sequence_dates, output_dir='results'):
    """
    Simulates the trading strategy over historical data using a PyTorch LSTM model
    and technical indicators (MACD, RSI). Trades are executed based on a combination
    of model prediction and indicator signals.
    Simulates trading on a single asset specified by trading_ticker.
    sequence_dates is used to determine the simulation start date.
    Saves simulation trades and equity curve to the specified output_dir.
    """
    cash = initial_cash
    position = 0
    trades = []
    equity_curve = []
    simulation_dates = []

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)


    if df.empty:
        print("Error: Input DataFrame for simulation is empty.")
        return initial_cash, trades, equity_curve, simulation_dates

    if trading_ticker not in df.columns.levels[0] or (trading_ticker, 'Close') not in df[trading_ticker].columns:
         print(f"Error: Trading ticker '{trading_ticker}' not found in the DataFrame or missing 'Close' price for simulation.")
         return initial_cash, trades, equity_curve, simulation_dates

    # Determine the starting index for the simulation
    start_index = sequence_length # Default start index

    if sequence_dates and not df.empty:
         try:
              last_sequence_end_date = sequence_dates[-1]
              last_sequence_end_index_in_df = df.index.get_loc(last_sequence_end_date)
              start_index = last_sequence_end_index_in_df + 1

              if start_index >= len(df):
                   print("Warning: Simulation start index is beyond the end of the data. No simulation performed.")
                   return initial_cash, trades, equity_curve, simulation_dates

         except KeyError:
              print(f"Warning: Last sequence end date {last_sequence_end_date} not found in df index. Starting backtest from the first date with enough historical data.")
              start_index = sequence_length
              if start_index >= len(df):
                   print("Warning: Not enough data for simulation after sequence length. No simulation performed.")
                   return initial_cash, trades, equity_curve, simulation_dates

         except Exception as e:
              print(f"An unexpected error occurred while finding backtest start index: {e}")
              start_index = sequence_length


    # Dynamically find MACD Histogram and RSI column names for the trading ticker
    macd_hist_col = None
    rsi_col = None
    if trading_ticker in df.columns.levels[0]:
        # Search for columns containing 'MACDh' and 'RSI' within the trading ticker's columns
        ticker_cols = df[trading_ticker].columns
        macd_hist_col_name = next((col for col in ticker_cols if 'MACDh' in col), None)
        rsi_col_name = next((col for col in ticker_cols if 'RSI' in col), None)

        if macd_hist_col_name:
            macd_hist_col = (trading_ticker, macd_hist_col_name)
        if rsi_col_name:
            rsi_col = (trading_ticker, rsi_col_name)


    if not macd_hist_col or not rsi_col:
         print(f"Error: Required technical indicator columns (MACDh or RSI) not found for ticker {trading_ticker}. Cannot simulate strategy.")
         if trading_ticker in df.columns.levels[0]:
              print(f"Available columns for {trading_ticker}: {list(df[trading_ticker].columns)}")
         else:
              print(f"Ticker {trading_ticker} not found in DataFrame columns.")
         return initial_cash, trades, equity_curve, simulation_dates


    device = next(model.parameters()).device # Get model's device

    for i in range(start_index, len(df)):
        current_date = df.index[i]
        try:
             current_price = df[trading_ticker, 'Close'].iloc[i]
             macd_hist = df[macd_hist_col].iloc[i]
             rsi_val = df[rsi_col].iloc[i]
        except IndexError:
             print(f"Error accessing data at index {i} for ticker {trading_ticker}. Dataframe might be shorter than expected or columns are missing.")
             equity_curve.append(cash + position * current_price if 'current_price' in locals() and current_price else cash)
             simulation_dates.append(current_date)
             continue
        except KeyError as e:
             print(f"Error accessing column {e} at index {i} for ticker {trading_ticker}. Dataframe columns might have changed.")
             equity_curve.append(cash + position * current_price if 'current_price' in locals() and current_price else cash)
             simulation_dates.append(current_date)
             continue

        sequence_end_index_for_prediction = i - 1
        if sequence_end_index_for_prediction < sequence_length -1 :
             equity_curve.append(cash + position * current_price)
             simulation_dates.append(current_date)
             continue

        sequence_start_index_for_prediction = max(0, sequence_end_index_for_prediction - sequence_length + 1)
        latest_data_sequence = df.loc[df.index[sequence_start_index_for_prediction] : df.index[sequence_end_index_for_prediction], (slice(None), common_features)]

        if scaler is None:
             print("Error: Scaler is not fitted. Cannot scale data for prediction.")
             equity_curve.append(cash + position * current_price)
             simulation_dates.append(current_date)
             continue

        latest_data_sequence = latest_data_sequence.sort_index(axis=1)

        # Ensure the number of features in the sequence matches what the scaler expects
        expected_features_scaler = scaler.n_features_in_ # Number of features the scaler was fitted on
        # Handle the case where scaler.n_features_in_ is not available (e.g., older sklearn versions or different scaler types)
        if not hasattr(scaler, 'n_features_in_'):
            print("Warning: scaler.n_features_in_ not available. Assuming feature count matches input data shape for scaling.")
            expected_features_scaler = latest_data_sequence.shape[1] # Use the shape of the input data

        num_features_in_sequence = latest_data_sequence.values.reshape(-1, latest_data_sequence.shape[-1]).shape[1]

        if num_features_in_sequence != expected_features_scaler:
             print(f"Error: Feature count mismatch for scaling at index {i}. Expected {expected_features_scaler}, got {num_features_in_sequence}. Skipping prediction for this step.")
             equity_curve.append(cash + position * current_price)
             simulation_dates.append(current_date)
             continue


        scaled_latest_data_sequence = scaler.transform(latest_data_sequence.values.reshape(-1, num_features_in_sequence))
        X_test_tensor = torch.tensor(scaled_latest_data_sequence.reshape(1, sequence_length, num_features_in_sequence), dtype=torch.float32).to(device)


        if model is None:
             print("Error: Model is not available. Cannot make predictions.")
             equity_curve.append(cash + position * current_price)
             simulation_dates.append(current_date)
             continue

        model.eval()
        with torch.no_grad():
            prediction = model(X_test_tensor).item()


        if prediction > 0.5 and position == 0 and macd_hist > 0 and rsi_val < 70:
            buy_quantity = (cash // current_price) if current_price > 0 else 0
            if buy_quantity > 0:
                position = buy_quantity
                cash -= buy_quantity * current_price
                trades.append({
                    'type': 'buy',
                    'date': current_date.strftime('%Y-%m-%d'),
                    'price': float(current_price),
                    'quantity': int(buy_quantity),
                    'prediction': float(prediction)
                })

        elif position > 0 and (prediction <= 0.5 or macd_hist < 0 or rsi_val >= 70):
            sell_quantity = position
            cash += sell_quantity * current_price
            position = 0
            trades.append({
                'type': 'sell',
                'date': current_date.strftime('%Y-%m-%d'),
                    'price': float(current_price),
                    'quantity': int(sell_quantity),
                    'prediction': float(prediction)
                })

        equity_curve.append(cash + position * current_price)
        simulation_dates.append(current_date)

    final_equity = cash + position * df[trading_ticker, 'Close'].iloc[-1] if position > 0 and not df.empty and (trading_ticker, 'Close') in df.columns else cash

    # Save trades and equity curve data
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(os.path.join(output_dir, 'simulation_trades.csv'), index=False)
        print(f"Simulation trades saved to {os.path.join(output_dir, 'simulation_trades.csv')}")

    if equity_curve and simulation_dates:
        equity_df = pd.DataFrame({'Date': simulation_dates, 'Equity': equity_curve})
        equity_df.set_index('Date', inplace=True)
        equity_df.to_csv(os.path.join(output_dir, 'simulation_equity_curve.csv'))
        print(f"Simulation equity curve saved to {os.path.join(output_dir, 'simulation_equity_curve.csv')}")

    # Optional: Plot equity curve from simulation script
    if equity_curve and simulation_dates and len(simulation_dates) == len(equity_curve):
        plt.figure(figsize=(12, 6))
        plt.plot(simulation_dates, equity_curve, label=f'{trading_ticker} Trading Strategy', color='blue')
        plt.title(f'{trading_ticker} Trading Strategy Over Time')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'equity_curve.png')) # Save the plot
        print(f"Equity curve plot saved to {os.path.join(output_dir, 'equity_curve.png')}")
        # plt.show() # Uncomment to show plot when running the script directly


    return final_equity, trades, equity_curve, simulation_dates