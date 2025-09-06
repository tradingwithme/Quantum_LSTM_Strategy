import pandas as pd
import numpy as np
from functools import reduce
from sklearn.preprocessing import MinMaxScaler
from ta import add_all_ta_features
from pandas_ta import rsi, macd
from pandas import concat, Series, DataFrame, MultiIndex, to_datetime, Timedelta
from math import ceil

# Import fetch_data from a separate file
from fetch_data import fetch_data


# --- Data Preprocessing ---
def add_indicators(ticker, add_all_features=False):
    """
    Fetches data for a ticker(s), adds technical indicators, and prepares a DataFrame.
    Can add all features from 'ta' or specific ones (MACD, RSI).
    Handles single ticker (string) or multiple tickers (list).
    Returns a MultiIndex DataFrame and list of processed tickers.
    Calls fetch_data (imported from fetch_data.py).
    """
    combined_df = pd.DataFrame()
    if isinstance(ticker, list):
        tickers = ticker
    elif isinstance(ticker, str):
        tickers = [ticker]
    else:
        raise Exception('Invalid ticker type')

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    ta_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

    processed_tickers = []

    print(f"Starting add_indicators for tickers: {tickers}")

    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        # Call the imported fetch_data function
        df = fetch_data(ticker)
        if df.empty:
            print(f"Warning: No data fetched for {ticker}. Skipping.")
            continue

        for col in required_cols:
            if col.lower() in df.columns:
                df.rename(columns={col.lower(): col}, inplace=True)

        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"Warning: Skipping ticker {ticker} due to missing required columns for TA: {missing}")
            continue

        available_ta_cols = [col for col in ta_cols if col in df.columns]
        df_for_ta = df[available_ta_cols].copy()

        try:
            if add_all_features:
                df_processed_ta = add_all_ta_features(df_for_ta, open="Open", high="High", low="Low", close="Close", volume="Volume")
                cols_to_drop_present = [col for col in available_ta_cols if col in df.columns]
                df = df.drop(columns=cols_to_drop_present).merge(df_processed_ta, left_index=True, right_index=True, how='left')
            else:
                if 'Close' in df.columns:
                    df = df.join(macd(df['Close']))
                    df = df.join(rsi(df['Close']))
                else:
                    print(f"Warning: 'Close' column not found for {ticker}. Cannot add MACD/RSI. Skipping ticker.")
                    continue

            if 'Close' in df.columns:
                df['Returns'] = df['Close'].pct_change()
                df['Target'] = (df['Returns'].shift(-1) > 0).astype(int)
            else:
                 print(f"Warning: 'Close' column not found for {ticker} after merging TA features. Cannot calculate Returns/Target. Skipping ticker.")
                 continue


            df.dropna(axis=1, how='all', inplace=True)
            df.ffill(inplace=True)

            if not df.empty:
                df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
                combined_df = pd.concat([combined_df, df], axis=1)
                processed_tickers.append(ticker)
        except Exception as e:
            print(f"Error processing ticker {ticker}: {e}. Skipping.")
            continue

    if combined_df.empty:
        print("Error: No valid data processed for any ticker. Combined DataFrame is empty.")
        return pd.DataFrame(), []

    combined_df.ffill(inplace=True)
    combined_df.dropna(inplace=True)

    if combined_df.empty:
        print("Error: Combined DataFrame is empty after final ffill and dropna.")
        return pd.DataFrame(), []

    return combined_df, processed_tickers

def create_sequences(dataframe, features, sequence_length):
    """
    Creates sequences of data and corresponding targets for LSTM.
    """
    X, y, sequence_dates = [], [], []
    tickers = dataframe.columns.levels[0]

    if not tickers.empty:
        num_possible_sequences = len(dataframe) - sequence_length

        if num_possible_sequences <= 0:
             print(f"Warning: Not enough data ({len(dataframe)} rows) to create sequences of length {sequence_length}. Need more than {sequence_length} rows.")
             return np.array([]), np.array([]), []

        for i in range(num_possible_sequences):
            sequence_data = dataframe.iloc[i : (i + sequence_length)]
            try:
                cols_to_select = [(ticker, feature) for ticker in tickers for feature in features if (ticker, feature) in sequence_data.columns]
                if not cols_to_select:
                     continue

                sequence_features_df = sequence_data.loc[:, cols_to_select]
                sequence_features_df = sequence_features_df.sort_index(axis=1)
                sequence_values = sequence_features_df.values

                if not np.isfinite(sequence_values).all():
                    continue

            except KeyError as e:
                 continue

            first_ticker = tickers[0]
            target_index = i + sequence_length
            if (first_ticker, 'Target') in dataframe.columns and target_index < len(dataframe):
                 target_value = dataframe[first_ticker]['Target'].iloc[target_index]
                 if not np.isfinite(target_value):
                     continue

                 X.append(sequence_values)
                 y.append(target_value)
                 sequence_dates.append(dataframe.index[target_index - 1])
            else:
                 pass

    X = np.array(X)
    y = np.array(y)

    if X.shape[0] != y.shape[0]:
        min_len = min(X.shape[0], y.shape[0])
        X = X[:min_len]
        y = y[:min_len]
        sequence_dates = sequence_dates[:min_len]

    return X, y, sequence_dates

def preprocess_data(tickers, sequence_length=10):
    """
    Orchestrates the data preprocessing steps: fetching, adding indicators,
    creating sequences, and scaling the features.
    Returns scaled features (X), targets (y), the fitted scaler object,
    the processed dataframe (df_processed), sequence dates, and list of processed tickers.
    Calls add_indicators and create_sequences.
    """
    print("--- Starting preprocess_data ---")
    df, processed_tickers = add_indicators(tickers, True)

    if df.empty:
        print('MultiIndex dataframe is empty after adding indicators. Cannot proceed.')
        print("--- preprocess_data finished with error ---")
        return None, None, None, pd.DataFrame(), [], []

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    if df.empty:
        print('DataFrame is empty after handling infinite values and NaNs. Cannot create sequences.')
        print("--- preprocess_data finished with error ---")
        return None, None, None, pd.DataFrame(), [], []

    inner_columns_per_ticker = [df[ticker].columns.tolist() for ticker in df.columns.levels[0]]
    if not inner_columns_per_ticker:
        print("Error: No ticker columns found in the DataFrame after cleaning.")
        print("--- preprocess_data finished with error ---")
        return None, None, None, df, [], processed_tickers

    if len(df.columns.levels[0]) > 1:
        common_inner_columns = list(reduce(set.intersection, map(set, inner_columns_per_ticker)))
    else:
        common_inner_columns = inner_columns_per_ticker[0]

    if not common_inner_columns:
        print("Error: No common inner columns found across tickers.")
        print("--- preprocess_data finished with error ---")
        return None, None, None, df, [], processed_tickers

    if 'Target' in common_inner_columns:
        common_inner_columns.remove('Target')

    X, y, sequence_dates = create_sequences(df, common_inner_columns, sequence_length)

    if X is None or y is None or X.size == 0 or y.size == 0:
        print("Error: No sequences created. Check data and sequence length.")
        print("--- preprocess_data finished with error ---")
        return None, None, None, df, [], processed_tickers

    scaler = MinMaxScaler()
    X_reshaped_for_scaling = X.reshape(-1, X.shape[-1])

    if X_reshaped_for_scaling.shape[0] == 0 or X_reshaped_for_scaling.shape[1] == 0:
        print("Error: X_reshaped_for_scaling is empty or has no features after creating sequences from cleaned data. Cannot scale.")
        print("--- preprocess_data finished with error ---")
        return None, None, None, df, sequence_dates, processed_tickers

    X_scaled_reshaped = scaler.fit_transform(X_reshaped_for_scaling)
    X_scaled = X_scaled_reshaped.reshape(X.shape[0], sequence_length, X_reshaped_for_scaling.shape[1])

    print(f"--- Finished preprocess_data ---")
    print(f"Finished preprocessing data for {len(processed_tickers)} tickers.")
    print(f"Processed tickers: {processed_tickers}")
    print(f"Shape of X_scaled: {X_scaled.shape}")
    print(f"Shape of y: {y.shape}")
    print(f"Number of sequences: {len(sequence_dates)}")

    return X_scaled, y, scaler, df, sequence_dates, processed_tickers