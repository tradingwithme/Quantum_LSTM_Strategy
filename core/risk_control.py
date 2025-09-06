import pandas as pd
import numpy as np

# --- Risk Control ---
def check_stop_loss(trades, current_price, stop_loss_percentage=0.05):
    """
    Checks if a stop-loss condition is met for any open positions.
    Args:
        trades (list): List of trade dictionaries.
        current_price (float): The current price of the asset.
        stop_loss_percentage (float): The percentage drop from the buy price that triggers a stop-loss.
    Returns:
        bool: True if a stop-loss is triggered, False otherwise.
        dict or None: The trade dictionary for the triggered stop-loss, or None.
    """
    for trade in reversed(trades): # Check latest trades first
        if trade['type'] == 'buy' and trade.get('is_open', True): # Assuming 'is_open' flag or similar
            buy_price = trade['price']
            if current_price < buy_price * (1 - stop_loss_percentage):
                print(f"Stop-loss triggered: Current Price {current_price:.2f} dropped below {buy_price * (1 - stop_loss_percentage):.2f} (Stop Loss at {stop_loss_percentage*100:.2f}%).")
                return True, trade # Return True and the trade that triggered it
    return False, None


def enforce_cash_limit(current_cash, min_cash_buffer=1000):
    """
    Checks if the current cash is below a minimum buffer.
    Args:
        current_cash (float): The current amount of cash in the portfolio.
        min_cash_buffer (float): The minimum amount of cash to keep as a buffer.
    Returns:
        bool: True if cash is below the limit, False otherwise.
    """
    return current_cash < min_cash_buffer


def rebalance_portfolio(current_portfolio_value, cash, positions, optimal_weights, current_prices):
    """
    Calculates necessary trades to rebalance the portfolio to target optimal weights.
    Args:
        current_portfolio_value (float): The total current value of the portfolio (cash + positions).
        cash (float): Current cash holding.
        positions (dict): Dictionary mapping ticker to quantity held.
        optimal_weights (dict): Dictionary mapping ticker to target weight (0-1).
        current_prices (dict): Dictionary mapping ticker to current price.
    Returns:
        list: A list of trade actions (buy/sell) needed for rebalancing.
    """
    trade_actions = []
    # Calculate target allocation value for each asset
    target_allocations = {ticker: weight * current_portfolio_value for ticker, weight in optimal_weights.items()}

    # Calculate current allocation value for each asset
    current_allocations = {ticker: positions.get(ticker, 0) * current_prices.get(ticker, 0) for ticker in optimal_weights.keys()}

    # Determine necessary trades
    for ticker in optimal_weights.keys():
        target_value = target_allocations.get(ticker, 0)
        current_value = current_allocations.get(ticker, 0)
        current_price = current_prices.get(ticker, 0)

        if current_price <= 0:
             continue # Cannot trade if price is zero or negative

        # Difference in value needed
        value_difference = target_value - current_value

        if value_difference > 0: # Need to buy
            # Calculate quantity to buy. Use current cash constraint.
            # Only buy if we have enough cash and need to increase position
            buy_quantity = int(value_difference / current_price)
            cost = buy_quantity * current_price
            if buy_quantity > 0 and cost <= cash:
                 trade_actions.append({'type': 'buy', 'ticker': ticker, 'quantity': buy_quantity, 'price': float(current_price)})
                 cash -= cost # Simulate cash change for subsequent trades in this rebalancing step
                 print(f"Rebalance: Plan to BUY {buy_quantity} of {ticker} at {current_price:.2f}")
            # Note: This simple logic doesn't guarantee reaching the exact target allocation if cash is limited.
            # More sophisticated rebalancing might prioritize assets or use fractional shares.


        elif value_difference < 0: # Need to sell
            # Calculate quantity to sell
            sell_quantity = int(abs(value_difference) / current_price)
            current_quantity_held = positions.get(ticker, 0)
            # Only sell if we hold the asset and need to decrease position
            quantity_to_sell = min(sell_quantity, current_quantity_held)
            if quantity_to_sell > 0:
                 trade_actions.append({'type': 'sell', 'ticker': ticker, 'quantity': quantity_to_sell, 'price': float(current_price)})
                 cash += quantity_to_sell * current_price # Simulate cash change
                 print(f"Rebalance: Plan to SELL {quantity_to_sell} of {ticker} at {current_price:.2f}")

    # Note: Executing these trade_actions would require updating the actual cash and position holdings
    return trade_actions