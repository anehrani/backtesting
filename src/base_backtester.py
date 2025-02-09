#!/usr/bin/env python
"""
Backtesting Script with Multiple Execution-Price Methods (Functional Style)

Input CSV (or dummy data) must contain:
  • date      : Date/time (up to seconds)
  • Open      : Opening price
  • High      : Highest price of the period
  • Low       : Lowest price of the period
  • Close     : Closing price
  • Volume    : Volume

Trade signals are provided to the backtest as a list of dictionaries.
Each signal should be of the form:
    {
        "symbol": str,
        "date": "YYYYMMDDHHMMSS",
        "order": "buy" or "sell",
        "size": float (in [0, 1])
    }
For every symbol there will be separate data.
"""

import pandas as pd
import numpy as np
import typer
from typing import Optional, List, Dict, Any, Tuple

app = typer.Typer(help="Functional Backtesting CLI: simulate trades and generate performance metrics.")

###############################################################################
# Helper functions
###############################################################################

def parse_signal_date(date_str: str) -> pd.Timestamp:
    """Parse a signal date given as YYYYMMDDHHMMSS."""
    return pd.to_datetime(date_str, format="%Y%m%d%H%M%S")

def merge_signals(
    df: pd.DataFrame, 
    trade_signals: List[Dict[str, Any]], 
    symbol: Optional[str] = None,
    threshold_seconds: Optional[float] = None
) -> pd.DataFrame:
    """
    Merge trade signals into the DataFrame by matching on the date.
    
    For each signal:
      - If an exact match is found in the data's "date" column, assign that signal.
      - Otherwise, search for the closest date by checking the tick immediately after
        (using searchsorted) and the tick immediately before the signal date.
      - If a candidate is found, optionally check if the time difference is within
        a threshold (if threshold_seconds is provided). If not, skip the signal.
    
    Args:
        df (pd.DataFrame): Input data containing a "date" column (should be sorted).
        trade_signals (List[Dict[str, Any]]): List of trade signals.
        symbol (Optional[str]): If provided, only signals matching this symbol are merged.
        threshold_seconds (Optional[float]): Optional threshold in seconds; if the closest 
            date difference exceeds this threshold, the signal is skipped.
    
    Returns:
        pd.DataFrame: A new DataFrame with two additional columns: 'order' and 'size'.
    """
    df = df.copy()
    df["order"] = None
    df["size"] = None
    df.sort_values("date", inplace=True)  # ensure data is sorted
    for signal in trade_signals:
        if symbol is not None and signal.get("symbol") != symbol:
            continue
        try:
            sig_date = parse_signal_date(signal["date"])
        except Exception as e:
            print(f"Error parsing signal date {signal['date']}: {e}")
            continue

        # Check for an exact match.
        exact_match = df["date"] == sig_date
        if exact_match.any():
            df.loc[exact_match, "order"] = signal["order"]
            df.loc[exact_match, "size"] = signal["size"]
            continue

        # Find the closest candidate using searchsorted.
        pos = df["date"].searchsorted(sig_date)
        candidate_index = None
        candidate_diff = None
        # Check candidate tick immediately after the signal date.
        if pos < len(df):
            candidate_after = df.iloc[pos]["date"]
            diff_after = abs((candidate_after - sig_date).total_seconds())
            candidate_index = pos
            candidate_diff = diff_after
        # Check candidate tick immediately before the signal date.
        if pos > 0:
            candidate_before = df.iloc[pos - 1]["date"]
            diff_before = abs((sig_date - candidate_before).total_seconds())
            if candidate_index is None or diff_before < candidate_diff:
                candidate_index = pos - 1
                candidate_diff = diff_before
        
        # If a threshold is provided, skip the signal if the time difference is too large.
        if threshold_seconds is not None and candidate_diff > threshold_seconds:
            continue
        
        # Assign the signal if a candidate row was found.
        if candidate_index is not None:
            df.loc[df.index[candidate_index], "order"] = signal["order"]
            df.loc[df.index[candidate_index], "size"] = signal["size"]
    return df

def get_execution_price(
    row: pd.Series, 
    side: str, 
    execution_method: str, 
    slippage_rate: float, 
    bid_ask_spread: float
) -> float:
    """
    Compute the execution price for a given row and side using the selected method.
    """
    if execution_method == "mid":
        return 0.5 * (row["High"] + row["Low"])
    elif execution_method == "vwap":
        return (row["High"] + row["Low"] + row["Close"]) / 3
    elif execution_method == "bid_ask":
        mid = 0.5 * (row["High"] + row["Low"])
        if side == "buy":
            return mid * (1 + bid_ask_spread / 2)
        elif side == "sell":
            return mid * (1 - bid_ask_spread / 2)
        else:
            return mid
    elif execution_method == "close":
        return row["Close"]
    elif execution_method == "slippage":
        mid = 0.5 * (row["High"] + row["Low"])
        if side == "buy":
            return mid * (1 + slippage_rate)
        elif side == "sell":
            return mid * (1 - slippage_rate)
        else:
            return mid
    else:
        return 0.5 * (row["High"] + row["Low"])

###############################################################################
# Simulation and Reporting functions
###############################################################################

def trade_simulator(
    df: pd.DataFrame,
    initial_capital: float,
    execution_method: str,
    slippage_rate: float,
    bid_ask_spread: float
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Simulate trades based on the signals in the DataFrame.
    Returns a new DataFrame (with portfolio values and daily returns) and a list of trade details.
    """
    df = df.copy()
    portfolio_values: List[float] = []
    trade_details: List[Dict[str, Any]] = []
    cash = initial_capital
    shares = 0.0
    invested_amount: Optional[float] = None  # Cash used to buy shares

    if execution_method == "next_open":
        # In next_open mode, orders are executed at the next candle's Open price.
        pending_order = None
        for i, row in df.iterrows():
            # Execute any pending order at the current candle's Open.
            if pending_order is not None:
                curr_open = row["Open"]
                if pending_order["order"] == "buy" and shares == 0:
                    amount_to_invest = pending_order["size"] * cash
                    if amount_to_invest > 0:
                        shares = amount_to_invest / curr_open
                        cash -= amount_to_invest
                        invested_amount = amount_to_invest
                        trade_details.append({
                            "Entry Date": pending_order["date"],
                            "Entry Price": curr_open,
                            "Shares": shares
                        })
                elif pending_order["order"] == "sell" and shares > 0:
                    proceeds = shares * curr_open
                    trade_return = (proceeds - invested_amount) / invested_amount if invested_amount else np.nan
                    if trade_details:
                        trade_details[-1].update({
                            "Exit Date": row["date"],
                            "Exit Price": curr_open,
                            "Trade Return": trade_return,
                        })
                    cash += proceeds
                    shares = 0.0
                    invested_amount = None
                pending_order = None

            # Check if a new signal exists at this candle.
            order = row["order"]
            if pd.notna(order):
                signal = str(order).lower().strip()
                if signal in ("buy", "sell"):
                    if signal == "buy" and shares == 0:
                        pending_order = {
                            "order": "buy",
                            "size": row["size"],
                            "date": row["date"]
                        }
                    elif signal == "sell" and shares > 0:
                        pending_order = {
                            "order": "sell",
                            "size": row["size"],
                            "date": row["date"]
                        }
            # Update portfolio value using the candle's Close.
            portfolio_val = cash + shares * row["Close"]
            portfolio_values.append(portfolio_val)
            df.at[i, "portfolio_value"] = portfolio_val

        # Execute any pending order at the end using the last candle's Open.
        if pending_order is not None:
            last_row = df.iloc[-1]
            curr_open = last_row["Open"]
            if pending_order["order"] == "buy" and shares == 0:
                amount_to_invest = pending_order["size"] * cash
                if amount_to_invest > 0:
                    shares = amount_to_invest / curr_open
                    cash -= amount_to_invest
                    invested_amount = amount_to_invest
                    trade_details.append({
                        "Entry Date": pending_order["date"],
                        "Entry Price": curr_open,
                        "Shares": shares
                    })
            elif pending_order["order"] == "sell" and shares > 0:
                proceeds = shares * curr_open
                trade_return = (proceeds - invested_amount) / invested_amount if invested_amount else np.nan
                if trade_details:
                    trade_details[-1].update({
                        "Exit Date": last_row["date"],
                        "Exit Price": curr_open,
                        "Trade Return": trade_return,
                    })
                cash += proceeds
                shares = 0.0
                invested_amount = None

        # If a position still remains open, close it at the last candle's Open.
        if shares > 0:
            last_row = df.iloc[-1]
            final_exec = last_row["Open"]
            proceeds = shares * final_exec
            trade_return = (proceeds - invested_amount) / invested_amount if invested_amount else np.nan
            if trade_details:
                trade_details[-1].update({
                    "Exit Date": last_row["date"],
                    "Exit Price": final_exec,
                    "Trade Return": trade_return,
                    "Open Trade": True
                })
            cash += proceeds
            shares = 0.0

        df["daily_return"] = df["portfolio_value"].pct_change().fillna(0)

    else:
        # Synchronous execution for methods other than next_open.
        for i, row in df.iterrows():
            order = row["order"]
            if pd.notna(order):
                signal = str(order).lower().strip()
                if signal == "buy" and shares == 0:
                    exec_price = get_execution_price(row, "buy", execution_method, slippage_rate, bid_ask_spread)
                    amount_to_invest = float(row["size"]) * cash
                    if amount_to_invest > 0:
                        shares = amount_to_invest / exec_price
                        cash -= amount_to_invest
                        invested_amount = amount_to_invest
                        trade_details.append({
                            "Entry Date": row["date"],
                            "Entry Price": exec_price,
                            "Shares": shares
                        })
                elif signal == "sell" and shares > 0:
                    exec_price = get_execution_price(row, "sell", execution_method, slippage_rate, bid_ask_spread)
                    proceeds = shares * exec_price
                    trade_return = (proceeds - invested_amount) / invested_amount if invested_amount else np.nan
                    if trade_details:
                        trade_details[-1].update({
                            "Exit Date": row["date"],
                            "Exit Price": exec_price,
                            "Trade Return": trade_return,
                        })
                    cash += proceeds
                    shares = 0.0
                    invested_amount = None
            portfolio_val = cash + shares * row["Close"]
            portfolio_values.append(portfolio_val)
            df.at[i, "portfolio_value"] = portfolio_val

        if shares > 0:
            last_row = df.iloc[-1]
            exec_price = get_execution_price(last_row, "sell", execution_method, slippage_rate, bid_ask_spread)
            proceeds = shares * exec_price
            trade_return = (proceeds - invested_amount) / invested_amount if invested_amount else np.nan
            if trade_details:
                trade_details[-1].update({
                    "Exit Date": last_row["date"],
                    "Exit Price": exec_price,
                    "Trade Return": trade_return,
                    "Open Trade": True
                })
            cash += proceeds
            shares = 0.0

        df["daily_return"] = df["portfolio_value"].pct_change().fillna(0)
    
    return df, trade_details

def generate_report(
    df: pd.DataFrame, 
    trade_details: List[Dict[str, Any]], 
    initial_capital: float
) -> Dict[str, Any]:
    """
    Compute performance metrics and create a performance report.
    """
    total_days = (df["date"].iloc[-1] - df["date"].iloc[0]).days
    years = total_days / 365.25 if total_days > 0 else 1.0
    final_value = df["portfolio_value"].iloc[-1]
    total_return = (final_value / initial_capital) - 1
    annual_return = (final_value / initial_capital) ** (1 / years) - 1
    annual_vol = df["daily_return"].std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_vol if annual_vol != 0 else np.nan

    cummax = df["portfolio_value"].cummax()
    drawdown = (df["portfolio_value"] - cummax) / cummax
    max_drawdown = drawdown.min()

    return {
        "Initial Capital": initial_capital,
        "Final Portfolio Value": final_value,
        "Total Return": total_return,
        "Annual Return": annual_return,
        "Annual Volatility": annual_vol,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Trade Details": pd.DataFrame(trade_details)
    }

def backtest(
    df: pd.DataFrame,
    trade_signals: List[Dict[str, Any]],
    initial_capital: float = 100000.0,
    execution_method: str = "mid",
    slippage_rate: float = 0.001,
    bid_ask_spread: float = 0.001,
    symbol: Optional[str] = None,
    threshold_seconds: Optional[float] = None  # Optional threshold for matching signals.
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run the backtest simulation in a functional style.
    Returns the simulation DataFrame and a performance report.
    """
    # Ensure dates are in datetime format.
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    # Merge the trade signals into the data.
    df = merge_signals(df, trade_signals, symbol, threshold_seconds)
    # Simulate trades.
    sim_df, trade_details = trade_simulator(df, initial_capital, execution_method, slippage_rate, bid_ask_spread)
    # Generate the performance report.
    report = generate_report(sim_df, trade_details, initial_capital)
    return sim_df, report

###############################################################################
# Main CLI Entry Point
###############################################################################

@app.command()
def main(
    csv_file: Optional[str] = typer.Option(None, help="CSV file with price data"),
    initial_capital: float = typer.Option(100000.0),
    execution_method: str = typer.Option("mid", help="Execution method: mid, next_open, vwap, bid_ask, close, slippage"),
    slippage_rate: float = typer.Option(0.001, help="Slippage rate for 'slippage' execution method"),
    bid_ask_spread: float = typer.Option(0.001, help="Bid-ask spread for 'bid_ask' execution method"),
    symbol: Optional[str] = typer.Option(None, help="Symbol for this backtest"),
    threshold_seconds: Optional[float] = typer.Option(None, help="Optional threshold (in seconds) for matching signals")
):
    # Load data from CSV if provided; otherwise, generate dummy data.
    if csv_file:
        df = pd.read_csv(csv_file)
    else:
        # Create dummy data.
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "Open": np.random.uniform(100, 200, size=len(dates)),
            "High": np.random.uniform(200, 300, size=len(dates)),
            "Low": np.random.uniform(50, 100, size=len(dates)),
            "Close": np.random.uniform(100, 200, size=len(dates)),
            "Volume": np.random.randint(1000, 5000, size=len(dates))
        })

    # Example trade signals (replace with actual signals as needed)
    trade_signals = [
        {"symbol": symbol if symbol else "AAPL", "date": "20230105000000", "order": "buy", "size": 0.5},
        {"symbol": symbol if symbol else "AAPL", "date": "20230115000000", "order": "sell", "size": 0.5},
    ]

    sim_df, report = backtest(
        df, trade_signals, 
        initial_capital=initial_capital, 
        execution_method=execution_method, 
        slippage_rate=slippage_rate, 
        bid_ask_spread=bid_ask_spread, 
        symbol=symbol,
        threshold_seconds=threshold_seconds
    )
    typer.echo("Backtest Report:")
    typer.echo(report)
    sim_df.to_csv("backtest_results.csv", index=False)

if __name__ == "__main__":
    app()
