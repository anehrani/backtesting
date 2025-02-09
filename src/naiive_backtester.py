#!/usr/bin/env python
"""
Backtesting Script with Multiple Execution-Price Methods

Input CSV (or dummy data) must contain:
  • date      : Date/time (up to seconds)
  • Open      : Opening price
  • High      : Highest price of the period
  • Low       : Lowest price of the period
  • Close     : Closing price
  • Volume    : Volume


Execution Price Methods (select via --execution-method):
  • mid       : (High+Low)/2
  • next_open : Execute at next row's Open price (orders are delayed one period)
  • vwap      : Proxy VWAP = (High+Low+Close)/3
  • bid_ask   : Use mid adjusted by half the bid/ask spread
  • close     : Use the current candle's Close price
  • slippage  : Use mid adjusted by a user-defined slippage rate

Usage examples:
  • With CSV data using mid-price execution:
      python backtester.py --csv-file "data.csv" --initial-capital 100000 --execution-method mid
  • Using next_open method with custom slippage/spread parameters:
      python backtester.py --csv-file "data.csv" --execution-method next_open --slippage-rate 0.002 --bid-ask-spread 0.002
  • Without a CSV file to generate dummy data:
      python backtester.py
"""

import pandas as pd
import numpy as np
import typer
from typing import Optional
from pathlib import Path
from threading import Lock

app = typer.Typer(help="Backtesting CLI: simulate trades and generate performance metrics.")


class NaiveBacktester:
    def __init__(self, 
                 historic_data, 
                 trade_signals, 
                 commission_value=0.0075, 
                 initial_capital=100000, 
                 execution_method="close"):
        """
        Initialize backtester with historical data and parameters
        
        Parameters:
        historic_data (pd.DataFrame): OHLCV data with datetime index
        trade_signals (list): List of trade dictionaries
        commission_value (float): Transaction cost as percentage
        initial_capital (float): Starting capital
        process_orders_on_close (bool): Execute at close prices
        """
                # Add this to __init__
        if not isinstance(historic_data.index, pd.DatetimeIndex):
            raise ValueError("Historic data must have datetime index")

        # Add this to _prepare_data
        if 'close' not in historic_data.columns:
            raise ValueError("Historic data must contain 'close' column")
        self.historic_data = historic_data.copy()
        self.trade_signals = sorted(trade_signals, key=lambda x: x['time_stamp'])
        self.commission = commission_value
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position = 0
        self.entry_price = None
        self.trade_history = []
        self._lock = Lock()
        self.portfolio_values = []
        self.portfolio_values = []
        self.execution_method = execution_method
        
        # Preprocess data
        self._prepare_data()
        self._validate_signals()

    def _prepare_data(self):
        """Ensure proper datetime format and sorting"""
        self.historic_data.index = pd.to_datetime(self.historic_data.index)
        self.historic_data.sort_index(inplace=True)
        self.historic_data['returns'] = self.historic_data['close'].pct_change()

    def _validate_signals(self):
        """Check signal timestamps exist in historical data"""
        valid_signals = []
        for signal in self.trade_signals:
            if signal['time_stamp'] in self.historic_data.index:
                valid_signals.append(signal)
            else:
                print(f"Warning: Skipping signal at {signal['time_stamp']} - no price data")
        self.trade_signals = valid_signals

    def _get_execution_price(self, i: int, side: str) -> float:
        """
        Compute the execution price based on the selected method (for synchronous methods).
        For non next_open methods, execution is assumed to be in the same row.

        Args:
            i (int): Row index.
            side (str): "buy" or "sell".

        Returns:
            float: Execution price.
        """
        row = self.historic_data.iloc[i]
        if self.execution_method == "mid":
            price = 0.5 * (row["High"] + row["Low"])
        elif self.execution_method == "vwap":
            price = (row["High"] + row["Low"] + row["Close"]) / 3
        elif self.execution_method == "bid_ask":
            mid = 0.5 * (row["High"] + row["Low"])
            if side == "buy":
                price = mid * (1 + self.bid_ask_spread / 2)
            elif side == "sell":
                price = mid * (1 - self.bid_ask_spread / 2)
            else:
                price = mid
        elif self.execution_method == "close":
            price = row["Close"]
        elif self.execution_method == "slippage":
            mid = 0.5 * (row["High"] + row["Low"])
            if side == "buy":
                price = mid * (1 + self.slippage_rate)
            elif side == "sell":
                price = mid * (1 - self.slippage_rate)
            else:
                price = mid
        else:
            # Fallback to mid.
            price = 0.5 * (row["High"] + row["Low"])
        return price


    def _process_trade(self, signal, price):
        """Execute a single trade with risk management"""
        # Open new position
        position_size = signal['position_size']
        direction = signal['direction']

        if direction == 'long':
            max_investment = self.current_capital * position_size
            traded_shares = max_investment / (price + 1e-10)
        else:  # short
            traded_shares = - self.position * position_size


        commission = abs(traded_shares) * price * self.commission
        self.position += traded_shares
        self.current_capital -= ( traded_shares * price + commission)  # Deduct both investment and commission
        self.entry_price = price

        # Record entry
        self.trade_history.append({
            'type': 'entry',
            'timestamp': signal['time_stamp'],
            'price': price,
            'traded_shares': traded_shares,
            'position_size': position_size,
            'capital': self.current_capital,
            'commission': commission,
            'direction': signal['direction'],
            "pnl": self.current_capital + self.position * price - self.initial_capital
        })

    # In _clean_portfolio_values
    def _clean_portfolio_values(self):
        """Ensure numeric values and handle NaNs"""
        cleaned = []
        for val in self.portfolio_values:
            try:
                cleaned.append(float(val))
            except (TypeError, ValueError):
                cleaned.append(np.nan)  
        # Forward fill and convert to list
        self.portfolio_values = pd.Series(cleaned).ffill().fillna(0).tolist()

    def run_backtest(self):
        """Main backtesting loop"""
        self.portfolio_values = []
        signal_idx = 0
        num_signals = len(self.trade_signals)

        for date, prices in self.historic_data.iterrows():
            # Update portfolio value
            with self._lock:
                current_value = self.current_capital + self.position * prices['close']
                self.portfolio_values.append(current_value)
            #current_value = self.current_capital + self.position * prices['close']
            #self.portfolio_values.append(current_value)

            # Process signals
            while signal_idx < num_signals and date >= self.trade_signals[signal_idx]['time_stamp']:
                if date == self.trade_signals[signal_idx]['time_stamp']:
                    signal = self.trade_signals[signal_idx]
                    self._process_trade(signal, prices['close'])
                signal_idx += 1

        # Final portfolio value
        self.final_value = self.portfolio_values[-1]

        # Ensure portfolio_values is a list of numeric values
        self.portfolio_values = [float(value) for value in self.portfolio_values]

    def calculate_metrics(self):
        """Calculate performance and risk metrics"""
        self._clean_portfolio_values()

        if not self.portfolio_values or len(self.portfolio_values) < 2:
            return {"error": "Insufficient data for metrics calculation"}
        
        returns = pd.Series(self.portfolio_values).pct_change().dropna()
        trade_df = pd.DataFrame(self.trade_history)
        
        # Basic metrics
        total_return = (self.final_value / self.initial_capital) - 1
        annualized_return = (1 + total_return) ** (252 / len(self.historic_data)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe = annualized_return / volatility if volatility != 0 else 0
        
        # Drawdown calculations
        peak = pd.Series(self.portfolio_values).cummax()
        drawdown = (pd.Series(self.portfolio_values) - peak) / peak
        max_drawdown = drawdown.min()
        
        # Trade metrics
        if not trade_df.empty:
            winning_trades = trade_df[trade_df['pnl'] > 0]
            win_rate = len(winning_trades) / len(trade_df) if len(trade_df) > 0 else 0
            avg_trade_return = trade_df['pnl'].mean() / self.initial_capital
        else:
            win_rate = avg_trade_return = 0

        # Calculate Sortino ratio directly
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annualized_return - 0.02) / downside_vol if downside_vol != 0 else 0

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_trade_return': avg_trade_return,
            'num_trades': len(trade_df)//2,
            'final_value': self.final_value,
            'sortino_ratio': sortino_ratio,  # Use directly calculated value
            'calmar_ratio': annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0,
            'profit_factor': self._calculate_profit_factor(trade_df)
        }

    def _calculate_profit_factor(self, trade_df):
        """More realistic profit factor calculation"""
        if not trade_df.empty:
            winning = trade_df[trade_df['pnl'] > 0]['pnl'].sum()
            losing = abs(trade_df[trade_df['pnl'] < 0]['pnl'].sum())
            return winning / losing if losing > 0 else float('inf')
        return 0

    def plot_performance(self, save_path: Path = None, plot_title: str = ''):
        """Visualize backtest results with positions and portfolio value"""
        import matplotlib.pyplot as plt
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Plot Price with Positions
        ax1.plot(self.historic_data.index, self.historic_data['close'], 
                label='Price', color='royalblue', linewidth=1.5)
        
        # Plot position markers
        long_entries = [trade for trade in self.trade_history 
                        if trade['type'] == 'entry' and trade['direction'] == 'long']
        short_entries = [trade for trade in self.trade_history 
                        if trade['type'] == 'entry' and trade['direction'] == 'short']
        
        ax1.scatter(
            [entry['timestamp'] for entry in long_entries],
            [entry['price'] for entry in long_entries],
            marker='^', color='limegreen', s=100, edgecolor='black',
            label='Long Entry', zorder=3
        )
        
        ax1.scatter(
            [entry['timestamp'] for entry in short_entries],
            [entry['price'] for entry in short_entries],
            marker='v', color='crimson', s=100, edgecolor='black',
            label='Short Entry', zorder=3
        )
        
        ax1.set_title(f'Price and Positions - {plot_title}')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.4)
        
        # Plot Portfolio Value
        ax2.plot(self.historic_data.index, self.portfolio_values,
                label='Strategy', color='darkorange', linewidth=2)
        
        # Plot Buy & Hold comparison
        buy_hold = (self.historic_data['close'] / self.historic_data['close'].iloc[0] 
                    * self.initial_capital)
        ax2.plot(self.historic_data.index, buy_hold, 
                label='Buy & Hold', color='purple', alpha=0.7, linestyle='--')
        
        ax2.set_title('Portfolio Value Development')
        ax2.set_ylabel('Portfolio Value')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.4)
        
        # Formatting
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(f'{save_path}/backtest_results_{plot_title}.png', 
                        bbox_inches='tight', dpi=300)
        else:
            plt.show()
        
        plt.close()

    def run_test(self, save_path:Path=None, plot_title:str='_', save_results:bool=True):
        """Run backtest and calculate metrics"""
        self.run_backtest()
        metrics = self.calculate_metrics()
        if save_results:
            with open(f'{save_path}/metrics_{plot_title}.txt', 'w') as f:
                for k, v in metrics.items():
                    f.write(f"{k:>20}: {v:.4f}\n")
            self.plot_performance(save_path, plot_title)

        return metrics
        




