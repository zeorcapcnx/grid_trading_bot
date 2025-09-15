import logging
from typing import Any

import numpy as np
import pandas as pd
from tabulate import tabulate

from config.config_manager import ConfigManager
from core.grid_management.grid_level import GridLevel
from core.order_handling.order import Order
from core.order_handling.order_book import OrderBook

ANNUAL_RISK_FREE_RATE = 0.03  # annual risk free rate 3%


class TradingPerformanceAnalyzer:
    def __init__(
        self,
        config_manager: ConfigManager,
        order_book: OrderBook,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_manager: ConfigManager = config_manager
        self.order_book: OrderBook = order_book
        self.base_currency, self.quote_currency, self.trading_fee = self._extract_config()

    def _extract_config(self) -> tuple[str, str, float]:
        """
        Extract trading-related configuration values.

        Returns:
            Tuple[str, str, float]: Base currency, quote currency, and trading fee.
        """
        base_currency = self.config_manager.get_base_currency()
        quote_currency = self.config_manager.get_quote_currency()
        trading_fee = self.config_manager.get_trading_fee()
        return base_currency, quote_currency, trading_fee

    def _calculate_roi(
        self,
        initial_balance: float,
        final_balance: float,
    ) -> float:
        """
        Calculate the return on investment (ROI) percentage.

        Args:
            Initial_balance (float): The initial account balance.
            final_balance (float): The final account balance.

        Returns:
            float: The calculated ROI percentage.
        """
        roi = (final_balance - initial_balance) / initial_balance * 100
        return round(roi, 2)

    def _calculate_trading_gains(self) -> str:
        """
        Calculates the total trading gains from completed buy and sell orders.

        The computation uses only closed orders to determine the net profit or loss
        from executed trades.

        Returns:
            str: The total grid trading gains as a formatted string, or "N/A" if there are no sell orders.
        """
        total_buy_cost = 0.0
        total_sell_revenue = 0.0
        closed_buy_orders = [order for order in self.order_book.get_all_buy_orders() if order.is_filled()]
        closed_sell_orders = [order for order in self.order_book.get_all_sell_orders() if order.is_filled()]

        for buy_order in closed_buy_orders:
            trade_value = buy_order.amount * buy_order.price
            buy_fee = buy_order.fee.get("cost", 0.0) if buy_order.fee else 0.0
            total_buy_cost += trade_value + buy_fee

        for sell_order in closed_sell_orders:
            trade_value = sell_order.amount * sell_order.price
            sell_fee = sell_order.fee.get("cost", 0.0) if sell_order.fee else 0.0
            total_sell_revenue += trade_value - sell_fee

        return "N/A" if total_sell_revenue == 0 else f"{total_sell_revenue - total_buy_cost:.2f}"

    def _calculate_drawdown(self, data: pd.DataFrame) -> float:
        peak = data["account_value"].expanding(min_periods=1).max()
        drawdown = (peak - data["account_value"]) / peak * 100
        max_drawdown = drawdown.max()
        return max_drawdown

    def _calculate_runup(self, data: pd.DataFrame) -> float:
        trough = data["account_value"].expanding(min_periods=1).min()
        runup = (data["account_value"] - trough) / trough * 100
        max_runup = runup.max()
        return max_runup

    def _calculate_time_in_profit_loss(
        self,
        initial_balance: float,
        data: pd.DataFrame,
    ) -> tuple[float, float]:
        time_in_profit = (data["account_value"] > initial_balance).mean() * 100
        time_in_loss = (data["account_value"] <= initial_balance).mean() * 100
        return time_in_profit, time_in_loss

    def _calculate_sharpe_ratio(self, data: pd.DataFrame) -> float:
        """
        Calculate the Sharpe ratio based on the account value.

        Args:
            data (pd.DataFrame): Historical account value data.

        Returns:
            float: The Sharpe ratio.
        """
        if len(data) < 2:
            self.logger.warning("Insufficient data for Sharpe ratio calculation")
            return 0.0

        # Calculate total return and time period
        initial_value = data["account_value"].iloc[0]
        final_value = data["account_value"].iloc[-1]

        if initial_value <= 0 or np.isnan(initial_value) or np.isnan(final_value):
            self.logger.warning(f"Invalid account values for Sharpe: initial={initial_value}, final={final_value}")
            return 0.0

        # Total return
        total_return = (final_value / initial_value) - 1

        if np.isnan(total_return) or np.isinf(total_return):
            self.logger.warning(f"Invalid total return for Sharpe: {total_return}")
            return 0.0

        # Time period in years - use actual dates, not data point count
        start_date = data.index[0]
        end_date = data.index[-1]
        time_period_days = (end_date - start_date).days + 1
        time_period_years = time_period_days / 365.25

        if time_period_years <= 0:
            self.logger.warning(f"Invalid time period for Sharpe: {time_period_years} years")
            return 0.0

        # Annualized return
        annual_return = ((1 + total_return) ** (1 / time_period_years)) - 1

        if np.isnan(annual_return) or np.isinf(annual_return):
            self.logger.warning(f"Invalid annual return for Sharpe: {annual_return}")
            return 0.0

        # Calculate returns for volatility (respecting data frequency)
        returns = data["account_value"].pct_change(fill_method=None).dropna()
        if len(returns) == 0:
            self.logger.warning("No valid returns for Sharpe calculation")
            return 0.0

        # Remove infinite and NaN values from returns
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if len(returns) == 0:
            self.logger.warning("No valid returns after cleaning for Sharpe calculation")
            return 0.0

        # Determine data frequency and adjust volatility scaling
        if len(returns) > 1:
            # Calculate time delta between observations
            time_diff = data.index[1] - data.index[0]
            if hasattr(time_diff, 'seconds'):
                minutes_per_observation = time_diff.seconds / 60
                observations_per_day = 1440 / minutes_per_observation  # 1440 minutes per day
                observations_per_year = observations_per_day * 252     # Trading days
            else:
                # Fallback: assume daily data
                observations_per_year = 252
        else:
            observations_per_year = 252

        period_volatility = returns.std()
        if period_volatility == 0 or np.isnan(period_volatility):
            # No volatility - return simplified ratio
            return round((annual_return - ANNUAL_RISK_FREE_RATE) * 10, 2) if annual_return > ANNUAL_RISK_FREE_RATE else 0.0

        # Properly annualize volatility based on data frequency
        annual_volatility = period_volatility * np.sqrt(observations_per_year)

        if np.isnan(annual_volatility) or annual_volatility == 0:
            self.logger.warning(f"Invalid annual volatility for Sharpe: {annual_volatility}")
            return 0.0

        # Calculate Sharpe ratio
        sharpe_ratio = (annual_return - ANNUAL_RISK_FREE_RATE) / annual_volatility

        if np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio):
            self.logger.warning(f"Invalid Sharpe ratio result: {sharpe_ratio}")
            return 0.0

        self.logger.info(f"Sharpe calculation: {time_period_days} days ({time_period_years:.2f} years)")
        self.logger.info(f"Data frequency: {observations_per_year:.0f} observations/year (√{observations_per_year:.0f} volatility scaling)")
        self.logger.info(f"Total return: {total_return:.4f} ({total_return*100:.2f}%), Annual return: {annual_return:.4f} ({annual_return*100:.2f}%)")
        self.logger.info(f"Period volatility: {period_volatility:.6f}, Annual volatility: {annual_volatility:.4f}")
        self.logger.info(f"Sharpe ratio: ({annual_return:.4f} - {ANNUAL_RISK_FREE_RATE:.4f}) / {annual_volatility:.4f} = {sharpe_ratio:.4f}")
        return round(sharpe_ratio, 2)

    def _calculate_sortino_ratio(self, data: pd.DataFrame) -> float:
        """
        Calculate the Sortino ratio based on the account value.

        Args:
            data (pd.DataFrame): Historical account value data.

        Returns:
            float: The Sortino ratio.
        """
        if len(data) < 2:
            self.logger.warning("Insufficient data for Sortino ratio calculation")
            return 0.0

        # Calculate total return and time period (same as Sharpe)
        initial_value = data["account_value"].iloc[0]
        final_value = data["account_value"].iloc[-1]

        if initial_value <= 0 or np.isnan(initial_value) or np.isnan(final_value):
            self.logger.warning(f"Invalid account values for Sortino: initial={initial_value}, final={final_value}")
            return 0.0

        # Total return
        total_return = (final_value / initial_value) - 1

        if np.isnan(total_return) or np.isinf(total_return):
            self.logger.warning(f"Invalid total return for Sortino: {total_return}")
            return 0.0

        # Time period in years - use actual dates, not data point count
        start_date = data.index[0]
        end_date = data.index[-1]
        time_period_days = (end_date - start_date).days + 1
        time_period_years = time_period_days / 365.25

        if time_period_years <= 0:
            self.logger.warning(f"Invalid time period for Sortino: {time_period_years} years")
            return 0.0

        # Annualized return
        annual_return = ((1 + total_return) ** (1 / time_period_years)) - 1

        if np.isnan(annual_return) or np.isinf(annual_return):
            self.logger.warning(f"Invalid annual return for Sortino: {annual_return}")
            return 0.0

        # Calculate returns for downside deviation (respecting data frequency)
        returns = data["account_value"].pct_change(fill_method=None).dropna()
        if len(returns) == 0:
            self.logger.warning("No valid returns for Sortino calculation")
            return 0.0

        # Remove infinite and NaN values from returns
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if len(returns) == 0:
            self.logger.warning("No valid returns after cleaning for Sortino calculation")
            return 0.0

        # Determine data frequency for proper risk-free rate scaling
        if len(returns) > 1:
            # Calculate time delta between observations
            time_diff = data.index[1] - data.index[0]
            if hasattr(time_diff, 'seconds'):
                minutes_per_observation = time_diff.seconds / 60
                observations_per_day = 1440 / minutes_per_observation  # 1440 minutes per day
                observations_per_year = observations_per_day * 252     # Trading days
            else:
                # Fallback: assume daily data
                observations_per_year = 252
        else:
            observations_per_year = 252

        # Calculate period risk-free rate (not daily)
        period_risk_free = ANNUAL_RISK_FREE_RATE / observations_per_year
        downside_returns = returns[returns < period_risk_free] - period_risk_free

        if len(downside_returns) == 0:
            # No downside risk - return high positive value if annual return > risk free
            result = round((annual_return - ANNUAL_RISK_FREE_RATE) * 10, 2) if annual_return > ANNUAL_RISK_FREE_RATE else 0.0
            self.logger.debug(f"No downside - Sortino ratio: {result}")
            return result

        downside_std = downside_returns.std()
        if downside_std == 0 or np.isnan(downside_std):
            self.logger.warning(f"Invalid downside standard deviation for Sortino: {downside_std}")
            return 0.0

        # Properly annualize downside deviation based on data frequency
        annual_downside_deviation = downside_std * np.sqrt(observations_per_year)

        if np.isnan(annual_downside_deviation) or annual_downside_deviation == 0:
            self.logger.warning(f"Invalid annual downside deviation for Sortino: {annual_downside_deviation}")
            return 0.0

        # Calculate Sortino ratio
        sortino_ratio = (annual_return - ANNUAL_RISK_FREE_RATE) / annual_downside_deviation

        if np.isnan(sortino_ratio) or np.isinf(sortino_ratio):
            self.logger.warning(f"Invalid Sortino ratio result: {sortino_ratio}")
            return 0.0

        self.logger.info(f"Sortino calculation: {time_period_days} days ({time_period_years:.2f} years)")
        self.logger.info(f"Data frequency: {observations_per_year:.0f} observations/year (√{observations_per_year:.0f} volatility scaling)")
        self.logger.info(f"Total return: {total_return:.4f} ({total_return*100:.2f}%), Annual return: {annual_return:.4f} ({annual_return*100:.2f}%)")
        self.logger.info(f"Downside periods: {len(downside_returns)}/{len(returns)}, Annual downside deviation: {annual_downside_deviation:.4f}")
        self.logger.info(f"Sortino ratio: ({annual_return:.4f} - {ANNUAL_RISK_FREE_RATE:.4f}) / {annual_downside_deviation:.4f} = {sortino_ratio:.4f}")
        return round(sortino_ratio, 2)

    def get_formatted_orders(self) -> list[list[str | float]]:
        """
        Retrieve a formatted list of filled buy and sell orders.

        Returns:
            List[List[Union[str, float]]]: Formatted orders with details like side, type,
            status, price, quantity, timestamp, etc.
        """
        orders = []
        buy_orders_with_grid = self.order_book.get_buy_orders_with_grid()
        sell_orders_with_grid = self.order_book.get_sell_orders_with_grid()

        for buy_order, grid_level in buy_orders_with_grid:
            if buy_order.is_filled():
                orders.append(self._format_order(buy_order, grid_level))

        for sell_order, grid_level in sell_orders_with_grid:
            if sell_order.is_filled():
                orders.append(self._format_order(sell_order, grid_level))

        orders.sort(key=lambda x: (x[5] is None, x[5]))  # x[5] is the timestamp, sort None to the end
        return orders

    def _format_order(self, order: Order, grid_level: GridLevel | None) -> list[str | float]:
        grid_level_price = grid_level.price if grid_level else "N/A"
        if grid_level and order.average is not None:
            # Assuming order.price is the execution price and grid level price the expected price
            slippage = ((order.average - grid_level_price) / grid_level_price) * 100
            slippage_str = f"{slippage:.2f}%"
        else:
            slippage = "N/A"
            slippage_str = "N/A"
        return [
            order.side.name,
            order.order_type.name,
            order.status.name,
            order.price,
            order.filled,
            order.format_last_trade_timestamp(),
            grid_level_price,
            slippage_str,
        ]

    def _calculate_trade_counts(self) -> tuple[int, int]:
        """
        Count the number of filled buy and sell orders.

        Returns:
            Tuple[int, int]: Number of buy trades and number of sell trades.
        """
        num_buy_trades = len([order for order in self.order_book.get_all_buy_orders() if order.is_filled()])
        num_sell_trades = len([order for order in self.order_book.get_all_sell_orders() if order.is_filled()])
        return num_buy_trades, num_sell_trades

    def _calculate_buy_and_hold_return(
        self,
        data: pd.DataFrame,
        initial_price: float,
        final_price: float,
    ) -> float:
        """
        Calculate the buy-and-hold return percentage.

        Args:
            data (pd.DataFrame): Historical price data.
            initial_price (float): The initial cryptocurrency price.
            final_price (float): The final cryptocurrency price.

        Returns:
            float: The buy-and-hold return percentage.
        """
        return ((final_price - initial_price) / initial_price) * 100

    def _calculate_buy_and_hold_metrics(
        self,
        data: pd.DataFrame,
        initial_balance: float,
        initial_price: float,
        final_price: float,
    ) -> dict:
        """
        Calculate comprehensive buy-and-hold performance metrics.
        Uses the same time period as the grid strategy data.

        Args:
            data: Historical data with price information (truncated to same period as grid)
            initial_balance: Initial investment amount
            initial_price: Starting crypto price
            final_price: Final crypto price (at the same end point as grid strategy)

        Returns:
            Dict with buy-and-hold performance metrics
        """
        # Create buy-and-hold portfolio value series using the SAME time period as grid strategy
        if 'close' in data.columns:
            # Use actual price data from the same period
            prices = data['close'].copy()
        else:
            # Fallback: estimate from account values (less accurate)
            prices = pd.Series(index=data.index, dtype=float)
            prices.iloc[0] = initial_price
            prices.iloc[-1] = final_price
            # Linear interpolation for missing values
            prices = prices.interpolate()

        # Calculate buy-and-hold portfolio values using the same time period
        crypto_quantity = initial_balance / initial_price
        bh_portfolio_values = prices * crypto_quantity

        # Create buy-and-hold data frame with the SAME index as grid strategy data
        bh_data = pd.DataFrame({'account_value': bh_portfolio_values}, index=data.index)

        # Calculate all metrics for buy-and-hold using the SAME period
        bh_total_return = ((final_price / initial_price) - 1) * 100
        bh_max_drawdown = self._calculate_drawdown(bh_data)
        bh_max_runup = self._calculate_runup(bh_data)
        bh_sharpe = self._calculate_sharpe_ratio(bh_data)
        bh_sortino = self._calculate_sortino_ratio(bh_data)
        bh_time_in_profit, bh_time_in_loss = self._calculate_time_in_profit_loss(initial_balance, bh_data)

        self.logger.info(f"Buy-and-hold calculation period: {data.index[0]} to {data.index[-1]} ({len(data)} data points)")
        self.logger.info(f"Buy-and-hold: {initial_price:.2f} -> {final_price:.2f} = {bh_total_return:.2f}% return")

        return {
            'return': bh_total_return,
            'max_drawdown': bh_max_drawdown,
            'max_runup': bh_max_runup,
            'sharpe_ratio': bh_sharpe,
            'sortino_ratio': bh_sortino,
            'time_in_profit': bh_time_in_profit,
            'time_in_loss': bh_time_in_loss,
            'final_value': bh_portfolio_values.iloc[-1]
        }

    def generate_performance_summary(
        self,
        data: pd.DataFrame,
        initial_price: float,
        final_fiat_balance: float,
        final_crypto_balance: float,
        final_crypto_price: float,
        total_fees: float,
    ) -> tuple[dict[str, Any], list[list[str | float]]]:
        """
        Generate a detailed performance summary for the trading session.

        Args:
            data (pd.DataFrame): Account value and price data.
            final_fiat_balance (float): Final fiat currency balance.
            final_crypto_balance (float): Final cryptocurrency balance.
            final_crypto_price (float): Final cryptocurrency price.
            total_fees (float): Total trading fees incurred.

        Returns:
            Tuple[Dict[str, Any], List[List[Union[str, float]]]]: A dictionary of
            performance metrics and a list of formatted orders.
        """
        pair = f"{self.base_currency}/{self.quote_currency}"
        start_date = data.index[0]
        end_date = data.index[-1]
        initial_balance = data["account_value"].iloc[0]
        duration = end_date - start_date
        final_crypto_value = final_crypto_balance * final_crypto_price
        final_balance = final_fiat_balance + final_crypto_value
        roi = self._calculate_roi(initial_balance, final_balance)
        grid_trading_gains = self._calculate_trading_gains()
        max_drawdown = self._calculate_drawdown(data)
        max_runup = self._calculate_runup(data)
        time_in_profit, time_in_loss = self._calculate_time_in_profit_loss(initial_balance, data)
        sharpe_ratio = self._calculate_sharpe_ratio(data)
        sortino_ratio = self._calculate_sortino_ratio(data)
        buy_and_hold_return = self._calculate_buy_and_hold_return(data, initial_price, final_crypto_price)
        buy_and_hold_metrics = self._calculate_buy_and_hold_metrics(data, initial_balance, initial_price, final_crypto_price)
        num_buy_trades, num_sell_trades = self._calculate_trade_counts()
        
        # Get final cumulative profit if available
        final_cumulative_profit = 0.0
        if "cumulative_profit" in data.columns:
            final_cumulative_profit = data["cumulative_profit"].iloc[-1] if not data["cumulative_profit"].empty else 0.0

        performance_summary = {
            "Pair": pair,
            "Start Date": start_date,
            "End Date": end_date,
            "Duration": duration,
            
            # === GRID TRADING PERFORMANCE ===
            "ROI": f"{roi:.2f}%",
            "Max Drawdown": f"{max_drawdown:.2f}%",
            "Max Runup": f"{max_runup:.2f}%",
            "Time in Profit %": f"{time_in_profit:.2f}%",
            "Time in Loss %": f"{time_in_loss:.2f}%",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}" if not np.isnan(sharpe_ratio) else "0.00",
            "Sortino Ratio": f"{sortino_ratio:.2f}" if not np.isnan(sortino_ratio) else "0.00",
            
            # === BUY & HOLD PERFORMANCE ===
            "Buy and Hold Return %": f"{buy_and_hold_metrics['return']:.2f}%",
            "Buy and Hold Max Drawdown": f"{buy_and_hold_metrics['max_drawdown']:.2f}%",
            "Buy and Hold Max Runup": f"{buy_and_hold_metrics['max_runup']:.2f}%",
            "Buy and Hold Sharpe Ratio": f"{buy_and_hold_metrics['sharpe_ratio']:.2f}" if not np.isnan(buy_and_hold_metrics['sharpe_ratio']) else "0.00",
            "Buy and Hold Sortino Ratio": f"{buy_and_hold_metrics['sortino_ratio']:.2f}" if not np.isnan(buy_and_hold_metrics['sortino_ratio']) else "0.00",
            "Buy and Hold Time in Profit %": f"{buy_and_hold_metrics['time_in_profit']:.2f}%",
            "Buy and Hold Time in Loss %": f"{buy_and_hold_metrics['time_in_loss']:.2f}%",
            
            # === TRADING DETAILS ===
            "Cash from Profit Taking": f"{final_cumulative_profit:.2f} {self.quote_currency}",
            "Cash Gain from Profit Taking %": f"{(final_cumulative_profit / initial_balance) * 100:.2f}%",
            "Total Fees": f"{total_fees:.2f}",
            "Final Balance (Fiat)": f"{final_balance:.2f} {self.quote_currency}",
            "Final Crypto Balance": f"{final_crypto_balance:.4f} {self.base_currency}",
            "Final Crypto Balance (USDT)": f"{final_crypto_value:.2f} {self.quote_currency}",
            "Fiat Balance (USDT)": f"{final_fiat_balance:.2f} {self.quote_currency}",
            "Number of Buy Trades": num_buy_trades,
            "Number of Sell Trades": num_sell_trades,
        }

        formatted_orders = self.get_formatted_orders()

        orders_table = tabulate(
            formatted_orders,
            headers=["Order Side", "Type", "Status", "Price", "Quantity", "Timestamp", "Grid Level", "Slippage"],
            tablefmt="pipe",
        )
        self.logger.info("\nFormatted Orders:\n" + orders_table)

        summary_table = tabulate(performance_summary.items(), headers=["Metric", "Value"], tablefmt="grid")
        self.logger.info("\nPerformance Summary:\n" + summary_table)

        return performance_summary, formatted_orders
