import logging

import numpy as np
import pandas as pd

from config.config_manager import ConfigManager
from config.trading_mode import TradingMode
from core.bot_management.event_bus import EventBus, Events
from core.grid_management.grid_level import GridLevel, GridCycleState
from core.grid_management.grid_manager import GridManager
from core.order_handling.balance_tracker import BalanceTracker
from core.order_handling.order import OrderStatus
from core.order_handling.order_manager import OrderManager
from core.services.exchange_interface import ExchangeInterface
from strategies.plotter import Plotter
from strategies.trading_performance_analyzer import TradingPerformanceAnalyzer

from .trading_strategy_interface import TradingStrategyInterface


class GridTradingStrategy(TradingStrategyInterface):
    TICKER_REFRESH_INTERVAL = 3  # in seconds

    def __init__(
        self,
        config_manager: ConfigManager,
        event_bus: EventBus,
        exchange_service: ExchangeInterface,
        grid_manager: GridManager,
        order_manager: OrderManager,
        balance_tracker: BalanceTracker,
        trading_performance_analyzer: TradingPerformanceAnalyzer,
        trading_mode: TradingMode,
        trading_pair: str,
        plotter: Plotter | None = None,
    ):
        super().__init__(config_manager, balance_tracker)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.event_bus = event_bus
        self.exchange_service = exchange_service
        self.grid_manager = grid_manager
        self.order_manager = order_manager
        self.trading_performance_analyzer = trading_performance_analyzer
        self.trading_mode = trading_mode
        self.trading_pair = trading_pair
        self.plotter = plotter
        self.data = self._initialize_historical_data()
        self.live_trading_metrics = []
        self._running = True
        self._cumulative_profit = 0.0  # Track cumulative profit from trading pairs
        self._grid_buy_costs = {}  # Track buy costs per grid level
        self._initial_purchase_cost = None  # Track initial purchase cost basis
        self._initial_purchase_quantity = None  # Track initial purchase quantity
        # Subscribe to order fill events to track profit
        self.event_bus.subscribe(Events.ORDER_FILLED, self._on_order_filled)

    def _initialize_historical_data(self) -> pd.DataFrame | None:
        """
        Initializes historical market data (OHLCV).
        In LIVE or PAPER_TRADING mode returns None.
        """
        if self.trading_mode != TradingMode.BACKTEST:
            return None

        try:
            timeframe, start_date, end_date = self._extract_config()
            return self.exchange_service.fetch_ohlcv(self.trading_pair, timeframe, start_date, end_date)
        except Exception as e:
            self.logger.error(f"Failed to initialize data for backtest trading mode: {e}")
            return None

    def _extract_config(self) -> tuple[str, str, str]:
        """
        Extracts configuration values for timeframe, start date, and end date.

        Returns:
            tuple: A tuple containing the timeframe, start date, and end date as strings.
        """
        timeframe = self.config_manager.get_timeframe()
        start_date = self.config_manager.get_start_date()
        end_date = self.config_manager.get_end_date()
        return timeframe, start_date, end_date

    def initialize_strategy(self):
        """
        Initializes the trading strategy by setting up the grid and levels.
        This method prepares the strategy to be ready for trading.
        """
        # Get first price for range calculation if needed
        first_price = None
        if self.data is not None and len(self.data) > 0:
            first_price = self.data["close"].iloc[0]
            
        self.grid_manager.initialize_grids_and_levels(first_price)

    async def stop(self):
        """
        Stops the trading execution.

        This method halts all trading activities, closes active exchange
        connections, and updates the internal state to indicate the bot
        is no longer running.
        """
        self._running = False
        await self.exchange_service.close_connection()
        self.logger.info("Trading execution stopped.")

    async def restart(self):
        """
        Restarts the trading session. If the strategy is not running, starts it.
        """
        if not self._running:
            self.logger.info("Restarting trading session.")
            await self.run()

    async def run(self):
        """
        Starts the trading session based on the configured mode.

        For backtesting, this simulates the strategy using historical data.
        For live or paper trading, this interacts with the exchange to manage
        real-time trading.

        Raises:
            Exception: If any error occurs during the trading session.
        """
        self._running = True
        trigger_price = self.grid_manager.get_trigger_price()

        if self.trading_mode == TradingMode.BACKTEST:
            await self._run_backtest(trigger_price)
            self.logger.info("Ending backtest simulation")
            self._running = False
        else:
            await self._run_live_or_paper_trading(trigger_price)

    async def _run_live_or_paper_trading(self, trigger_price: float):
        """
        Executes live or paper trading sessions based on real-time ticker updates.

        The method listens for ticker updates, initializes grid orders when
        the trigger price is reached, and manages take-profit and stop-loss events.

        Args:
            trigger_price (float): The price at which grid orders are triggered.
        """
        self.logger.info(f"Starting {'live' if self.trading_mode == TradingMode.LIVE else 'paper'} trading")
        last_price: float | None = None
        grid_orders_initialized = False

        async def on_ticker_update(current_price):
            nonlocal last_price, grid_orders_initialized
            try:
                if not self._running:
                    self.logger.info("Trading stopped; halting price updates.")
                    return

                account_value = self.balance_tracker.get_total_balance_value(current_price)
                self.live_trading_metrics.append((pd.Timestamp.now(), account_value, current_price))

                grid_orders_initialized = await self._initialize_grid_orders_once(
                    current_price,
                    trigger_price,
                    grid_orders_initialized,
                    last_price,
                )

                if not grid_orders_initialized:
                    last_price = current_price
                    return

                if await self._handle_take_profit_stop_loss(current_price):
                    return

                last_price = current_price

            except Exception as e:
                self.logger.error(f"Error during ticker update: {e}", exc_info=True)

        try:
            await self.exchange_service.listen_to_ticker_updates(
                self.trading_pair,
                on_ticker_update,
                self.TICKER_REFRESH_INTERVAL,
            )

        except Exception as e:
            self.logger.error(f"Error in live/paper trading loop: {e}", exc_info=True)

        finally:
            self.logger.info("Exiting live/paper trading loop.")

    async def _run_backtest(self, trigger_price: float) -> None:
        """
        Executes the backtesting simulation based on historical OHLCV data.

        This method simulates trading using preloaded data, managing grid levels,
        executing orders, and updating account values over the timeframe.

        Args:
            trigger_price (float): The price at which grid orders are triggered.
        """
        if self.data is None:
            self.logger.error("No data available for backtesting.")
            return

        self.logger.info("Starting backtest simulation")
        self.data["account_value"] = np.nan
        self.data["cumulative_profit"] = 0.0  # Initialize cumulative profit column
        self.close_prices = self.data["close"].values
        high_prices = self.data["high"].values
        low_prices = self.data["low"].values
        timestamps = self.data.index
        self.data.loc[timestamps[0], "account_value"] = self.balance_tracker.get_total_balance_value(
            price=self.close_prices[0],
        )
        self.data.loc[timestamps[0], "cumulative_profit"] = self._cumulative_profit
        grid_orders_initialized = False
        last_price = None

        for i, (current_price, high_price, low_price, timestamp) in enumerate(
            zip(self.close_prices, high_prices, low_prices, timestamps, strict=False),
        ):
            grid_orders_initialized = await self._initialize_grid_orders_once(
                current_price,
                trigger_price,
                grid_orders_initialized,
                last_price,
            )

            if not grid_orders_initialized:
                self.data.loc[timestamps[i], "account_value"] = self.balance_tracker.get_total_balance_value(
                    price=current_price,
                )
                self.data.loc[timestamps[i], "cumulative_profit"] = self._cumulative_profit
                last_price = current_price
                continue

            await self.order_manager.simulate_order_fills(high_price, low_price, timestamp)

            if await self._handle_take_profit_stop_loss(current_price):
                # Update final account value immediately after TP execution
                self.data.loc[timestamp, "account_value"] = self.balance_tracker.get_total_balance_value(current_price)
                self.data.loc[timestamp, "cumulative_profit"] = self._cumulative_profit
                self.logger.info(f"Take-profit executed. Final account value: ${self.balance_tracker.get_total_balance_value(current_price):.2f}")
                break

            self.data.loc[timestamp, "account_value"] = self.balance_tracker.get_total_balance_value(current_price)
            self.data.loc[timestamp, "cumulative_profit"] = self._cumulative_profit
            last_price = current_price

    async def _initialize_grid_orders_once(
        self,
        current_price: float,
        trigger_price: float,
        grid_orders_initialized: bool,
        last_price: float | None = None,
    ) -> bool:
        """
        Extracts configuration values for timeframe, start date, and end date.

        Returns:
            tuple: A tuple containing the timeframe, start date, and end date as strings.
        """
        if grid_orders_initialized:
            return True

        if last_price is None:
            self.logger.debug("No previous price recorded yet. Waiting for the next price update.")
            return False

        if last_price <= trigger_price <= current_price or last_price == trigger_price:
            self.logger.info(
                f"Current price {current_price} reached trigger price {trigger_price}. Will perform initial purhcase",
            )
            await self.order_manager.perform_initial_purchase(current_price)
            self.logger.info("Initial purchase done, will initialize grid orders")
            await self.order_manager.initialize_grid_orders(current_price)
            return True

        self.logger.info(
            f"Current price {current_price} did not cross trigger price {trigger_price}. Last price: {last_price}.",
        )
        return False

    def generate_performance_report(self) -> tuple[dict, list]:
        """
        Generates a performance report for the trading session.

        It evaluates the strategy's performance by analyzing
        the account value, fees, and final price over the given timeframe.

        Returns:
            tuple: A dictionary summarizing performance metrics and a list of formatted order details.
        """
        if self.trading_mode == TradingMode.BACKTEST:
            # Filter data to only include rows with valid account_value (non-NaN)
            # This ensures we only analyze data up to where the backtest actually ran
            valid_data = self.data.dropna(subset=['account_value'])

            if len(valid_data) == 0:
                self.logger.error("No valid data available for performance analysis")
                return {}, []

            initial_price = valid_data["close"].iloc[0]
            final_price = valid_data["close"].iloc[-1]

            self.logger.info(f"Performance analysis period: {valid_data.index[0]} to {valid_data.index[-1]} ({len(valid_data)} data points)")

            return self.trading_performance_analyzer.generate_performance_summary(
                valid_data,
                initial_price,
                self.balance_tracker.get_adjusted_fiat_balance(),
                self.balance_tracker.get_adjusted_crypto_balance(),
                final_price,
                self.balance_tracker.total_fees,
            )
        else:
            if not self.live_trading_metrics:
                self.logger.warning("No account value data available for live/paper trading mode.")
                return {}, []

            live_data = pd.DataFrame(self.live_trading_metrics, columns=["timestamp", "account_value", "price"])
            live_data.set_index("timestamp", inplace=True)
            initial_price = live_data.iloc[0]["price"]
            final_price = live_data.iloc[-1]["price"]

            return self.trading_performance_analyzer.generate_performance_summary(
                live_data,
                initial_price,
                self.balance_tracker.get_adjusted_fiat_balance(),
                self.balance_tracker.get_adjusted_crypto_balance(),
                final_price,
                self.balance_tracker.total_fees,
            )

    def plot_results(self) -> None:
        """
        Plots the backtest results using the provided plotter.

        This method generates and displays visualizations of the trading
        strategy's performance during backtesting. If the bot is running
        in live or paper trading mode, plotting is not available.
        """
        if self.trading_mode == TradingMode.BACKTEST:
            self.plotter.plot_results(self.data)
        else:
            self.logger.info("Plotting is not available for live/paper trading mode.")

    def plot_equity_curve_comparison(self) -> None:
        """
        Plots a comparison between grid trading and buy-and-hold equity curves.

        This method generates and displays a comparison chart showing the
        performance of the grid trading strategy versus a simple buy-and-hold
        approach over the same time period.
        """
        if self.trading_mode == TradingMode.BACKTEST:
            if self.data is not None and len(self.data) > 0:
                initial_price = self.close_prices[0]
                final_price = self.close_prices[-1]
                self.plotter.plot_equity_curve_comparison(self.data, initial_price, final_price)
            else:
                self.logger.error("No data available for equity curve comparison.")
        else:
            self.logger.info("Equity curve comparison is only available for backtest mode.")

    async def _handle_take_profit_stop_loss(self, current_price: float) -> bool:
        """
        Handles take-profit or stop-loss events based on the current price.
        In dynamic mode, restarts the grid instead of stopping.
        In traditional mode, publishes a STOP_BOT event if either condition is triggered.
        """
        if self.config_manager.is_dynamic_mode_enabled():
            return await self._handle_dynamic_boundary_hit(current_price)
        else:
            tp_or_sl_triggered = await self._evaluate_tp_or_sl(current_price)
            if tp_or_sl_triggered:
                self.logger.info("Take-profit or stop-loss triggered, ending trading session.")
                await self.event_bus.publish(Events.STOP_BOT, "TP or SL hit.")
                return True
            return False

    async def _evaluate_tp_or_sl(self, current_price: float) -> bool:
        """
        Evaluates whether take-profit or stop-loss conditions are met.
        Returns True if any condition is triggered.
        """
        return await self._handle_take_profit(current_price) or await self._handle_stop_loss(current_price)

    async def _handle_dynamic_boundary_hit(self, current_price: float) -> bool:
        """
        Handles dynamic grid restart when price hits grid boundaries.
        Uses ALL available funds (fiat + crypto) to restart grid with current price as trigger.
        """
        min_grid_price = min(self.grid_manager.price_grids)
        max_grid_price = max(self.grid_manager.price_grids)

        # Check if current price is beyond grid boundaries
        if current_price >= max_grid_price:
            available_fiat = self.balance_tracker.balance
            available_crypto = self.balance_tracker.crypto_balance
            total_balance_value = self.balance_tracker.get_total_balance_value(current_price)

            if total_balance_value > 0:
                self.logger.info(f"🔴 TOP BOUNDARY HIT: ${current_price:.2f} >= ${max_grid_price:.2f}")
                crypto_value = available_crypto * current_price
                self.logger.info(f"💰 Portfolio: ${available_fiat:.2f} fiat + {available_crypto:.4f} crypto (${crypto_value:.2f}) = ${total_balance_value:.2f}")
                await self._restart_grid_with_all_funds(current_price, "top")
                return False  # Don't stop trading, continue with new grid
            else:
                self.logger.warning(f"🔴 TOP BOUNDARY: ${current_price:.2f} - No funds available for restart")

        elif current_price <= min_grid_price:
            available_fiat = self.balance_tracker.balance
            available_crypto = self.balance_tracker.crypto_balance

            if available_fiat > 0:
                self.logger.info(f"🔵 BOTTOM BOUNDARY HIT: ${current_price:.2f} <= ${min_grid_price:.2f}")
                crypto_value = available_crypto * current_price
                self.logger.info(f"💵 Extending grid: ${available_fiat:.2f} fiat + {available_crypto:.4f} crypto (${crypto_value:.2f} kept)")
                await self._extend_grid_downward(current_price, available_fiat)
                return False  # Don't stop trading, continue with extended grid
            else:
                self.logger.warning(f"🔵 BOTTOM BOUNDARY: ${current_price:.2f} - No fiat available for extension")

        return False  # Never stop trading in dynamic mode

    async def _extend_grid_downward(self, current_price: float, available_fiat: float) -> None:
        """
        Extends the grid downward when hitting bottom boundary.
        Uses available fiat to create new lower grid levels with same spacing.
        """
        try:
            # Get current grid spacing
            current_spacing = self._calculate_current_grid_spacing()
            current_bottom = min(self.grid_manager.price_grids)

            self.logger.info(f"📊 Extending grid: ${current_bottom:.2f} → lower by ${current_spacing:.2f} spacing")

            # Calculate how many new grid levels we can afford with available fiat
            # Using equal-dollar sizing
            order_sizing = self.config_manager.get_order_sizing_type()
            num_grids = self.config_manager.get_num_grids()
            dollar_per_grid = available_fiat / max(1, num_grids // 4)  # Use 1/4 of original grid count for extension

            # Calculate new lower grid levels
            new_grid_levels = []
            new_price = current_bottom - current_spacing
            grid_count = 0
            remaining_fiat = available_fiat

            while remaining_fiat >= dollar_per_grid and new_price > 0 and grid_count < num_grids // 2:
                new_grid_levels.append(new_price)
                remaining_fiat -= dollar_per_grid
                new_price -= current_spacing
                grid_count += 1

            if new_grid_levels:
                levels_preview = f"${new_grid_levels[0]:.2f}" + (f"...${new_grid_levels[-1]:.2f}" if len(new_grid_levels) > 1 else "")
                self.logger.info(f"✅ Adding {len(new_grid_levels)} levels: {levels_preview} (${dollar_per_grid:.0f} each)")
                await self._add_grid_levels_below(new_grid_levels, dollar_per_grid)
            else:
                self.logger.warning("❌ Insufficient fiat for grid extension")

        except Exception as e:
            self.logger.error(f"Failed to extend grid downward: {e}")

    def _calculate_current_grid_spacing(self) -> float:
        """Calculate the spacing between current grid levels"""
        if len(self.grid_manager.price_grids) < 2:
            return 100.0  # Default spacing if can't calculate

        sorted_grids = sorted(self.grid_manager.price_grids)
        spacing = sorted_grids[1] - sorted_grids[0]
        return abs(spacing)

    async def _add_grid_levels_below(self, new_price_levels: list[float], dollar_per_grid: float) -> None:
        """Add new grid levels below current bottom and place buy orders"""
        try:
            for price in new_price_levels:
                # Add to grid manager's price grids and grid levels
                self.grid_manager.price_grids.append(price)
                self.grid_manager.price_grids.sort()  # Keep sorted

                # Create proper grid level for profit-taking cycle
                grid_level = GridLevel(price, GridCycleState.READY_TO_BUY)
                self.grid_manager.grid_levels[price] = grid_level

                # Add to sorted buy grids
                self.grid_manager.sorted_buy_grids.append(price)
                self.grid_manager.sorted_buy_grids.sort()

                # Set up profit-taking relationship with higher levels
                self._setup_profit_taking_for_new_level(grid_level)

                # Calculate quantity for equal-dollar sizing
                quantity = dollar_per_grid / price

                # Grid level added - logging done above in the summary

            # After adding all levels, place orders for the new grid levels
            current_price = new_price_levels[0] + self._calculate_current_grid_spacing()  # Estimate current price
            self.logger.debug(f"Integrated {len(new_price_levels)} levels into profit system")

            # Place orders for the newly added grid levels
            await self._place_orders_for_new_levels(new_price_levels, current_price)

        except Exception as e:
            self.logger.error(f"Failed to add grid levels below: {e}")

    def _setup_profit_taking_for_new_level(self, new_grid_level: GridLevel) -> None:
        """Setup profit-taking relationships for newly added grid level"""
        try:
            # For hedged grid strategy, find appropriate sell level above this buy level
            if self.config_manager.get_strategy_type().name == "HEDGED_GRID":
                current_spacing = self._calculate_current_grid_spacing()
                target_sell_price = new_grid_level.price + current_spacing

                # Find existing sell level at or near target price
                closest_sell_level = None
                min_distance = float('inf')

                for price, level in self.grid_manager.grid_levels.items():
                    if price > new_grid_level.price:  # Only consider higher prices
                        distance = abs(price - target_sell_price)
                        if distance < min_distance:
                            min_distance = distance
                            closest_sell_level = level

                if closest_sell_level:
                    # Set up the profit-taking relationship
                    new_grid_level.paired_sell_level = closest_sell_level
                    self.logger.debug(f"Paired new buy level ${new_grid_level.price:.2f} with sell level ${closest_sell_level.price:.2f}")

        except Exception as e:
            self.logger.warning(f"Failed to setup profit-taking for new level: {e}")

    async def _place_orders_for_new_levels(self, new_price_levels: list[float], current_price: float) -> None:
        """Place buy orders for newly added grid levels"""
        try:
            for price in new_price_levels:
                if price < current_price:  # Only place buy orders below current price
                    grid_level = self.grid_manager.grid_levels.get(price)
                    if grid_level and grid_level.state == GridCycleState.READY_TO_BUY:
                        # Use the order manager's existing logic to place buy orders
                        # This is a simplified approach - in full implementation would need proper order placement
                        self.logger.info(f"Would place buy order at ${price:.2f}")
        except Exception as e:
            self.logger.error(f"Failed to place orders for new levels: {e}")

    async def _cancel_all_pending_orders(self) -> None:
        """Cancel all pending orders for grid restart and release reserved funds"""
        try:
            open_orders = self.order_manager.order_book.get_open_orders()
            if open_orders:
                self.logger.info(f"🚫 Cancelling {len(open_orders)} pending orders")

                # Log balances before cancellation
                self.logger.debug(f"Before cancellation: ${self.balance_tracker.balance:.2f} available, ${self.balance_tracker.reserved_fiat:.2f} reserved fiat")

                # Mark all orders as cancelled
                for order in open_orders:
                    order.status = OrderStatus.CANCELED

                # Release all reserved funds back to available balance
                self.balance_tracker.release_all_reserved_funds()

                # Log balances after release
                self.logger.debug(f"After release: ${self.balance_tracker.balance:.2f} available, ${self.balance_tracker.reserved_fiat:.2f} reserved fiat")

                # Clear the order book
                self.order_manager.order_book.clear_all_orders()
            else:
                self.logger.debug("No orders to cancel")

        except Exception as e:
            self.logger.error(f"Failed to cancel pending orders: {e}")

    async def _restart_grid_with_all_funds(self, current_price: float, boundary_type: str) -> None:
        """
        Restarts the grid using ALL available funds (fiat + crypto).
        Uses current price as the trigger/center point for the new grid.

        Args:
            current_price: The current market price to use as trigger point
            boundary_type: "top" or "bottom" for logging purposes
        """
        try:
            # Cancel all pending orders to free up locked funds
            await self._cancel_all_pending_orders()

            # Get total available funds after cancelling orders
            available_fiat = self.balance_tracker.balance
            available_crypto = self.balance_tracker.crypto_balance
            total_value = self.balance_tracker.get_total_balance_value(current_price)

            self.logger.info(f"🔄 GRID RESTART ({boundary_type}): ${total_value:.2f} total → new trigger ${current_price:.2f}")
            self.logger.info(f"📈 Performance so far: ${self._cumulative_profit:.2f} profit, ${self.balance_tracker.total_fees:.2f} fees")

            # Clear grid state for fresh restart
            self._reset_grid_state()

            # Apply boundary-specific rebalancing logic
            if boundary_type == "top":
                await self._rebalance_for_top_boundary(current_price, available_fiat, available_crypto)
            elif boundary_type == "bottom":
                await self._rebalance_for_bottom_boundary(current_price, available_fiat, available_crypto)

            # Initialize new grid with current price as trigger point
            self.grid_manager.initialize_grids_and_levels(current_price)

            # Place new grid orders using all available funds
            await self.order_manager.initialize_grid_orders(current_price)

            self.logger.info(f"✅ Grid restart complete - now trading around ${current_price:.2f}")

        except Exception as e:
            self.logger.error(f"Failed to restart grid from {boundary_type}: {e}")

    async def _rebalance_for_top_boundary(self, current_price: float, available_fiat: float, available_crypto: float) -> None:
        """
        Rebalancing logic for top boundary hit:
        - After cancelling all orders, we have too much fiat (from refunded buy orders)
        - Need to buy crypto to balance the portfolio for new grid
        - Target: ~65% fiat, ~35% crypto for equal_dollar grid
        """
        try:
            crypto_value = available_crypto * current_price
            total_portfolio_value = self.balance_tracker.get_total_balance_value(current_price)

            # For equal_dollar grid, we need balanced 50/50 split
            target_fiat_ratio = 0.5  # 50% fiat for buy orders
            target_crypto_ratio = 0.5  # 50% crypto for sell orders

            target_fiat = total_portfolio_value * target_fiat_ratio
            target_crypto_value = total_portfolio_value * target_crypto_ratio

            self.logger.info(f"💼 Portfolio (after refunds): ${available_fiat:.0f} fiat ({available_fiat/total_portfolio_value*100:.0f}%) + {available_crypto:.4f} crypto (${crypto_value:.0f}, {crypto_value/total_portfolio_value*100:.0f}%) = ${total_portfolio_value:.0f} total")
            self.logger.debug(f"🎯 Target: ${target_fiat:.0f} fiat ({target_fiat_ratio*100:.0f}%) / ${target_crypto_value:.0f} crypto ({target_crypto_ratio*100:.0f}%)")

            # Calculate how much we need to adjust
            fiat_excess = available_fiat - target_fiat
            crypto_shortage_value = target_crypto_value - crypto_value

            # Check if rebalancing is needed (use small threshold for precision)
            threshold = total_portfolio_value * 0.01  # 1% threshold for rebalancing
            self.logger.debug(f"💡 Rebalancing check: fiat_excess=${fiat_excess:.0f}, crypto_shortage=${crypto_shortage_value:.0f}")
            self.logger.debug(f"💡 Thresholds: fiat_excess>{threshold:.0f}, crypto_shortage>0")

            # Check what type of rebalancing is needed
            if fiat_excess > threshold and crypto_shortage_value > 0:
                # We have excess fiat and need more crypto - buy crypto to achieve 50/50 balance
                crypto_to_buy = crypto_shortage_value / current_price  # Buy exactly what's needed

                if crypto_to_buy > 0.001:
                    self.logger.info(f"⚖️ Balancing: Buy {crypto_to_buy:.4f} crypto (${crypto_to_buy * current_price:.0f}) with excess fiat")
                    await self._execute_market_buy_order(current_price, crypto_to_buy)
                else:
                    self.logger.debug("📊 Fiat excess too small to rebalance")

            elif crypto_shortage_value < -threshold and fiat_excess < 0:
                # We have excess crypto and need more fiat - sell crypto to achieve 50/50 balance
                crypto_excess_value = -crypto_shortage_value
                crypto_to_sell = crypto_excess_value / current_price  # Sell exactly what's needed

                if crypto_to_sell > 0.001:
                    self.logger.info(f"⚖️ Balancing: Sell {crypto_to_sell:.4f} crypto (${crypto_to_sell * current_price:.0f}) to get more fiat")
                    await self._execute_market_sell_order(current_price, crypto_to_sell)
                else:
                    self.logger.debug("📊 Crypto excess too small to rebalance")
            else:
                self.logger.debug(f"📊 Portfolio balance acceptable for grid restart (no rebalancing needed)")

        except Exception as e:
            self.logger.warning(f"Top boundary rebalancing failed, continuing with current balances: {e}")

    async def _rebalance_for_bottom_boundary(self, current_price: float, available_fiat: float, available_crypto: float) -> None:
        """
        Rebalancing logic for bottom boundary hit:
        - Use available fiat to establish new grid levels BELOW current bottom
        - Keep same spacing as original grid
        - Use equal-dollar sizing for new lower levels
        - Don't sell existing crypto, keep it for potential selling
        """
        try:
            crypto_value = available_crypto * current_price

            self.logger.info(f"Bottom boundary rebalancing:")
            self.logger.info(f"  - Available fiat: ${available_fiat:.2f}")
            self.logger.info(f"  - Available crypto: {available_crypto:.6f} (${crypto_value:.2f})")
            self.logger.info(f"  - Strategy: Use fiat to create lower grid levels, keep crypto for selling")

            # For bottom boundary, we use available fiat to extend the grid downward
            # The existing crypto stays available for potential sell orders
            # New grid will be calculated to use the available fiat optimally

            if available_fiat > 0:
                self.logger.info(f"Will use ${available_fiat:.2f} fiat to create new lower grid levels")
                self.logger.info(f"Existing {available_crypto:.6f} crypto will remain available for sell orders")
            else:
                self.logger.warning("No fiat available for creating lower grid levels")

        except Exception as e:
            self.logger.warning(f"Bottom boundary rebalancing failed, continuing with current balances: {e}")

    async def _execute_market_buy_order(self, current_price: float, crypto_amount: float) -> None:
        """Execute a market buy order for rebalancing purposes"""
        try:
            total_cost = crypto_amount * current_price

            # Simulate the market buy by updating balances directly for backtest mode
            if self.trading_mode.value == "backtest":
                self.balance_tracker.balance -= total_cost
                self.balance_tracker.crypto_balance += crypto_amount
                self.logger.debug(f"Market buy: +{crypto_amount:.4f} crypto for ${total_cost:.0f}")
            else:
                self.logger.warning("Live market orders not implemented")

        except Exception as e:
            self.logger.error(f"Failed to execute market buy order: {e}")

    async def _execute_market_sell_order(self, current_price: float, crypto_amount: float) -> None:
        """Execute a market sell order for rebalancing purposes"""
        try:
            total_revenue = crypto_amount * current_price

            # Simulate the market sell by updating balances directly for backtest mode
            if self.trading_mode.value == "backtest":
                self.balance_tracker.balance += total_revenue
                self.balance_tracker.crypto_balance -= crypto_amount
                self.logger.debug(f"Market sell: -{crypto_amount:.4f} crypto for ${total_revenue:.0f}")
            else:
                self.logger.warning("Live market orders not implemented")

        except Exception as e:
            self.logger.error(f"Failed to execute market sell order: {e}")

    def _reset_grid_state(self) -> None:
        """
        Resets internal grid state for a fresh restart.
        Preserves all performance metrics and trading history.
        """
        # Clear grid-specific tracking data
        grid_levels_cleared = len(self._grid_buy_costs)
        self._grid_buy_costs.clear()

        # Performance data that is PRESERVED across restarts:
        # - self._cumulative_profit (total profit from all grid cycles)
        # - self.balance_tracker.total_fees (total fees paid)
        # - self.balance_tracker.balance and crypto_balance (account balances)
        # - self.data (historical performance data for backtesting)

        self.logger.debug(f"🔄 Grid state reset: cleared {grid_levels_cleared} cost basis entries")
        self.logger.debug(f"📊 Performance preserved: ${self._cumulative_profit:.2f} total profit, ${self.balance_tracker.total_fees:.2f} total fees")

    async def _on_order_filled(self, order) -> None:
        """
        Track cumulative profit from grid trading pairs.
        In grid trading, we only make profit when selling at a higher grid level than we bought.
        
        Args:
            order: The filled order.
        """
        from core.order_handling.order import OrderSide, OrderType
        
        # Get the grid level for this order
        grid_level = self.order_manager.order_book.get_grid_level_for_order(order)
        if not grid_level:
            # Check if this is the initial purchase order (market order without grid level)
            if order.side == OrderSide.BUY and order.order_type == OrderType.MARKET:
                # Track initial purchase cost
                buy_cost = order.filled * order.price
                buy_fee = order.fee.get("cost", 0.0) if order.fee else 0.0
                self._initial_purchase_cost = buy_cost + buy_fee
                self._initial_purchase_quantity = order.filled
                self.logger.debug(f"Initial purchase tracked: {order.filled:.6f} @ ${order.price:.2f} (cost: ${self._initial_purchase_cost:.2f})")
            return
            
        grid_price = grid_level.price
        
        if order.side == OrderSide.BUY:
            # Track buy cost at this grid level
            buy_cost = order.filled * order.price
            buy_fee = order.fee.get("cost", 0.0) if order.fee else 0.0
            total_buy_cost = buy_cost + buy_fee
            
            # Store the cost basis for this grid level
            if grid_price not in self._grid_buy_costs:
                self._grid_buy_costs[grid_price] = {'total_cost': 0.0, 'quantity': 0.0}
            
            self._grid_buy_costs[grid_price]['total_cost'] += total_buy_cost
            self._grid_buy_costs[grid_price]['quantity'] += order.filled
            
            self.logger.debug(f"Buy tracked at grid ${grid_price:.2f}: {order.filled:.6f} @ ${order.price:.2f} (cost: ${total_buy_cost:.2f})")
            
        elif order.side == OrderSide.SELL:
            # For sells, we need to find which buy level this came from
            # In grid trading, sells are always at higher levels than their corresponding buys
            sell_revenue = order.filled * order.price
            sell_fee = order.fee.get("cost", 0.0) if order.fee else 0.0
            net_revenue = sell_revenue - sell_fee
            
            # Find the corresponding buy level (should be the paired buy level)
            buy_grid_level = None
            if hasattr(grid_level, 'paired_buy_level') and grid_level.paired_buy_level:
                buy_grid_level = grid_level.paired_buy_level
            else:
                # Fallback: find the closest lower grid level with buy costs
                buy_price = None
                for price in sorted(self._grid_buy_costs.keys(), reverse=True):
                    if price < grid_price and self._grid_buy_costs[price]['quantity'] > 0:
                        buy_price = price
                        break
                
                if buy_price:
                    # Create a mock grid level object for the buy price
                    class MockGridLevel:
                        def __init__(self, price):
                            self.price = price
                    buy_grid_level = MockGridLevel(buy_price)
            
            if buy_grid_level and buy_grid_level.price in self._grid_buy_costs:
                buy_price = buy_grid_level.price
                buy_data = self._grid_buy_costs[buy_price]
                
                if buy_data['quantity'] >= order.filled:
                    # Calculate cost basis for this sell quantity
                    cost_per_unit = buy_data['total_cost'] / buy_data['quantity']
                    cost_basis = order.filled * cost_per_unit
                    
                    # Calculate profit (should always be positive in grid trading)
                    profit = net_revenue - cost_basis
                    self._cumulative_profit += profit
                    
                    # Update buy data
                    buy_data['quantity'] -= order.filled
                    buy_data['total_cost'] -= cost_basis
                    
                    # Clean up if quantity becomes zero
                    if buy_data['quantity'] <= 0.001:  # Small threshold for floating point precision
                        del self._grid_buy_costs[buy_price]
                    
                    self.logger.info(f"💰 Profit: ${profit:.2f} | Total: ${self._cumulative_profit:.2f}")
                else:
                    self.logger.warning(f"Insufficient buy quantity at grid ${buy_price:.2f} for sell order")
            else:
                # Try to use initial purchase cost as fallback
                if self._initial_purchase_cost and self._initial_purchase_quantity and self._initial_purchase_quantity >= order.filled:
                    # Calculate cost basis from initial purchase
                    cost_per_unit = self._initial_purchase_cost / self._initial_purchase_quantity
                    cost_basis = order.filled * cost_per_unit
                    
                    # Calculate profit from initial purchase
                    profit = net_revenue - cost_basis
                    self._cumulative_profit += profit
                    
                    # Update initial purchase data
                    self._initial_purchase_quantity -= order.filled
                    self._initial_purchase_cost -= cost_basis
                    
                    # Clean up if quantity becomes zero
                    if self._initial_purchase_quantity <= 0.001:
                        self._initial_purchase_cost = None
                        self._initial_purchase_quantity = None
                    
                    self.logger.info(f"💰 Profit (from initial): ${profit:.2f} | Total: ${self._cumulative_profit:.2f}")
                else:
                    self.logger.warning(f"No corresponding buy level found for sell at grid ${grid_price:.2f}")

    async def _handle_take_profit(self, current_price: float) -> bool:
        """
        Handles take-profit logic and executes a TP order if conditions are met.
        Returns True if take-profit is triggered.
        """
        if (
            self.config_manager.is_take_profit_enabled()
            and current_price >= self.config_manager.get_take_profit_threshold()
        ):
            self.logger.info(f"Take-profit triggered at {current_price}. Executing TP order...")
            # Stop all grid operations immediately to prevent competing for crypto balance
            self._stop_trading = True
            # Give a brief moment for any pending grid orders to complete first
            import asyncio
            await asyncio.sleep(0.001)  # 1ms delay to ensure pending orders are processed
            await self.order_manager.execute_take_profit_or_stop_loss_order(
                current_price=current_price,
                take_profit_order=True,
            )
            return True
        return False

    async def _handle_stop_loss(self, current_price: float) -> bool:
        """
        Handles stop-loss logic and executes an SL order if conditions are met.
        Returns True if stop-loss is triggered.
        """
        if (
            self.config_manager.is_stop_loss_enabled()
            and current_price <= self.config_manager.get_stop_loss_threshold()
        ):
            self.logger.info(f"Stop-loss triggered at {current_price}. Executing SL order...")
            await self.order_manager.execute_take_profit_or_stop_loss_order(
                current_price=current_price,
                stop_loss_order=True,
            )
            return True
        return False

    def get_formatted_orders(self):
        """
        Retrieves a formatted summary of all orders.

        Returns:
            list: A list of formatted orders.
        """
        return self.trading_performance_analyzer.get_formatted_orders()
