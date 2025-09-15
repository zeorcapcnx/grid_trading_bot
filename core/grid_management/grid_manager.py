import logging

import numpy as np

from config.config_manager import ConfigManager
from strategies.order_sizing_type import OrderSizingType
from strategies.range_mode import RangeMode
from strategies.spacing_type import SpacingType
from strategies.strategy_type import StrategyType

from ..order_handling.order import Order, OrderSide
from .grid_level import GridCycleState, GridLevel


class GridManager:
    def __init__(
        self,
        config_manager: ConfigManager,
        strategy_type: StrategyType,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_manager: ConfigManager = config_manager
        self.strategy_type: StrategyType = strategy_type
        self.price_grids: list[float]
        self.central_price: float
        self.sorted_buy_grids: list[float]
        self.sorted_sell_grids: list[float]
        self.grid_levels: dict[float, GridLevel] = {}

    def initialize_grids_and_levels(self, first_price: float | None = None) -> None:
        """
        Initializes the grid levels and assigns their respective states based on the chosen strategy.

        For the `SIMPLE_GRID` strategy:
        - Buy orders are placed on grid levels below the central price.
        - Sell orders are placed on grid levels above the central price.
        - Levels are initialized with `READY_TO_BUY` or `READY_TO_SELL` states.

        For the `HEDGED_GRID` strategy:
        - Grid levels are divided into buy levels (all except the top grid) and
        sell levels (all except the bottom grid).
        - Buy grid levels are initialized with `READY_TO_BUY`, except for the topmost grid.
        - Sell grid levels are initialized with `READY_TO_SELL`.
        """
        self.price_grids, self.central_price = self._calculate_price_grids_and_central_price(first_price)

        if self.strategy_type == StrategyType.SIMPLE_GRID:
            self.sorted_buy_grids = [price_grid for price_grid in self.price_grids if price_grid <= self.central_price]
            self.sorted_sell_grids = [price_grid for price_grid in self.price_grids if price_grid > self.central_price]
            self.grid_levels = {
                price: GridLevel(
                    price,
                    GridCycleState.READY_TO_BUY if price <= self.central_price else GridCycleState.READY_TO_SELL,
                )
                for price in self.price_grids
            }

        elif self.strategy_type == StrategyType.HEDGED_GRID:
            self.sorted_buy_grids = self.price_grids[:-1]  # All except the top grid
            self.sorted_sell_grids = self.price_grids[1:]  # All except the bottom grid
            self.grid_levels = {
                price: GridLevel(
                    price,
                    GridCycleState.READY_TO_BUY_OR_SELL
                    if price != self.price_grids[-1]
                    else GridCycleState.READY_TO_SELL,
                )
                for price in self.price_grids
            }
        self.logger.info(f"📊 Grid initialized: {len(self.price_grids)} levels, trigger ${self.central_price:.2f}")
        self.logger.debug(f"Range: ${min(self.price_grids):.2f} - ${max(self.price_grids):.2f}")
        self.logger.debug(f"Buy levels: {len(self.sorted_buy_grids)}, Sell levels: {len(self.sorted_sell_grids)}")

    def get_trigger_price(self) -> float:
        return self.central_price

    def get_order_size_for_grid_level(
        self,
        total_balance: float,
        grid_price: float,
    ) -> float:
        """
        Calculates the order size for a grid level based on the selected order sizing strategy.

        Args:
            total_balance: The total balance available for trading.
            grid_price: The price of the specific grid level.

        Returns:
            The calculated order size as a float (crypto amount).
        """
        order_sizing_type = self.config_manager.get_order_sizing_type()
        total_grids = len(self.grid_levels)
        
        if order_sizing_type == OrderSizingType.EQUAL_DOLLAR:
            # Equal dollar amount per grid - same dollar value at each grid level
            dollar_amount_per_grid = total_balance / total_grids
            order_size = dollar_amount_per_grid / grid_price
        else:
            # EQUAL_CRYPTO - same crypto amount at each grid level (current behavior)
            # This uses central/current price to determine crypto amount, then same amount at all levels
            central_price = self.central_price
            crypto_amount_per_grid = (total_balance / total_grids) / central_price
            order_size = crypto_amount_per_grid
            
        return order_size

    def get_initial_order_quantity(
        self,
        current_fiat_balance: float,
        current_crypto_balance: float,
        current_price: float,
    ) -> float:
        """
        Calculates the initial quantity of crypto to purchase for grid initialization.

        Args:
            current_fiat_balance (float): The current fiat balance.
            current_crypto_balance (float): The current crypto balance.
            current_price (float): The current market price of the crypto.

        Returns:
            float: The quantity of crypto to purchase.
        """
        current_crypto_value_in_fiat = current_crypto_balance * current_price
        total_portfolio_value = current_fiat_balance + current_crypto_value_in_fiat
        target_crypto_allocation_in_fiat = total_portfolio_value / 2  # Allocate 50% of balance for initial buy
        fiat_to_allocate_for_purchase = target_crypto_allocation_in_fiat - current_crypto_value_in_fiat
        fiat_to_allocate_for_purchase = max(0, min(fiat_to_allocate_for_purchase, current_fiat_balance))
        return fiat_to_allocate_for_purchase / current_price

    def pair_grid_levels(
        self,
        source_grid_level: GridLevel,
        target_grid_level: GridLevel,
        pairing_type: str,
    ) -> None:
        """
        Dynamically pairs grid levels for buy or sell purposes.

        Args:
            source_grid_level: The grid level initiating the pairing.
            target_grid_level: The grid level being paired.
            pairing_type: "buy" or "sell" to specify the type of pairing.
        """
        if pairing_type == "buy":
            source_grid_level.paired_buy_level = target_grid_level
            target_grid_level.paired_sell_level = source_grid_level
            self.logger.info(
                f"Paired sell grid level {source_grid_level.price} with buy grid level {target_grid_level.price}.",
            )

        elif pairing_type == "sell":
            source_grid_level.paired_sell_level = target_grid_level
            target_grid_level.paired_buy_level = source_grid_level
            self.logger.info(
                f"Paired buy grid level {source_grid_level.price} with sell grid level {target_grid_level.price}.",
            )

        else:
            raise ValueError(f"Invalid pairing type: {pairing_type}. Must be 'buy' or 'sell'.")

    def get_paired_sell_level(
        self,
        buy_grid_level: GridLevel,
    ) -> GridLevel | None:
        """
        Determines the paired sell level for a given buy grid level based on the strategy type.

        Args:
            buy_grid_level: The buy grid level for which the paired sell level is required.

        Returns:
            The paired sell grid level, or None if no valid level exists.
        """
        if self.strategy_type == StrategyType.SIMPLE_GRID:
            self.logger.info(f"Looking for paired sell level for buy level at {buy_grid_level}")
            self.logger.info(f"Available sell grids: {self.sorted_sell_grids}")

            for sell_price in self.sorted_sell_grids:
                sell_level = self.grid_levels[sell_price]
                self.logger.info(f"Checking sell level {sell_price}, state: {sell_level.state}")

                if sell_level and not self.can_place_order(sell_level, OrderSide.SELL):
                    self.logger.info(
                        f"Skipping sell level {sell_price} - cannot place order. State: {sell_level.state}",
                    )
                    continue

                if sell_price > buy_grid_level.price:
                    self.logger.info(f"Paired sell level found at {sell_price} for buy level {buy_grid_level}.")
                    return sell_level

            self.logger.warning(f"No suitable sell level found above {buy_grid_level}")
            return None

        elif self.strategy_type == StrategyType.HEDGED_GRID:
            self.logger.info(f"Available price grids: {self.price_grids}")
            sorted_prices = sorted(self.price_grids)
            current_index = sorted_prices.index(buy_grid_level.price)
            self.logger.info(f"Current index of buy level {buy_grid_level.price}: {current_index}")

            if current_index + 1 < len(sorted_prices):
                paired_sell_price = sorted_prices[current_index + 1]
                sell_level = self.grid_levels[paired_sell_price]
                self.logger.info(
                    f"Paired sell level for buy level {buy_grid_level.price} is at "
                    f"{paired_sell_price} (state: {sell_level.state})",
                )
                return sell_level

            self.logger.warning(f"No suitable sell level found for buy grid level {buy_grid_level}")
            return None

        else:
            self.logger.error(f"Unsupported strategy type: {self.strategy_type}")
            return None

    def get_grid_level_below(self, grid_level: GridLevel) -> GridLevel | None:
        """
        Returns the grid level immediately below the given grid level.

        Args:
            grid_level: The current grid level.

        Returns:
            The grid level below the given grid level, or None if it doesn't exist.
        """
        sorted_levels = sorted(self.grid_levels.keys())
        current_index = sorted_levels.index(grid_level.price)

        if current_index > 0:
            lower_price = sorted_levels[current_index - 1]
            return self.grid_levels[lower_price]
        return None

    def mark_order_pending(
        self,
        grid_level: GridLevel,
        order: Order,
    ) -> None:
        """
        Marks a grid level as having a pending order (buy or sell).

        Args:
            grid_level: The grid level to update.
            order: The Order object representing the pending order.
            order_side: The side of the order (buy or sell).
        """
        grid_level.add_order(order)

        if order.side == OrderSide.BUY:
            grid_level.state = GridCycleState.WAITING_FOR_BUY_FILL
            self.logger.info(f"Buy order placed and marked as pending at grid level {grid_level.price}.")
        elif order.side == OrderSide.SELL:
            grid_level.state = GridCycleState.WAITING_FOR_SELL_FILL
            self.logger.info(f"Sell order placed and marked as pending at grid level {grid_level.price}.")

    def complete_order(
        self,
        grid_level: GridLevel,
        order_side: OrderSide,
    ) -> None:
        """
        Marks the completion of an order (buy or sell) and transitions the grid level.

        Args:
            grid_level: The grid level where the order was completed.
            order_side: The side of the completed order (buy or sell).
        """
        if self.strategy_type == StrategyType.SIMPLE_GRID:
            if order_side == OrderSide.BUY:
                grid_level.state = GridCycleState.READY_TO_SELL
                self.logger.info(
                    f"Buy order completed at grid level {grid_level.price}. Transitioning to READY_TO_SELL.",
                )
            elif order_side == OrderSide.SELL:
                grid_level.state = GridCycleState.READY_TO_BUY
                self.logger.info(
                    f"Sell order completed at grid level {grid_level.price}. Transitioning to READY_TO_BUY.",
                )

        elif self.strategy_type == StrategyType.HEDGED_GRID:
            if order_side == OrderSide.BUY:
                grid_level.state = GridCycleState.READY_TO_BUY_OR_SELL
                self.logger.info(
                    f"Buy order completed at grid level {grid_level.price}. Transitioning to READY_TO_BUY_OR_SELL.",
                )

                # Transition the paired buy level to "READY_TO_SELL"
                if grid_level.paired_sell_level:
                    grid_level.paired_sell_level.state = GridCycleState.READY_TO_SELL
                    self.logger.info(
                        f"Paired sell grid level {grid_level.paired_sell_level.price} transitioned to READY_TO_SELL.",
                    )

            elif order_side == OrderSide.SELL:
                grid_level.state = GridCycleState.READY_TO_BUY_OR_SELL
                self.logger.info(
                    f"Sell order completed at grid level {grid_level.price}. Transitioning to READY_TO_BUY_OR_SELL.",
                )

                # Transition the paired buy level to "READY_TO_BUY"
                if grid_level.paired_buy_level:
                    grid_level.paired_buy_level.state = GridCycleState.READY_TO_BUY
                    self.logger.info(
                        f"Paired buy grid level {grid_level.paired_buy_level.price} transitioned to READY_TO_BUY.",
                    )

        else:
            self.logger.error("Unexpected strategy type")

    def can_place_order(
        self,
        grid_level: GridLevel,
        order_side: OrderSide,
    ) -> bool:
        """
        Determines if an order can be placed on the given grid level for the current strategy.

        Args:
            grid_level: The grid level being evaluated.
            order_side: The side of the order (buy or sell).

        Returns:
            bool: True if the order can be placed, False otherwise.
        """
        if self.strategy_type == StrategyType.SIMPLE_GRID:
            if order_side == OrderSide.BUY:
                return grid_level.state == GridCycleState.READY_TO_BUY
            elif order_side == OrderSide.SELL:
                return grid_level.state == GridCycleState.READY_TO_SELL

        elif self.strategy_type == StrategyType.HEDGED_GRID:
            if order_side == OrderSide.BUY:
                return grid_level.state in {GridCycleState.READY_TO_BUY, GridCycleState.READY_TO_BUY_OR_SELL}
            elif order_side == OrderSide.SELL:
                return grid_level.state in {GridCycleState.READY_TO_SELL, GridCycleState.READY_TO_BUY_OR_SELL}

        else:
            return False

    def _extract_grid_config(self, first_price: float | None = None) -> tuple[float, float, int, str]:
        """
        Extracts grid configuration parameters from the configuration manager.
        Calculates range automatically based on range mode if needed.
        """
        range_mode = self.config_manager.get_range_mode()
        num_grids = self.config_manager.get_num_grids()
        spacing_type = self.config_manager.get_spacing_type()
        
        if range_mode == RangeMode.CRYPTO_ZERO:
            if first_price is None:
                raise ValueError("first_price is required for CRYPTO_ZERO range mode")
            
            # crypto_zero formula: bottom = price/5, top = price + (price - bottom)
            bottom_range = first_price / 5
            top_range = first_price + (first_price - bottom_range)
            
            self.logger.info(f"CRYPTO_ZERO range mode: first_price={first_price:.2f}, bottom={bottom_range:.2f}, top={top_range:.2f}")
            
            # Auto-configure take profit and stop loss thresholds  
            # TP = top_range, SL = 0.0
            self.config_manager.set_auto_calculated_ranges(bottom_range, top_range)
            
        else:
            # Default to MANUAL mode - use configured values
            bottom_range = self.config_manager.get_bottom_range()
            top_range = self.config_manager.get_top_range()
            
        return bottom_range, top_range, num_grids, spacing_type

    def _calculate_price_grids_and_central_price(self, first_price: float | None = None) -> tuple[list[float], float]:
        """
        Calculates price grids and the central price based on the configuration.

        Args:
            first_price: The first candle price for auto-calculating ranges

        Returns:
            Tuple[List[float], float]: A tuple containing:
                - grids (List[float]): The list of calculated grid prices.
                - central_price (float): The central price of the grid.
        """
        bottom_range, top_range, num_grids, spacing_type = self._extract_grid_config(first_price)

        if spacing_type == SpacingType.ARITHMETIC:
            # For even number of grids, add +1 to make it odd
            actual_num_grids = num_grids + 1 if num_grids % 2 == 0 else num_grids
            all_grids = np.linspace(bottom_range, top_range, actual_num_grids)
            
            if num_grids % 2 == 0:
                # Store central price
                central_index = len(all_grids) // 2
                central_price = all_grids[central_index]
                
                # For HEDGED_GRID strategy, keep central price in grid levels
                if self.strategy_type == StrategyType.HEDGED_GRID:
                    grids = all_grids  # Keep all grids including central price
                else:
                    # For SIMPLE_GRID, remove the middle grid to get back to original count
                    grids = np.concatenate([all_grids[:central_index], all_grids[central_index+1:]])
            else:
                # Odd grids: central price is the middle grid
                grids = all_grids
                central_index = len(grids) // 2
                central_price = grids[central_index]

        elif spacing_type == SpacingType.GEOMETRIC:
            # For even number of grids, add +1 to make it odd
            actual_num_grids = num_grids + 1 if num_grids % 2 == 0 else num_grids
            all_grids = []
            ratio = (top_range / bottom_range) ** (1 / (actual_num_grids - 1))
            current_price = bottom_range

            for _ in range(actual_num_grids):
                all_grids.append(current_price)
                current_price *= ratio

            if num_grids % 2 == 0:
                # Store central price
                central_index = len(all_grids) // 2
                central_price = all_grids[central_index]
                
                # For HEDGED_GRID strategy, keep central price in grid levels
                if self.strategy_type == StrategyType.HEDGED_GRID:
                    grids = all_grids  # Keep all grids including central price
                else:
                    # For SIMPLE_GRID, remove the middle grid to get back to original count
                    grids = all_grids[:central_index] + all_grids[central_index+1:]
            else:
                # Odd grids: central price is the middle grid
                grids = all_grids
                central_index = len(grids) // 2
                central_price = grids[central_index]

        else:
            raise ValueError(f"Unsupported spacing type: {spacing_type}")

        return grids, central_price
