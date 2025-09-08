from unittest.mock import Mock

import numpy as np
import pytest

from config.config_manager import ConfigManager
from core.grid_management.grid_level import GridCycleState, GridLevel
from core.grid_management.grid_manager import GridManager
from core.order_handling.order import Order, OrderSide
from strategies.order_sizing_type import OrderSizingType
from strategies.range_mode import RangeMode
from strategies.spacing_type import SpacingType
from strategies.strategy_type import StrategyType


class TestGridManager:
    @pytest.fixture
    def config_manager(self):
        mock_config_manager = Mock(spec=ConfigManager)
        mock_config_manager.get_bottom_range.return_value = 1000
        mock_config_manager.get_top_range.return_value = 2000
        mock_config_manager.get_num_grids.return_value = 10
        mock_config_manager.get_spacing_type.return_value = SpacingType.ARITHMETIC
        return mock_config_manager

    @pytest.fixture
    def grid_manager(self, config_manager):
        return GridManager(config_manager, StrategyType.SIMPLE_GRID)

    def test_initialize_grids_and_levels_simple_grid(self, grid_manager):
        grid_manager.initialize_grids_and_levels()
        assert len(grid_manager.grid_levels) == len(grid_manager.price_grids)

        for price, grid_level in grid_manager.grid_levels.items():
            assert isinstance(grid_level, GridLevel)
            if price <= grid_manager.central_price:
                assert grid_level.state == GridCycleState.READY_TO_BUY
            else:
                assert grid_level.state == GridCycleState.READY_TO_SELL

    def test_initialize_grids_and_levels_hedged_grid(self, config_manager):
        grid_manager = GridManager(config_manager, StrategyType.HEDGED_GRID)

        grid_manager.initialize_grids_and_levels()
        assert len(grid_manager.grid_levels) == len(grid_manager.price_grids)

        for price, grid_level in grid_manager.grid_levels.items():
            assert isinstance(grid_level, GridLevel)
            if price == grid_manager.price_grids[-1]:
                assert grid_level.state == GridCycleState.READY_TO_SELL
            else:
                assert grid_level.state == GridCycleState.READY_TO_BUY_OR_SELL

    def test_get_trigger_price(self, grid_manager):
        grid_manager.initialize_grids_and_levels()
        assert grid_manager.get_trigger_price() == grid_manager.central_price

    def test_get_order_size_for_grid_level_equal_crypto(self, grid_manager):
        """Test equal crypto order sizing (default behavior)"""
        grid_manager.initialize_grids_and_levels()
        grid_price = 2000
        total_balance = 10000
        
        # Mock config to return EQUAL_CRYPTO
        grid_manager.config_manager.get_order_sizing_type = Mock(return_value=OrderSizingType.EQUAL_CRYPTO)
        
        # Equal crypto uses central price to determine crypto amount
        expected_crypto_amount = (total_balance / len(grid_manager.grid_levels)) / grid_manager.central_price
        result = grid_manager.get_order_size_for_grid_level(total_balance, grid_price)
        assert result == expected_crypto_amount
        
    def test_get_order_size_for_grid_level_equal_dollar(self, grid_manager):
        """Test equal dollar order sizing"""
        grid_manager.initialize_grids_and_levels()
        grid_price = 2000
        total_balance = 10000
        
        # Mock config to return EQUAL_DOLLAR
        grid_manager.config_manager.get_order_sizing_type = Mock(return_value=OrderSizingType.EQUAL_DOLLAR)
        
        # Equal dollar divides total balance equally across grids
        expected_dollar_per_grid = total_balance / len(grid_manager.grid_levels)
        expected_crypto_amount = expected_dollar_per_grid / grid_price
        result = grid_manager.get_order_size_for_grid_level(total_balance, grid_price)
        assert result == expected_crypto_amount
        
    def test_get_order_size_for_grid_level_default_fallback(self, grid_manager):
        """Test fallback to equal crypto when order_sizing is None"""
        grid_manager.initialize_grids_and_levels()
        grid_price = 2000
        total_balance = 10000
        
        # Mock config to return None (fallback case)
        grid_manager.config_manager.get_order_sizing_type = Mock(return_value=None)
        
        # Should fallback to equal crypto behavior
        expected_crypto_amount = (total_balance / len(grid_manager.grid_levels)) / grid_manager.central_price
        result = grid_manager.get_order_size_for_grid_level(total_balance, grid_price)
        assert result == expected_crypto_amount

    def test_get_initial_order_quantity(self, grid_manager):
        current_fiat_balance = 5000  # Half of the total balance
        current_crypto_balance = 0.5
        current_price = 2000
        expected_quantity = (
            (current_fiat_balance + (current_crypto_balance * current_price)) / 2
            - (current_crypto_balance * current_price)
        ) / current_price
        result = grid_manager.get_initial_order_quantity(current_fiat_balance, current_crypto_balance, current_price)
        assert result == expected_quantity

    def test_pair_grid_levels(self, grid_manager):
        source_grid_level = Mock(spec=GridLevel, price=1000)
        target_grid_level = Mock(spec=GridLevel, price=1100)
        grid_manager.pair_grid_levels(source_grid_level, target_grid_level, pairing_type="buy")
        assert source_grid_level.paired_buy_level == target_grid_level
        assert target_grid_level.paired_sell_level == source_grid_level

    def test_pair_grid_levels_invalid_type(self, grid_manager):
        source_grid_level = Mock(spec=GridLevel, price=1000)
        target_grid_level = Mock(spec=GridLevel, price=1100)

        with pytest.raises(ValueError, match="Invalid pairing type"):
            grid_manager.pair_grid_levels(source_grid_level, target_grid_level, pairing_type="invalid")

    def test_get_paired_sell_level_simple_grid(self, grid_manager):
        grid_manager.initialize_grids_and_levels()
        buy_grid_level = grid_manager.grid_levels[grid_manager.sorted_buy_grids[0]]
        paired_sell_level = grid_manager.get_paired_sell_level(buy_grid_level)
        assert paired_sell_level.price > buy_grid_level.price
        assert paired_sell_level.state == GridCycleState.READY_TO_SELL

    def test_get_paired_sell_level_hedged_grid(self, config_manager):
        grid_manager = GridManager(config_manager, StrategyType.HEDGED_GRID)
        grid_manager.initialize_grids_and_levels()

        buy_grid_level = grid_manager.grid_levels[grid_manager.sorted_buy_grids[0]]
        paired_sell_level = grid_manager.get_paired_sell_level(buy_grid_level)

        assert paired_sell_level is not None
        assert paired_sell_level.price > buy_grid_level.price
        assert paired_sell_level.state in {GridCycleState.READY_TO_SELL, GridCycleState.READY_TO_BUY_OR_SELL}

    def test_get_grid_level_below(self, grid_manager):
        grid_manager.initialize_grids_and_levels()
        grid_level = grid_manager.grid_levels[grid_manager.sorted_buy_grids[1]]
        lower_level = grid_manager.get_grid_level_below(grid_level)
        assert lower_level.price < grid_level.price

    def test_mark_order_pending_after_buy(self, grid_manager):
        grid_manager.initialize_grids_and_levels()
        grid_level = grid_manager.grid_levels[grid_manager.sorted_buy_grids[0]]
        order = Mock(spec=Order, side=OrderSide.BUY)

        grid_manager.mark_order_pending(grid_level, order)
        assert grid_level.state == GridCycleState.WAITING_FOR_BUY_FILL
        assert order in grid_level.orders

    def test_mark_order_pending_after_sell(self, grid_manager):
        grid_manager.initialize_grids_and_levels()
        grid_level = grid_manager.grid_levels[grid_manager.sorted_sell_grids[0]]
        order = Mock(spec=Order, side=OrderSide.SELL)

        grid_manager.mark_order_pending(grid_level, order)

        assert grid_level.state == GridCycleState.WAITING_FOR_SELL_FILL
        assert order in grid_level.orders

    def test_complete_order_simple_grid(self, grid_manager):
        grid_manager.initialize_grids_and_levels()
        grid_level = grid_manager.grid_levels[grid_manager.sorted_buy_grids[0]]

        grid_manager.complete_order(grid_level, OrderSide.BUY)
        assert grid_level.state == GridCycleState.READY_TO_SELL

    def test_complete_order_hedged_grid(self, config_manager):
        grid_manager = GridManager(config_manager, StrategyType.HEDGED_GRID)
        grid_manager.initialize_grids_and_levels()

        grid_level = grid_manager.grid_levels[grid_manager.sorted_buy_grids[0]]
        grid_manager.complete_order(grid_level, OrderSide.BUY)
        assert grid_level.state == GridCycleState.READY_TO_BUY_OR_SELL

    def test_complete_order_simple_grid_buy(self, grid_manager):
        grid_manager.initialize_grids_and_levels()
        grid_level = grid_manager.grid_levels[grid_manager.sorted_buy_grids[0]]

        grid_manager.complete_order(grid_level, OrderSide.BUY)

        # Validate the state transition to READY_TO_SELL
        assert grid_level.state == GridCycleState.READY_TO_SELL

    def test_complete_order_simple_grid_sell(self, grid_manager):
        grid_manager.initialize_grids_and_levels()
        grid_level = grid_manager.grid_levels[grid_manager.sorted_sell_grids[0]]

        grid_manager.complete_order(grid_level, OrderSide.SELL)

        assert grid_level.state == GridCycleState.READY_TO_BUY

    def test_complete_order_hedged_grid_buy(self, config_manager):
        grid_manager = GridManager(config_manager, StrategyType.HEDGED_GRID)
        grid_manager.initialize_grids_and_levels()

        buy_grid_level = grid_manager.grid_levels[grid_manager.sorted_buy_grids[0]]
        sell_grid_level = grid_manager.grid_levels[grid_manager.sorted_sell_grids[0]]

        # Pair levels for testing
        grid_manager.pair_grid_levels(sell_grid_level, buy_grid_level, "buy")

        grid_manager.complete_order(buy_grid_level, OrderSide.BUY)

        # Validate transitions
        assert buy_grid_level.state == GridCycleState.READY_TO_BUY_OR_SELL
        assert sell_grid_level.state == GridCycleState.READY_TO_SELL

    def test_complete_order_hedged_grid_sell(self, config_manager):
        grid_manager = GridManager(config_manager, StrategyType.HEDGED_GRID)
        grid_manager.initialize_grids_and_levels()

        sell_grid_level = grid_manager.grid_levels[grid_manager.sorted_sell_grids[0]]
        buy_grid_level = grid_manager.grid_levels[grid_manager.sorted_buy_grids[0]]

        # Pair levels for testing
        grid_manager.pair_grid_levels(buy_grid_level, sell_grid_level, "sell")

        grid_manager.complete_order(sell_grid_level, OrderSide.SELL)

        assert sell_grid_level.state == GridCycleState.READY_TO_BUY_OR_SELL
        assert buy_grid_level.state == GridCycleState.READY_TO_BUY

    def test_can_place_order_simple_grid(self, grid_manager):
        grid_manager.initialize_grids_and_levels()
        buy_grid_level = grid_manager.grid_levels[grid_manager.sorted_buy_grids[0]]
        sell_grid_level = grid_manager.grid_levels[grid_manager.sorted_sell_grids[0]]

        assert grid_manager.can_place_order(buy_grid_level, OrderSide.BUY) is True
        assert grid_manager.can_place_order(sell_grid_level, OrderSide.SELL) is True

    def test_can_place_order_hedged_grid(self, config_manager):
        grid_manager = GridManager(config_manager, StrategyType.HEDGED_GRID)
        grid_manager.initialize_grids_and_levels()

        buy_grid_level = grid_manager.grid_levels[grid_manager.sorted_buy_grids[0]]
        sell_grid_level = grid_manager.grid_levels[grid_manager.sorted_sell_grids[0]]

        assert grid_manager.can_place_order(buy_grid_level, OrderSide.BUY) is True
        assert grid_manager.can_place_order(sell_grid_level, OrderSide.SELL) is True

    def test_calculate_price_grids_and_central_price_arithmetic(self, grid_manager):
        expected_grids = np.linspace(1000, 2000, 10)
        grids, central_price = grid_manager._calculate_price_grids_and_central_price()
        np.testing.assert_array_equal(grids, expected_grids)
        assert central_price == 1500

    def test_calculate_price_grids_and_central_price_geometric(self, config_manager):
        config_manager.get_spacing_type.return_value = SpacingType.GEOMETRIC
        config_manager.get_top_range.return_value = 2000
        config_manager.get_bottom_range.return_value = 1000
        grid_manager = GridManager(config_manager, StrategyType.HEDGED_GRID)

        expected_grids = [
            1000,
            1080.059738892306,
            1166.5290395761165,
            1259.921049894873,
            1360.7900001743767,
            1469.7344922755985,
            1587.401051968199,
            1714.4879657061451,
            1851.7494245745802,
            2000,
        ]
        grids, central_price = grid_manager._calculate_price_grids_and_central_price()
        np.testing.assert_array_almost_equal(grids, expected_grids, decimal=5)
        assert central_price == 1415.2622462249876

    def test_range_mode_crypto_zero(self, config_manager):
        """Test CRYPTO_ZERO range mode calculation"""
        # Mock config for crypto_zero mode
        config_manager.get_range_mode.return_value = RangeMode.CRYPTO_ZERO
        config_manager.get_num_grids.return_value = 10
        config_manager.get_spacing_type.return_value = SpacingType.ARITHMETIC
        
        grid_manager = GridManager(config_manager, StrategyType.SIMPLE_GRID)
        
        first_price = 100.0
        bottom_range, top_range, num_grids, spacing_type = grid_manager._extract_grid_config(first_price)
        
        # crypto_zero formula: bottom = price/5, top = price + (price - bottom)
        expected_bottom = first_price / 5  # 20.0
        expected_top = first_price + (first_price - expected_bottom)  # 100 + 80 = 180.0
        
        assert bottom_range == expected_bottom
        assert top_range == expected_top
        assert num_grids == 10
        assert spacing_type == SpacingType.ARITHMETIC

    def test_range_mode_manual(self, config_manager):
        """Test MANUAL range mode uses configured values"""
        # Mock config for manual mode
        config_manager.get_range_mode.return_value = RangeMode.MANUAL
        config_manager.get_num_grids.return_value = 5
        config_manager.get_spacing_type.return_value = SpacingType.GEOMETRIC
        config_manager.get_top_range.return_value = 200.0
        config_manager.get_bottom_range.return_value = 50.0
        
        grid_manager = GridManager(config_manager, StrategyType.SIMPLE_GRID)
        
        bottom_range, top_range, num_grids, spacing_type = grid_manager._extract_grid_config()
        
        assert bottom_range == 50.0
        assert top_range == 200.0
        assert num_grids == 5
        assert spacing_type == SpacingType.GEOMETRIC

    def test_range_mode_crypto_zero_requires_first_price(self, config_manager):
        """Test CRYPTO_ZERO mode raises error without first_price"""
        config_manager.get_range_mode.return_value = RangeMode.CRYPTO_ZERO
        
        grid_manager = GridManager(config_manager, StrategyType.SIMPLE_GRID)
        
        with pytest.raises(ValueError, match="first_price is required for CRYPTO_ZERO range mode"):
            grid_manager._extract_grid_config(first_price=None)

    def test_initialize_grids_with_crypto_zero_mode(self, config_manager):
        """Test full grid initialization with CRYPTO_ZERO mode"""
        config_manager.get_range_mode.return_value = RangeMode.CRYPTO_ZERO
        config_manager.get_num_grids.return_value = 5
        config_manager.get_spacing_type.return_value = SpacingType.ARITHMETIC
        config_manager.get_order_sizing_type.return_value = OrderSizingType.EQUAL_DOLLAR
        
        grid_manager = GridManager(config_manager, StrategyType.SIMPLE_GRID)
        
        first_price = 150.0
        grid_manager.initialize_grids_and_levels(first_price)
        
        # Verify grids were created with crypto_zero formula
        expected_bottom = first_price / 5  # 30.0
        expected_top = first_price + (first_price - expected_bottom)  # 150 + 120 = 270.0
        
        # Check that grids span the expected range
        assert min(grid_manager.price_grids) == expected_bottom
        assert max(grid_manager.price_grids) == expected_top
        assert len(grid_manager.price_grids) == 5

    def test_crypto_zero_auto_configures_risk_management(self, config_manager):
        """Test that crypto_zero mode auto-configures take profit and stop loss"""
        config_manager.get_range_mode.return_value = RangeMode.CRYPTO_ZERO
        config_manager.get_num_grids.return_value = 10
        config_manager.get_spacing_type.return_value = SpacingType.ARITHMETIC
        config_manager.get_order_sizing_type.return_value = OrderSizingType.EQUAL_DOLLAR
        
        grid_manager = GridManager(config_manager, StrategyType.SIMPLE_GRID)
        
        first_price = 200.0
        grid_manager.initialize_grids_and_levels(first_price)
        
        # Verify auto-calculated ranges were set in config manager
        expected_bottom = first_price / 5  # 40.0
        expected_top = first_price + (first_price - expected_bottom)  # 200 + 160 = 360.0
        
        # Verify set_auto_calculated_ranges was called
        config_manager.set_auto_calculated_ranges.assert_called_once_with(expected_bottom, expected_top)

    def test_config_manager_returns_auto_thresholds_for_crypto_zero(self):
        """Test config manager returns auto-configured thresholds for crypto_zero mode"""
        from config.config_manager import ConfigManager
        from config.config_validator import ConfigValidator
        
        # Create a mock config
        mock_config = {
            "grid_strategy": {
                "range": {"mode": "crypto_zero"},
            },
            "risk_management": {
                "take_profit": {"enabled": True, "threshold": 999},  # Should be overridden
                "stop_loss": {"enabled": True, "threshold": 888},   # Should be overridden
            }
        }
        
        config_manager = ConfigManager.__new__(ConfigManager)
        config_manager.config = mock_config
        config_manager.logger = Mock()
        
        # Set auto-calculated values
        config_manager.set_auto_calculated_ranges(50.0, 300.0)
        
        # Test auto-configured values
        assert config_manager.get_take_profit_threshold() == 300.0  # top range
        assert config_manager.get_stop_loss_threshold() == 0.0      # always 0 for crypto_zero
