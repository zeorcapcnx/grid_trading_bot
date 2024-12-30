import pytest, asyncio, ccxt
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from core.services.live_exchange_service import LiveExchangeService
from core.services.exceptions import UnsupportedExchangeError, MissingEnvironmentVariableError, DataFetchError, OrderCancellationError
from config.config_manager import ConfigManager
from config.trading_mode import TradingMode

class TestLiveExchangeService:
    @pytest.fixture
    def config_manager(self):
        config_manager = Mock(spec=ConfigManager)
        config_manager.get_exchange_name.return_value = "binance"
        config_manager.get_trading_mode.return_value = TradingMode.LIVE
        return config_manager

    @pytest.fixture
    def mock_exchange_instance(self):
        exchange = AsyncMock()
        exchange.fetch_balance.return_value = {"total": {"USD": 1000}}
        exchange.fetch_ticker.return_value = {"last": 50000.0}
        exchange.fetch_order.return_value = {"status": "closed"}
        exchange.cancel_order.return_value = {"status": "canceled"}
        return exchange

    @pytest.fixture
    def setup_env_vars(self, monkeypatch):
        monkeypatch.setenv("EXCHANGE_API_KEY", "test_api_key")
        monkeypatch.setenv("EXCHANGE_SECRET_KEY", "test_secret_key")

    @patch("core.services.live_exchange_service.ccxt")
    @patch("core.services.live_exchange_service.getattr")
    def test_initialization_with_env_vars(self, mock_getattr, mock_ccxt, config_manager, setup_env_vars):
        mock_exchange_instance = Mock()
        mock_ccxt.binance.return_value = mock_exchange_instance
        mock_getattr.return_value = mock_ccxt.binance

        service = LiveExchangeService(config_manager, is_paper_trading_activated=False)
        assert isinstance(service, LiveExchangeService)
        assert service.exchange_name == "binance"
        assert service.api_key == "test_api_key"
        assert service.secret_key == "test_secret_key"
        assert service.exchange == mock_exchange_instance
        assert service.exchange.enableRateLimit, "Expected rate limiting to be enabled for live mode"

    @patch("core.services.live_exchange_service.getattr")
    def test_missing_secret_key_raises_error(self, config_manager, monkeypatch):
        monkeypatch.delenv("EXCHANGE_SECRET_KEY", raising=False)
        monkeypatch.setenv("EXCHANGE_API_KEY", "test_api_key")

        with pytest.raises(MissingEnvironmentVariableError, match="Missing required environment variable: EXCHANGE_SECRET_KEY"):
            LiveExchangeService(config_manager, is_paper_trading_activated=False)

    @patch("core.services.live_exchange_service.getattr")
    def test_missing_api_key_raises_error(self, config_manager, monkeypatch):
        monkeypatch.delenv("EXCHANGE_API_KEY", raising=False)
        monkeypatch.setenv("EXCHANGE_SECRET_KEY", "test_secret_key")

        with pytest.raises(MissingEnvironmentVariableError, match="Missing required environment variable: EXCHANGE_API_KEY"):
            LiveExchangeService(config_manager, is_paper_trading_activated=False)

    @patch("core.services.live_exchange_service.ccxt")
    @patch("core.services.live_exchange_service.getattr")
    def test_sandbox_mode_initialization(self, mock_getattr, mock_ccxt, config_manager, setup_env_vars):
        config_manager.get_trading_mode.return_value = TradingMode.PAPER_TRADING

        mock_exchange_instance = MagicMock()
        mock_exchange_instance.urls = {'api': 'https://api.binance.com'}  # Initial URL setup
        mock_ccxt.binance.return_value = mock_exchange_instance
        mock_getattr.return_value = mock_ccxt.binance

        service = LiveExchangeService(config_manager, is_paper_trading_activated=True)

        assert service.is_paper_trading_activated is True
        expected_sandbox_url = "https://testnet.binance.vision/api"
        assert mock_exchange_instance.urls['api'] == expected_sandbox_url, "Sandbox URL not correctly set for Binance."

    @patch("core.services.live_exchange_service.getattr")
    def test_unsupported_exchange_raises_error(self, mock_getattr, config_manager, setup_env_vars):
        config_manager.get_exchange_name.return_value = "unsupported_exchange"
        mock_getattr.side_effect = AttributeError

        with pytest.raises(UnsupportedExchangeError, match="The exchange 'unsupported_exchange' is not supported."):
            LiveExchangeService(config_manager, is_paper_trading_activated=False)

    @patch("core.services.live_exchange_service.ccxt")
    @patch("core.services.live_exchange_service.getattr")
    @pytest.mark.asyncio
    async def test_place_order_successful(self, mock_getattr, mock_ccxt, config_manager, setup_env_vars, mock_exchange_instance):
        mock_getattr.return_value = mock_ccxt.binance
        mock_ccxt.binance.return_value = mock_exchange_instance

        service = LiveExchangeService(config_manager, is_paper_trading_activated=False)
        await service.place_order("BTC/USD",  "limit", "buy", 1, 50000.0)

        mock_exchange_instance.create_order.assert_called_once_with("BTC/USD",  "limit", "buy", 1, 50000.0)

    @patch("core.services.live_exchange_service.ccxt")
    @patch("core.services.live_exchange_service.getattr")
    @pytest.mark.asyncio
    async def test_place_order_unexpected_error(self, mock_getattr, mock_ccxt, config_manager, setup_env_vars, mock_exchange_instance):
        mock_getattr.return_value = mock_ccxt.binance
        mock_ccxt.binance.return_value = mock_exchange_instance
        mock_exchange_instance.create_order.side_effect = Exception("Unexpected error")
        service = LiveExchangeService(config_manager, is_paper_trading_activated=False)

        with pytest.raises(DataFetchError, match="Unexpected error placing order"):
            await service.place_order("BTC/USD", "market", "buy", 1, 50000.0)
        mock_exchange_instance.create_order.assert_awaited_once_with("BTC/USD", "market", "buy", 1, 50000.0)

    @patch("core.services.live_exchange_service.ccxt")
    @patch("core.services.live_exchange_service.getattr")
    @pytest.mark.asyncio
    async def test_get_current_price_successful(self, mock_getattr, mock_ccxt, config_manager, setup_env_vars, mock_exchange_instance):
        mock_getattr.return_value = mock_ccxt.binance
        mock_ccxt.binance.return_value = mock_exchange_instance

        service = LiveExchangeService(config_manager, is_paper_trading_activated=False)
        result = await service.get_current_price("BTC/USD")

        assert result == 50000.0
        mock_exchange_instance.fetch_ticker.assert_called_once_with("BTC/USD")

    @patch("core.services.live_exchange_service.ccxt")
    @patch("core.services.live_exchange_service.getattr")
    @pytest.mark.asyncio
    async def test_cancel_order_successful(self, mock_getattr, mock_ccxt, config_manager,  setup_env_vars, mock_exchange_instance):
        mock_getattr.return_value = mock_ccxt.binance
        mock_ccxt.binance.return_value = mock_exchange_instance

        service = LiveExchangeService(config_manager, is_paper_trading_activated=False)
        result = await service.cancel_order("order123", "BTC/USD")

        assert result["status"] == "canceled"
        mock_exchange_instance.cancel_order.assert_called_once_with("order123", "BTC/USD")
    
    @pytest.mark.asyncio
    @patch("core.services.live_exchange_service.ccxt")
    @patch("core.services.live_exchange_service.getattr")
    async def test_cancel_order_unexpected_error(self, mock_getattr, mock_ccxt, config_manager, setup_env_vars, mock_exchange_instance):
        mock_getattr.return_value = mock_ccxt.binance
        mock_ccxt.binance.return_value = mock_exchange_instance
        mock_exchange_instance.cancel_order.side_effect = Exception("Unexpected error")
        service = LiveExchangeService(config_manager, is_paper_trading_activated=False)

        with pytest.raises(OrderCancellationError, match="Unexpected error while canceling order order123: Unexpected error"):
            await service.cancel_order("order123", "BTC/USD")
        mock_exchange_instance.cancel_order.assert_awaited_once_with("order123", "BTC/USD")

    @pytest.mark.asyncio
    @patch("core.services.live_exchange_service.ccxt")
    @patch("core.services.live_exchange_service.getattr")
    async def test_cancel_order_network_error(self, mock_getattr, mock_ccxt, config_manager, setup_env_vars, mock_exchange_instance):
        mock_getattr.return_value = mock_ccxt.binance
        mock_ccxt.binance.return_value = mock_exchange_instance
        mock_exchange_instance.cancel_order.side_effect = ccxt.NetworkError("Network issue")

        service = LiveExchangeService(config_manager, is_paper_trading_activated=False)

        with pytest.raises(OrderCancellationError, match="Network error while canceling order order123: Network issue"):
            await service.cancel_order("order123", "BTC/USD")
        mock_exchange_instance.cancel_order.assert_awaited_once_with("order123", "BTC/USD")
    
    @patch("core.services.live_exchange_service.getattr")
    @patch("core.services.live_exchange_service.ccxt")
    def test_fetch_ohlcv_not_implemented(self, mock_ccxt, mock_getattr, setup_env_vars, config_manager):
        mock_exchange_instance = Mock()
        mock_ccxt.binance.return_value = mock_exchange_instance
        mock_getattr.return_value = mock_ccxt.binance

        service = LiveExchangeService(config_manager, is_paper_trading_activated=False)

        with pytest.raises(NotImplementedError, match="fetch_ohlcv is not used in live or paper trading mode."):
            service.fetch_ohlcv("BTC/USD", "1m", "start_date", "end_date")
    
    @patch("core.services.live_exchange_service.getattr")
    @patch("core.services.live_exchange_service.ccxt")
    @pytest.mark.asyncio
    async def test_get_exchange_status_ok(self, mock_ccxt, mock_getattr, setup_env_vars, config_manager):
        mock_exchange_instance = Mock()
        mock_ccxt.binance.return_value = mock_exchange_instance
        mock_getattr.return_value = mock_ccxt.binance

        service = LiveExchangeService(config_manager, is_paper_trading_activated=False)

        service.exchange.fetch_status = AsyncMock(return_value={
            "status": "ok",
            "updated": 1622505600000,
            "eta": None,
            "url": "https://status.exchange.com",
            "info": "All systems operational."
        })

        result = await service.get_exchange_status()

        assert result == {
            "status": "ok",
            "updated": 1622505600000,
            "eta": None,
            "url": "https://status.exchange.com",
            "info": "All systems operational."
        }

    @patch("core.services.live_exchange_service.getattr")
    @patch("core.services.live_exchange_service.ccxt")
    @pytest.mark.asyncio
    async def test_get_exchange_status_unsupported(self, mock_ccxt, mock_getattr, setup_env_vars, config_manager):
        mock_exchange_instance = Mock()
        mock_ccxt.binance.return_value = mock_exchange_instance
        mock_getattr.return_value = mock_ccxt.binance

        service = LiveExchangeService(config_manager, is_paper_trading_activated=False)
        service.exchange.fetch_status.side_effect = AttributeError

        result = await service.get_exchange_status()

        assert result == {
            "status": "unsupported",
            "info": "fetch_status not supported by this exchange."
        }
    
    @patch("core.services.live_exchange_service.getattr")
    @patch("core.services.live_exchange_service.ccxt")
    @pytest.mark.asyncio
    async def test_get_exchange_status_error(self, mock_ccxt, mock_getattr, setup_env_vars, config_manager):
        mock_exchange_instance = Mock()
        mock_ccxt.binance.return_value = mock_exchange_instance
        mock_getattr.return_value = mock_ccxt.binance

        service = LiveExchangeService(config_manager, is_paper_trading_activated=False)
        service.exchange.fetch_status.side_effect = Exception("Network error")

        result = await service.get_exchange_status()

        assert result["status"] == "error"
        assert "Failed to fetch exchange status" in result["info"]
        assert "Network error" in result["info"]

    @pytest.mark.asyncio
    @patch("core.services.live_exchange_service.ccxt")
    @patch("core.services.live_exchange_service.getattr")
    async def test_close_connection(self, mock_getattr, mock_ccxt, config_manager, setup_env_vars, mock_exchange_instance):
        mock_getattr.return_value = mock_ccxt.binance
        mock_ccxt.binance.return_value = mock_exchange_instance
        service = LiveExchangeService(config_manager, is_paper_trading_activated=False)

        service.connection_active = True

        await service.close_connection()
        assert service.connection_active is False
    
    @pytest.mark.asyncio
    @patch("core.services.live_exchange_service.ccxt")
    @patch("core.services.live_exchange_service.getattr")
    async def test_subscribe_to_ticker_updates_success(self, mock_getattr, mock_ccxt, config_manager, setup_env_vars, mock_exchange_instance):
        mock_getattr.return_value = mock_ccxt.binance
        mock_ccxt.binance.return_value = mock_exchange_instance
        mock_exchange_instance.watch_ticker = AsyncMock(side_effect=[
            {"last": 50000.0},  # First price update
            asyncio.CancelledError(),  # Stop the loop
        ])
        
        on_ticker_update = AsyncMock()
        service = LiveExchangeService(config_manager, is_paper_trading_activated=False)
        service.connection_active = True

        await service._subscribe_to_ticker_updates("BTC/USD", on_ticker_update, 0.1)

        on_ticker_update.assert_awaited_once_with(50000.0)
        assert not service.connection_active

    @pytest.mark.asyncio
    @patch("core.services.live_exchange_service.ccxt")
    @patch("core.services.live_exchange_service.getattr")
    async def test_subscribe_to_ticker_updates_network_error(self, mock_getattr, mock_ccxt, config_manager, setup_env_vars, mock_exchange_instance):
        mock_getattr.return_value = mock_ccxt.binance
        mock_ccxt.binance.return_value = mock_exchange_instance
        mock_exchange_instance.watch_ticker = AsyncMock(side_effect=ccxt.NetworkError("Network issue"))
        on_ticker_update = AsyncMock()
        service = LiveExchangeService(config_manager, is_paper_trading_activated=False)
        service.connection_active = True

        with patch.object(service.logger, "error") as mock_logger_error:
            await service._subscribe_to_ticker_updates("BTC/USD", on_ticker_update, 0.1)
            service.connection_active = False

            mock_logger_error.assert_any_call("Error connecting to WebSocket for BTC/USD: Network issue. Retrying in 5 seconds (1/5).")
            on_ticker_update.assert_not_awaited()

        assert not service.connection_active
    
    @pytest.mark.asyncio
    @patch("core.services.live_exchange_service.LiveExchangeService._subscribe_to_ticker_updates", new_callable=AsyncMock)
    async def test_listen_to_ticker_updates(self, mock_subscribe_to_ticker_updates, config_manager, setup_env_vars):
        service = LiveExchangeService(config_manager, is_paper_trading_activated=False)
        on_ticker_update = AsyncMock()

        await service.listen_to_ticker_updates("BTC/USD", on_ticker_update, 0.1)
        mock_subscribe_to_ticker_updates.assert_awaited_once_with("BTC/USD", on_ticker_update, 0.1)
    
    @pytest.mark.asyncio
    @patch("core.services.live_exchange_service.ccxt")
    @patch("core.services.live_exchange_service.getattr")
    async def test_get_balance_success(self, mock_getattr, mock_ccxt, config_manager, setup_env_vars, mock_exchange_instance):
        mock_getattr.return_value = mock_ccxt.binance
        mock_ccxt.binance.return_value = mock_exchange_instance
        mock_exchange_instance.fetch_balance = AsyncMock(return_value={"total": {"BTC": 1.0, "USD": 1000.0}})

        service = LiveExchangeService(config_manager, is_paper_trading_activated=False)
        result = await service.get_balance()

        assert result == {"total": {"BTC": 1.0, "USD": 1000.0}}
        mock_exchange_instance.fetch_balance.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("core.services.live_exchange_service.ccxt")
    @patch("core.services.live_exchange_service.getattr")
    async def test_get_balance_error(self, mock_getattr, mock_ccxt, config_manager, setup_env_vars, mock_exchange_instance):
        mock_getattr.return_value = mock_ccxt.binance
        mock_ccxt.binance.return_value = mock_exchange_instance
        mock_exchange_instance.fetch_balance = AsyncMock(side_effect=ccxt.BaseError("Balance error"))

        service = LiveExchangeService(config_manager, is_paper_trading_activated=False)

        with pytest.raises(DataFetchError, match="Error fetching balance: Balance error"):
            await service.get_balance()
        mock_exchange_instance.fetch_balance.assert_awaited_once()
    
    @pytest.mark.asyncio
    @patch("core.services.live_exchange_service.ccxt")
    @patch("core.services.live_exchange_service.getattr")
    async def test_fetch_order_success(self, mock_getattr, mock_ccxt, config_manager, setup_env_vars, mock_exchange_instance):
        mock_getattr.return_value = mock_ccxt.binance
        mock_ccxt.binance.return_value = mock_exchange_instance
        mock_exchange_instance.fetch_order = AsyncMock(return_value={"id": "123", "status": "open"})

        service = LiveExchangeService(config_manager, is_paper_trading_activated=False)
        result = await service.fetch_order("BTC/USD", "123")

        assert result == {"id": "123", "status": "open"}
        mock_exchange_instance.fetch_order.assert_awaited_once_with("123", "BTC/USD")

    @pytest.mark.asyncio
    @patch("core.services.live_exchange_service.ccxt")
    @patch("core.services.live_exchange_service.getattr")
    async def test_fetch_order_error(self, mock_getattr, mock_ccxt, config_manager, setup_env_vars, mock_exchange_instance):
        mock_getattr.return_value = mock_ccxt.binance
        mock_ccxt.binance.return_value = mock_exchange_instance
        mock_exchange_instance.fetch_order = AsyncMock(side_effect=ccxt.BaseError("Order error"))

        service = LiveExchangeService(config_manager, is_paper_trading_activated=False)

        with pytest.raises(DataFetchError, match="Exchange-specific error occurred: Order error"):
            await service.fetch_order("BTC/USD", "123")
        mock_exchange_instance.fetch_order.assert_awaited_once_with("123", "BTC/USD")