import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.grid_management.grid_manager import GridManager
from core.order_handling.order import Order, OrderSide
from core.order_handling.order_book import OrderBook


class Plotter:
    def __init__(
        self,
        grid_manager: GridManager,
        order_book: OrderBook,
    ):
        self.grid_manager = grid_manager
        self.order_book = order_book

    def plot_results(
        self,
        data: pd.DataFrame,
    ) -> None:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.70, 0.15, 0.15], vertical_spacing=0.02)
        self._add_candlestick_trace(fig, data)
        trigger_price = self.grid_manager.get_trigger_price()
        self._add_trigger_price_line(fig, trigger_price)
        self._add_grid_lines(fig, self.grid_manager.price_grids, self.grid_manager.central_price)
        self._add_trade_markers(fig, self.order_book.get_completed_orders())
        self._add_cumulative_profit_trace(fig, data)
        self._add_account_value_trace(fig, data)

        fig.update_layout(
            title="Grid Trading Strategy Results",
            yaxis_title="Price (USDT)",
            yaxis2_title="Cash (USDT)",
            yaxis3_title="Equity",
            xaxis={"rangeslider": {"visible": False}},
            showlegend=False,
        )
        fig.show()

    def _add_candlestick_trace(
        self,
        fig: go.Figure,
        data: pd.DataFrame,
    ) -> None:
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["open"],
                high=data["high"],
                low=data["low"],
                close=data["close"],
                name="",
            ),
            row=1,
            col=1,
        )

    def _add_trigger_price_line(
        self,
        fig: go.Figure,
        trigger_price: float,
    ):
        fig.add_trace(
            go.Scatter(
                x=[fig.data[0].x[0], fig.data[0].x[-1]],
                y=[trigger_price, trigger_price],
                mode="lines",
                line={"color": "blue", "width": 2, "dash": "dash"},
                name="Central Price",
            ),
        )
        fig.add_annotation(
            x=fig.data[0].x[-1],
            y=trigger_price,
            text=f"Trigger Price: {trigger_price:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowcolor="blue",
            ax=20,
            ay=-20,
            font={"size": 10, "color": "blue"},
        )

    def _add_grid_lines(
        self,
        fig: go.Figure,
        grids: list[float],
        central_price: float,
    ) -> None:
        for price in grids:
            color = "green" if price < central_price else "red"
            fig.add_trace(
                go.Scatter(
                    x=[fig.data[0].x[0], fig.data[0].x[-1]],
                    y=[price, price],
                    mode="lines",
                    line={"color": color, "dash": "dash"},
                    showlegend=False,
                ),
            )

    def _add_trade_markers(
        self,
        fig: go.Figure,
        orders: list[Order],
    ) -> None:
        for order in orders:
            icon_name = "triangle-up" if order.side == OrderSide.BUY else "triangle-down"
            icon_color = "green" if order.side == OrderSide.BUY else "red"
            fig.add_trace(
                go.Scatter(
                    x=[order.format_last_trade_timestamp()],
                    y=[order.price],
                    mode="markers",
                    marker={
                        "symbol": icon_name,
                        "color": icon_color,
                        "size": 12,
                        "line": {"color": "black", "width": 2},
                    },
                    name=f"{order.side.name} Order",
                    text=f"Price: {order.price}\nQty: {order.filled}\nDate: {order.format_last_trade_timestamp()}",
                    hoverinfo="x+y+text",
                ),
                row=1,
                col=1,
            )

    def _add_cumulative_profit_trace(
        self,
        fig: go.Figure,
        data: pd.DataFrame,
    ) -> None:
        # Check if cumulative_profit column exists, if not initialize with zeros
        if "cumulative_profit" not in data.columns:
            data["cumulative_profit"] = 0.0

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["cumulative_profit"],
                mode="lines",
                name="",
                line={"color": "orange", "width": 2},
                fill="tozeroy",
                fillcolor="rgba(255, 165, 0, 0.3)",
            ),
            row=2,
            col=1,
        )

        fig.update_yaxes(
            title="Cash (USDT)",
            row=2,
            col=1,
        )

    def _add_account_value_trace(
        self,
        fig: go.Figure,
        data: pd.DataFrame,
    ) -> None:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["account_value"],
                mode="lines",
                name="",
                line={"color": "purple", "width": 2},
            ),
            row=3,
            col=1,
        )
