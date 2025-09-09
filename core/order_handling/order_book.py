from ..grid_management.grid_level import GridLevel
from .order import Order, OrderSide, OrderStatus


class OrderBook:
    def __init__(self):
        self.buy_orders: list[Order] = []
        self.sell_orders: list[Order] = []
        self.non_grid_orders: list[Order] = []  # Orders that are not linked to any grid level
        self.order_to_grid_map: dict[Order, GridLevel] = {}  # Mapping of Order -> GridLevel

    def add_order(
        self,
        order: Order,
        grid_level: GridLevel | None = None,
    ) -> None:
        if order.side == OrderSide.BUY:
            self.buy_orders.append(order)
        else:
            self.sell_orders.append(order)

        if grid_level:
            self.order_to_grid_map[order] = grid_level  # Store the grid level associated with this order
        else:
            self.non_grid_orders.append(order)  # This is a non-grid order like take profit or stop loss

    def get_buy_orders_with_grid(self) -> list[tuple[Order, GridLevel | None]]:
        return [(order, self.order_to_grid_map.get(order, None)) for order in self.buy_orders]

    def get_sell_orders_with_grid(self) -> list[tuple[Order, GridLevel | None]]:
        return [(order, self.order_to_grid_map.get(order, None)) for order in self.sell_orders]

    def get_all_buy_orders(self) -> list[Order]:
        return self.buy_orders

    def get_all_sell_orders(self) -> list[Order]:
        return self.sell_orders

    def get_open_orders(self) -> list[Order]:
        return [order for order in self.buy_orders + self.sell_orders if order.is_open()]

    def get_completed_orders(self) -> list[Order]:
        return [order for order in self.buy_orders + self.sell_orders if order.is_filled()]

    def get_grid_level_for_order(self, order: Order) -> GridLevel | None:
        # Try to find grid level by order object first, then by order ID as fallback
        grid_level = self.order_to_grid_map.get(order)
        if grid_level:
            return grid_level
        
        # Fallback: search by order ID in case order object changed
        for stored_order, stored_grid_level in self.order_to_grid_map.items():
            if stored_order.identifier == order.identifier:
                return stored_grid_level
        
        return None

    def update_order_status(
        self,
        order_id: str,
        new_status: OrderStatus,
    ) -> None:
        for order in self.buy_orders + self.sell_orders:
            if order.identifier == order_id:
                order.status = new_status
                break
