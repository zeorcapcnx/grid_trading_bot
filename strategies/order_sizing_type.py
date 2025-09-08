from enum import Enum


class OrderSizingType(Enum):
    EQUAL_CRYPTO = "equal_crypto"
    EQUAL_DOLLAR = "equal_dollar"

    @staticmethod
    def from_string(order_sizing_type_str: str):
        try:
            return OrderSizingType(order_sizing_type_str)
        except ValueError:
            available_types = ", ".join([sizing.value for sizing in OrderSizingType])
            raise ValueError(
                f"Invalid order sizing type: '{order_sizing_type_str}'. Available types are: {available_types}",
            ) from None