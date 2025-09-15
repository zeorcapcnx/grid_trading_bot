from enum import Enum


class RiskManagementMode(Enum):
    TAKE_PROFIT_STOP_LOSS = "take_profit_stop_loss"
    DYNAMIC = "dynamic"

    @staticmethod
    def from_string(mode_str: str):
        try:
            return RiskManagementMode(mode_str)
        except ValueError:
            available_modes = ", ".join([mode.value for mode in RiskManagementMode])
            raise ValueError(
                f"Invalid risk management mode: '{mode_str}'. Available modes are: {available_modes}",
            ) from None