from enum import Enum


class RangeMode(Enum):
    MANUAL = "manual"
    CRYPTO_ZERO = "crypto_zero"

    @staticmethod
    def from_string(range_mode_str: str):
        try:
            return RangeMode(range_mode_str)
        except ValueError:
            available_modes = ", ".join([mode.value for mode in RangeMode])
            raise ValueError(
                f"Invalid range mode: '{range_mode_str}'. Available modes are: {available_modes}",
            ) from None