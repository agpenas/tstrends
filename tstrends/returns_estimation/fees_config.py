from dataclasses import dataclass


@dataclass(frozen=True)
class FeesConfig:
    """Configuration for transaction and holding fees.

    This class provides a structured way to configure various types of fees that impact
    return calculations. It includes both transaction fees (applied when positions change)
    and holding fees (applied for maintaining positions).

    Attributes:
        lp_transaction_fees (float): Transaction fee percentage for long positions
        sp_transaction_fees (float): Transaction fee percentage for short positions
        lp_holding_fees (float): Daily absolute holding fee for long positions
        sp_holding_fees (float): Daily absolute holding fee for short positions

    Example:
        >>> fees_config = FeesConfig(
        ...     lp_transaction_fees=0.001,  # 0.1% fee for long position transactions
        ...     sp_transaction_fees=0.002,  # 0.2% fee for short position transactions
        ...     lp_holding_fees=0.0005,    # 0.05% daily fee for holding long positions
        ...     sp_holding_fees=0.0008     # 0.08% daily fee for holding short positions
        ... )
    """

    lp_transaction_fees: float = 0.0
    sp_transaction_fees: float = 0.0
    lp_holding_fees: float = 0.0
    sp_holding_fees: float = 0.0

    def __post_init__(self):
        """Validate the fees configuration and convert integers to floats."""
        # First validate types
        for field, value in self.__dict__.items():
            if not isinstance(value, (float, int)):
                raise ValueError(f"{field} must be float or int, got {type(value)}")
            if value < 0:
                raise ValueError(f"{field} must be non-negative, got {value}")

        # Then convert to float if needed
        for field, value in self.__dict__.items():
            if isinstance(value, int):
                object.__setattr__(self, field, float(value))
