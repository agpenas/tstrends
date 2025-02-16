import pytest
from returns_estimation.fees_config import FeesConfig


@pytest.fixture
def default_fees_config():
    """Fixture providing a default FeesConfig instance with zero fees."""
    return FeesConfig()


@pytest.fixture
def sample_fees_config():
    """Fixture providing a FeesConfig instance with sample non-zero fees."""
    return FeesConfig(
        lp_transaction_fees=0.001,  # 0.1% fee
        sp_transaction_fees=0.002,  # 0.2% fee
        lp_holding_fees=0.0005,  # 0.05% fee
        sp_holding_fees=0.0008,  # 0.08% fee
    )


class TestFeesConfig:
    def test_default_initialization(self, default_fees_config):
        """Test that FeesConfig initializes with default zero values."""
        assert default_fees_config.lp_transaction_fees == 0.0
        assert default_fees_config.sp_transaction_fees == 0.0
        assert default_fees_config.lp_holding_fees == 0.0
        assert default_fees_config.sp_holding_fees == 0.0

    def test_custom_initialization(self, sample_fees_config):
        """Test initialization with custom fee values."""
        assert sample_fees_config.lp_transaction_fees == 0.001
        assert sample_fees_config.sp_transaction_fees == 0.002
        assert sample_fees_config.lp_holding_fees == 0.0005
        assert sample_fees_config.sp_holding_fees == 0.0008

    @pytest.mark.parametrize(
        "invalid_params,error_type,error_match",
        [
            (
                {"lp_transaction_fees": "0.001"},
                ValueError,
                "lp_transaction_fees must be float or int",
            ),
            (
                {"sp_transaction_fees": -0.001},
                ValueError,
                "sp_transaction_fees must be non-negative",
            ),
            (
                {"lp_holding_fees": [0.001]},
                ValueError,
                "lp_holding_fees must be float or int",
            ),
            (
                {"sp_holding_fees": -0.0008},
                ValueError,
                "sp_holding_fees must be non-negative",
            ),
        ],
        ids=[
            "invalid_type_transaction_fees",
            "negative_transaction_fees",
            "invalid_type_holding_fees",
            "negative_holding_fees",
        ],
    )
    def test_invalid_initialization(self, invalid_params, error_type, error_match):
        """Test initialization with invalid parameters."""
        with pytest.raises(error_type, match=error_match):
            FeesConfig(**invalid_params)

    def test_immutability(self, sample_fees_config):
        """Test that FeesConfig is immutable (frozen)."""
        with pytest.raises(AttributeError):
            sample_fees_config.lp_transaction_fees = 0.002

    def test_valid_fee_ranges(self):
        """Test initialization with various valid fee ranges."""
        # Test with integer values
        config = FeesConfig(
            lp_transaction_fees=1,
            sp_transaction_fees=2,
            lp_holding_fees=1,
            sp_holding_fees=2,
        )
        assert isinstance(config.lp_transaction_fees, float)
        assert isinstance(config.sp_transaction_fees, float)

        # Test with zero values
        config = FeesConfig(
            lp_transaction_fees=0,
            sp_transaction_fees=0,
            lp_holding_fees=0,
            sp_holding_fees=0,
        )
        assert all(
            fee == 0.0
            for fee in [
                config.lp_transaction_fees,
                config.sp_transaction_fees,
                config.lp_holding_fees,
                config.sp_holding_fees,
            ]
        )
