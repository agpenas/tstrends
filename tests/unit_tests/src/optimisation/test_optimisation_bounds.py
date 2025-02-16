import pytest
from optimisation import OptimisationBounds
from trend_labelling import (
    BinaryCTL,
    TernaryCTL,
    OracleBinaryTrendLabeler,
    OracleTernaryTrendLabeler,
    BaseLabeller,
)


@pytest.fixture
def optimisation_bounds():
    """Fixture providing an OptimisationBounds instance."""
    return OptimisationBounds()


class TestOptimisationBounds:
    """Tests for the OptimisationBounds class."""

    def test_binary_ctl_bounds(self, optimisation_bounds):
        """Test bounds for BinaryCTL labeller."""
        bounds = optimisation_bounds.get_bounds(BinaryCTL)
        assert isinstance(bounds, dict)
        assert "omega" in bounds
        assert bounds["omega"] == (0.0, 1.0)

    def test_ternary_ctl_bounds(self, optimisation_bounds):
        """Test bounds for TernaryCTL labeller."""
        bounds = optimisation_bounds.get_bounds(TernaryCTL)
        assert isinstance(bounds, dict)
        assert "marginal_change_thres" in bounds
        assert "window_size" in bounds
        assert bounds["marginal_change_thres"] == (0.0, 0.1)
        assert bounds["window_size"] == (1, 5_000)

    def test_oracle_binary_bounds(self, optimisation_bounds):
        """Test bounds for OracleBinaryTrendLabeler."""
        bounds = optimisation_bounds.get_bounds(OracleBinaryTrendLabeler)
        assert isinstance(bounds, dict)
        assert "transaction_cost" in bounds
        assert bounds["transaction_cost"] == (0.0, 1.0)

    def test_oracle_ternary_bounds(self, optimisation_bounds):
        """Test bounds for OracleTernaryTrendLabeler."""
        bounds = optimisation_bounds.get_bounds(OracleTernaryTrendLabeler)
        assert isinstance(bounds, dict)
        assert "transaction_cost" in bounds
        assert "trend_coeff" in bounds
        assert bounds["transaction_cost"] == (0.0, 1.0)
        assert bounds["trend_coeff"] == (0.0, 1.0)

    def test_unimplemented_labeller(self, optimisation_bounds):
        """Test handling of unimplemented labeller class."""

        class UnimplementedLabeller(BaseLabeller):
            pass

        with pytest.raises(
            ValueError, match="Default bounds not implemented for labeller class"
        ):
            optimisation_bounds.get_bounds(UnimplementedLabeller)

    @pytest.mark.parametrize(
        "labeller_class,expected_params",
        [
            (BinaryCTL, ["omega"]),
            (TernaryCTL, ["marginal_change_thres", "window_size"]),
            (OracleBinaryTrendLabeler, ["transaction_cost"]),
            (OracleTernaryTrendLabeler, ["transaction_cost", "trend_coeff"]),
        ],
        ids=[
            "binary_ctl",
            "ternary_ctl",
            "oracle_binary",
            "oracle_ternary",
        ],
    )
    def test_bounds_parameters(
        self, optimisation_bounds, labeller_class, expected_params
    ):
        """Test that each labeller class returns the expected parameter bounds."""
        bounds = optimisation_bounds.get_bounds(labeller_class)
        assert set(bounds.keys()) == set(expected_params)
        for param in expected_params:
            assert isinstance(bounds[param], tuple)
            assert len(bounds[param]) == 2
            assert bounds[param][0] <= bounds[param][1]
