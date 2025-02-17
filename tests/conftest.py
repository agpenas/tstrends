import pytest
from pathlib import Path
import csv


@pytest.fixture(scope="session")
def sample_prices():
    """
    Fixture providing price data for testing.
    Loads data once per test session to optimize performance.

    Returns:
        list[float]: List of closing prices
    """
    data_path = Path(__file__).parent / "data" / "closing_prices.csv"

    with open(data_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header if present
        prices = [float(row[0]) for row in reader]  # Assumes price is in first column

    return prices
