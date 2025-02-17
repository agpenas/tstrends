# Python Trend Labeller
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="center"><img src="images/example_labelling2.png" width="300" height="200"/></p>

## Introduction

A robust Python package for automated trend labelling in time series data with a strong financial flavour, implementing SOTA trend labelling algorithms ([bibliography](#bibliography)) with returns estimation and parameter bayesian optimization capabilities. Main capabilities:
- Two-state (upwards/downwards) and three-state (upwards/**neutral**/downwards) trend labelling algorithms.
- Returns estimation with transaction costs and holding fees
- Bayesian parameter optimization to select the optimal labelling (powered by [bayesian-optimization](https://github.com/bayesian-optimization/BayesianOptimization))




## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Components](#core-components)
  - [Trend Labellers](#trend-labellers)
  - [Returns Estimation](#returns-estimation)
  - [Parameter Optimization](#parameter-optimization)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [Bibliography](#bibliography)
- [License](#license)

## Overview

The Python Trend Labeller package provides a comprehensive suite of tools for automated trend labelling in time series data, with a strong finantial flavour (e.g. transaction costs, holding fees, prices etc. *although the tools are useful to any time series data*). It implements sophisticated algorithms for detecting and classifying movements into distinct trend states, coupled with returns estimation and parameter optimization capabilities. This makes it a powerful tool for time series analysis, algorithmic trading strategy development, and research applications.

## Features

### Trend Labelling Approaches
- **Continuous Trend Labelling (CTL)**:
  - Binary CTL (Up/Down trends) - Based on [Wu et al.](#bibliography)
  - Ternary CTL (Up/Neutral/Down trends) - Inspired by [Dezhkam et al.](#bibliography)
- **Oracle Labelling**
  - Binary Oracle (optimizes for maximum returns) - Based on [Kovačević et al.](#bibliography)
  - Ternary Oracle (includes neutral state optimization) 

### Returns Estimation
- Simple returns calculation
- Transaction costs consideration
- Holding fees support
- Position-specific fee structures

### Parameter Optimization
- Bayesian optimization for parameter tuning
- Support for multiple time series optimization
- Customizable acquisition functions
- Empirically tested and customizable parameter bounds


## Installation

```bash
pip install python-trend-labeller
```

## Quick Start

```python
from trend_labelling import BinaryCTL, TernaryCTL, OracleBinaryTrendLabeller
from returns_estimation import SimpleReturnEstimator
from optimization import Optimizer

# Sample price data
prices = [100.0, 102.0, 105.0, 103.0, 98.0, 97.0, 99.0, 102.0, 104.0]

# 1. Basic naïve CTL Labelling
binary_labeller = BinaryCTL(omega=0.02)
binary_labels = binary_labeller.get_labels(prices)

# 2. Returns Estimation
estimator = SimpleReturnEstimator()
returns = estimator.estimate_return(prices, binary_labels)

# 3. Parameter Optimization
optimizer = Optimizer(
    returns_estimator=SimpleReturnEstimator,
    initial_points=5,
    nb_iter=100
)
optimal_params = optimizer.optimize(BinaryCTL, prices)
print(f"Optimal parameters: {optimal_params['params']}")
```

## Core Components

### Trend Labellers

#### 1. Continuous Trend Labelling (CTL)
- **BinaryCTL**: Implements the Wu et al. algorithm for binary trend labelling
  - Parameters:
    - `omega`: Threshold for trend changes (float). Key to manage the sensitivity of the labeller.\
    **For instance, for a value of 0.001, 0.005, 0.01, 0.015, the labeller behaves as follows:**
    
      <img src="images/binary_ctl_omega_effect.png" alt="Omega effect on binary CTL" width="800"/>

  
- **TernaryCTL**: Extends CTL with a neutral state.
  - Parameters:
    - `marginal_change_thres`: Threshold for significant time series movements as a percentage of the current value.
    - `window_size`: Maximum window to look for trend confirmation before resetting state to neutral
    **For instance, for different combinations of `marginal_change_thres` and `window_size`, the labeller behaves as follows:**

    <p align="center"><img src="images/ternaryCTL_params_effect.png" alt="Ternary CTL parameters effect" width="800"/></p>

#### 2. Oracle Labellers
- **OracleBinaryTrendLabeller**: Optimizes labels for maximum returns
  - Parameters:
    - `transaction_cost`: Cost coefficient for position changes

- **OracleTernaryTrendLabeller**: Includes neutral state in optimization
  - Parameters:
    - `transaction_cost`: Cost coefficient for position changes
    - `neutral_reward_factor`: Coefficient for neutral state rewards

### Returns Estimation

The package provides flexible returns estimation with transaction costs:

```python
from returns_estimation import ReturnsEstimatorWithFees, FeesConfig

# Configure fees
fees_config = FeesConfig(
    lp_transaction_fees=0.001,  # 0.1% fee for long positions
    sp_transaction_fees=0.001,  # 0.1% fee for short positions
    lp_holding_fees=0.0001,    # 0.01% daily fee for long positions
    sp_holding_fees=0.0001     # 0.01% daily fee for short positions
)

# Create estimator with fees
estimator = ReturnsEstimatorWithFees(fees_config)

# Calculate returns with fees
returns = estimator.estimate_return(prices, labels)
```

### Parameter Optimization

The package uses Bayesian optimization to find optimal parameters:

```python
from optimization import Optimizer, OptimizationBounds

# Create optimizer
optimizer = Optimizer(
    returns_estimator=ReturnsEstimatorWithFees,
    initial_points=10,
    nb_iter=1000
)

# Custom bounds (optional)
bounds = {
    'transaction_cost': (0.0, 0.01),
    'neutral_reward_factor': (0.0, 0.1)
}

# Optimize parameters
result = optimizer.optimize(
    labeller_class=OracleTernaryTrendLabeller,
    time_series_list=prices,
    bounds=bounds
)

print(f"Optimal parameters: {result['params']}")
print(f"Maximum return: {result['target']}")
```

## API Reference

### Trend Labellers

```python
class BinaryCTL(BaseLabeller):
    """Binary Continuous Trend Labeller.
    
    Args:
        omega (float): Threshold for trend changes
    """

class TernaryCTL(BaseLabeller):
    """Ternary Continuous Trend Labeller.
    
    Args:
        marginal_change_thres (float): Threshold for significant movements
        window_size (int): Maximum window for trend confirmation
    """

class OracleBinaryTrendLabeller(BaseLabeller):
    """Binary Oracle Trend Labeller.
    
    Args:
        transaction_cost (float): Cost coefficient for position changes
    """

class OracleTernaryTrendLabeller(BaseLabeller):
    """Ternary Oracle Trend Labeller.
    
    Args:
        transaction_cost (float): Cost coefficient for position changes
        neutral_reward_factor (float): Coefficient for neutral state rewards
    """
```

### Returns Estimation

```python
class SimpleReturnEstimator(BaseReturnEstimator):
    """Basic returns calculator without fees."""

class ReturnsEstimatorWithFees(SimpleReturnEstimator):
    """Returns calculator with transaction and holding fees.
    
    Args:
        fees_config (FeesConfig): Configuration for transaction and holding fees
    """

@dataclass
class FeesConfig:
    """Configuration for transaction and holding fees.
    
    Args:
        lp_transaction_fees (float): Long position transaction fees
        sp_transaction_fees (float): Short position transaction fees
        lp_holding_fees (float): Long position holding fees
        sp_holding_fees (float): Short position holding fees
    """
```

### Optimization

```python
class Optimizer:
    """Bayesian optimization for trend labelling parameters.
    
    Args:
        returns_estimator (Type[BaseReturnEstimator]): Returns estimator class
        initial_points (int): Number of initial random points
        nb_iter (int): Number of optimization iterations
        random_state (int | None): Random seed for reproducibility
    """
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate and adhere to the existing coding style.

## Bibliography

The algorithms implemented in this package are based or inspired by the following academic papers:

1. Wu, D., Wang, X., Su, J., Tang, B., & Wu, S. "A Labeling Method for Financial Time Series Prediction Based on Trends". *Referenced in BinaryCTL implementation.*

2. Dezhkam, A., Moazeni, S., & Moazeni, S. "A Bayesian-based classification framework for financial time series trend prediction". *Referenced in TernaryCTL implementation.*

3. Kovačević, T., Merćep, A., Begušić, S., & Kostanjčar, Z. "Optimal Trend Labeling in Financial Time Series". *Referenced in Oracle Labellers implementation.*

## License

[MIT License](LICENSE)