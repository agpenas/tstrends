Bibliography
============

Academic References
----------------------------

The algorithms and methodologies implemented in TStrends are based on or inspired by the following academic research. These papers provide the theoretical foundation for our trend labelling approaches.

Continuous Trend Labelling (CTL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**[1] Wu, D., Wang, X., Su, J., Tang, B., & Wu, S. (2020). "A Labeling Method for Financial Time Series Prediction Based on Trends". Entropy, 22(10), 1162. https://doi.org/10.3390/e22101162**

   This research introduces a labelling method for financial time series that identifies upward and downward trends based on price movements. The algorithm, which we've implemented as ``BinaryCTL``, labels segments as rising when the market rises above a certain proportion (omega) from the current lowest point or as falling when it recedes from the current highest point to a certain proportion.

Ternary Trend Labelling
~~~~~~~~~~~~~~~~~~~~~~~

**[2] Dezhkam, A., Manzuri, M. T., Aghapour, A., Karimi, A., Rabiee, A., & Shalmani, S. M. (2023). "A Bayesian-based classification framework for financial time series trend prediction". The Journal of supercomputing, 79(4), 4622–4659. https://doi.org/10.1007/s11227-022-04834-4**

   This paper presents a Bayesian-based approach for financial time series classification into upward, neutral, and downward trends. Our ``TernaryCTL`` implementation is inspired by this work, extending the binary approach to include a neutral state and introducing a window size parameter to look for trend confirmation.

Oracle Labelling
~~~~~~~~~~~~~~~~~~~~~~~

**[3] Kovačević, T., Merćep, A., Begušić, S., & Kostanjčar, Z. (2023). "Optimal Trend Labeling in Financial Time Series". IEEE Access. PP. 1-1. 10.1109/ACCESS.2023.3303283.**

   This research proposes an optimal trend labelling approach that maximizes returns while considering transaction costs. Our ``OracleBinaryTrendLabeller`` is a direct implementation of this algorithm, using dynamic programming to efficiently compute optimal labels. We've also extended this approach with our ``OracleTernaryTrendLabeller`` to include a neutral state in the optimization process.

