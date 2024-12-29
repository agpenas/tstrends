from typing import List
from matplotlib import pyplot as plt


def plot_trend_labels_matplotly(time_series_list: List[str], labels: List[int]) -> None:
    """
    Visualize the price series with trend labels.

    Args:
        time_series_list (List[str]): The price series.
        labels (List[int]): Optimal trend labels (0 for downtrend, 1 for uptrend).
    """

    plt.figure(figsize=(10, 6))
    plt.plot(time_series_list, label="Price", color="blue", linewidth=2)

    # Highlight trends
    for t in range(len(time_series_list) - 1):
        if labels[t] == 1:  # Uptrend
            plt.axvspan(
                t,
                t + 1,
                color="green",
                alpha=0.3,
                label="Uptrend" if t == 0 else "",
            )
        elif labels[t] == 0:  # Downtrend
            plt.axvspan(
                t,
                t + 1,
                color="red",
                alpha=0.3,
                label="Downtrend" if t == 0 else "",
            )

    plt.xlabel("Time")
    plt.title("Trend Labeling Visualization")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
