from matplotlib import pyplot as plt


def plot_trend_labels(
    time_series_list: list[str], labels: list[int], title: str = None
) -> None:
    """Simple visualization of the price series with trend labels.

    Creates a matplotlib plot showing the price series with colored backgrounds
    indicating the trend labels. Uptrends are shown in green, downtrends in brown.

    Args:
        time_series_list (list[str]): The price series data points.
        labels (list[int]): Trend labels (-1 for downtrend, 1 for uptrend).
        title (str, optional): Title for the plot. Defaults to None.

    Example:
        >>> prices = [100.0, 101.0, 99.0, 98.0]
        >>> labels = [1, 1, -1, -1]
        >>> plot_trend_labels(prices, labels, "Price Trends")

    Note:
        This function uses matplotlib's pyplot interface and will display
        the plot immediately using plt.show().
    """

    plt.figure(figsize=(10, 6))
    plt.plot(time_series_list, label="Price", color="black", linewidth=2)

    # Create empty plots for legend entries
    plt.fill_between([], [], color="darkgreen", label="Uptrend")
    plt.fill_between([], [], color="brown", label="Downtrend")

    # Highlight trends
    for t in range(len(time_series_list)):
        if labels[t] == 1:  # Uptrend
            plt.axvspan(
                t,
                t + 1,
                color="darkgreen",
                alpha=1,
            )
        elif labels[t] == -1:  # Downtrend
            plt.axvspan(
                t,
                t + 1,
                color="brown",
                alpha=1,
            )

    plt.xlabel("Time")
    if title:
        plt.title(title)
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
