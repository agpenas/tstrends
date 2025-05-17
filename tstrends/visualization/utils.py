from matplotlib import pyplot as plt
import numpy as np
import matplotlib.colors as mcolors


def plot_trend_labels(
    time_series_list: list[float],
    labels: list[int],
    title: str = None,
    title_size: int = 12,
) -> None:
    """Simple visualization of the time series series with trend labels.

    Creates a matplotlib plot showing the time series series with colored backgrounds
    indicating the trend labels. Uptrends are shown in green, downtrends in brown.

    Args:
        time_series_list (list[float]): The time series series data points.
        labels (list[int]): Trend labels (-1 for downtrend, 1 for uptrend).
        title (str, optional): Title for the plot. Defaults to None.
        title_size (int, optional): Font size for the plot title. Defaults to 12.

    Example:
        >>> time series = [100.0, 101.0, 99.0, 98.0]
        >>> labels = [1, 1, -1, -1]
        >>> plot_trend_labels(time series, labels, "Time series Trends")

    Note:
        This function uses matplotlib's pyplot interface and will display
        the plot immediately using plt.show().
    """

    plt.figure(figsize=(10, 6))
    plt.plot(time_series_list, label="time series", color="black", linewidth=2)

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
        plt.title(title, fontsize=title_size)
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()


def plot_trend_labels_with_gradation(
    time_series_list: list[float],
    labels: list[float],
    title: str = None,
    title_size: int = 12,
    normalize: bool = True,
) -> None:
    """Visualization of time series with trend labels showing gradation based on label intensity.

    Creates a matplotlib plot showing the time series with colored backgrounds
    indicating trend labels. Uptrends are shown in green, downtrends in brown.
    The color intensity indicates the strength of the trend, with
    stronger trends having more saturated colors.

    Args:
        time_series_list (list[float]): The time series data points.
        labels (list[float]): Trend labels with values. Can be any range of values.
        title (str, optional): Title for the plot. Defaults to None.
        title_size (int, optional): Font size for the plot title. Defaults to 12.
        normalize (bool, optional): Whether to normalize labels to [-1, 1] range. Defaults to True.

    Example:
        >>> time_series = [100.0, 101.0, 99.0, 98.0]
        >>> labels = [0.5, 0.8, -0.3, -0.9]
        >>> plot_trend_labels_with_gradation(time_series, labels, "Time Series Trends with Gradation")

    Note:
        This function uses matplotlib's pyplot interface and will display
        the plot immediately using plt.show().
    """

    plt.figure(figsize=(10, 6))
    plt.plot(time_series_list, label="Time Series", color="black", linewidth=2)

    # Normalize labels to [-1, 1] range if requested
    if normalize:
        max_abs_value = max(abs(label) for label in labels)
        if max_abs_value > 0:  # Avoid division by zero
            normalized_labels = [label / max_abs_value for label in labels]
        else:
            normalized_labels = labels.copy()
    else:
        normalized_labels = labels.copy()

    # Create colormaps for uptrend and downtrend
    # Create custom colormaps for uptrend (white to green) and downtrend (white to brown)
    uptrend_cmap = mcolors.LinearSegmentedColormap.from_list(
        "uptrend", [(1, 1, 1, 0), (0.0, 0.5, 0.0, 1)]
    )
    downtrend_cmap = mcolors.LinearSegmentedColormap.from_list(
        "downtrend", [(1, 1, 1, 0), (0.70, 0.13, 0.13, 1)]
    )

    # Create empty plots for legend entries
    plt.fill_between([], [], color="darkgreen", label="Uptrend")
    plt.fill_between([], [], color="brown", label="Downtrend")

    # Highlight trends with linear gradation using colormaps
    for t in range(len(time_series_list)):
        label_value = normalized_labels[t]

        if label_value > 0:  # Uptrend
            intensity = abs(label_value)
            color = uptrend_cmap(intensity)
            plt.axvspan(t, t + 1, color=color)
        elif label_value < 0:  # Downtrend
            intensity = abs(label_value)
            color = downtrend_cmap(intensity)
            plt.axvspan(t, t + 1, color=color)

    plt.xlabel("Time")
    if title:
        plt.title(title, fontsize=title_size)
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
