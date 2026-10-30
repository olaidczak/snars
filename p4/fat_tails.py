"""Fat-tailed degree distribution analysis (task 4.0)."""

import numpy as np
import matplotlib.pyplot as plt


def get_degrees(G):
    return np.array([d for _, d in G.degree()])


def plot_histogram(G, ax=None):
    """Degree histogram."""
    degrees = get_degrees(G)
    ax = ax or plt.gca()
    ax.hist(degrees, bins=range(max(degrees) + 2))
    ax.set_xlabel("Degree")
    ax.set_ylabel("Count")
    ax.set_title("Degree distribution")
    return ax


def plot_loglog_histogram(G, ax=None):
    """Degree histogram on double log axes."""
    ax = plot_histogram(G, ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Degree distribution log-log")
    return ax


def log_binned_histogram(G, n_bins=15):
    """P(k) with logarithmic bins. Returns bin centers and P(k)."""
    degrees = get_degrees(G)
    degrees = degrees[degrees > 0]
    bins = np.logspace(0, np.log10(max(degrees) + 1), n_bins)
    hist, bin_edges = np.histogram(degrees, bins=bins, density=True)
    centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    return centers, hist


def plot_log_binned_histogram(G, n_bins=15, ax=None):
    degrees = get_degrees(G)
    degrees = degrees[degrees > 0]
    bins = np.logspace(0, np.log10(max(degrees) + 1), n_bins)
    ax = ax or plt.gca()
    ax.hist(degrees, bins=bins, density=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree")
    ax.set_ylabel("P(k)")
    ax.set_title("Logarithmic binning")
    return ax


def survival_function(G):
    """S(k) = P(K >= k). Returns degrees k and S(k)."""
    degrees = get_degrees(G)
    k = np.unique(degrees)
    survival = np.array([np.mean(degrees >= x) for x in k])
    return k, survival


def plot_survival_function(G, ax=None):
    k, survival = survival_function(G)
    ax = ax or plt.gca()
    ax.plot(k, survival, marker=".", linestyle="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree")
    ax.set_ylabel("S(k)")
    ax.set_title("Survival function")
    return ax


def fit_loglog_line(x, y, x_min=1):
    """Least squares line on log-log axes for points with x >= x_min. Returns slope, intercept."""
    x, y = np.asarray(x), np.asarray(y)
    mask = (x >= x_min) & (y > 0)
    slope, intercept = np.polyfit(np.log10(x[mask]), np.log10(y[mask]), 1)
    return slope, intercept


def alpha_regression(G, x_min=10, n_bins=15):
    """alpha estimated by linear regression on the three log-log plots."""
    values, counts = np.unique(get_degrees(G), return_counts=True)
    centers, hist = log_binned_histogram(G, n_bins)
    k, survival = survival_function(G)
    return {
        "loglog": -fit_loglog_line(values, counts, x_min)[0],
        "log_binning": -fit_loglog_line(centers, hist, x_min)[0],
        "survival": 1 - fit_loglog_line(k, survival, x_min)[0],
    }


def plot_alpha_regression(G, x_min=10, n_bins=15):
    """Exercise 6: the three plots with fitted lines."""
    values, counts = np.unique(get_degrees(G), return_counts=True)
    centers, hist = log_binned_histogram(G, n_bins)
    k, survival = survival_function(G)
    plots = [
        ("Log-log", values, counts),
        ("Logarithmic binning", centers, hist),
        ("Survival function", k, survival),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (title, x, y) in zip(axes, plots):
        slope, intercept = fit_loglog_line(x, y, x_min)
        alpha = 1 - slope if title == "Survival function" else -slope
        tail = x[(x >= x_min) & (y > 0)]

        ax.plot(x[y > 0], y[y > 0], marker=".", linestyle="none")
        ax.plot(tail, 10**intercept * tail**slope, color="red")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Degree")
        ax.set_title(f"{title}, alpha = {alpha:.2f}")
    plt.tight_layout()
    return fig


def mle_alpha(G, x_min):
    """Exercise 7: MLE of alpha for a known x_min."""
    x = get_degrees(G)
    x = x[x >= x_min]
    return 1 + len(x) / np.sum(np.log(x / x_min))


def ks_distance(G, x_min, alpha):
    """Max distance between the empirical and fitted survival functions for x >= x_min."""
    x = get_degrees(G)
    x = np.sort(x[x >= x_min])
    empirical = 1 - np.arange(len(x)) / len(x)
    fitted = (x / x_min) ** (1 - alpha)
    return np.max(np.abs(empirical - fitted))


def find_x_min(G, min_tail=100):
    """Exercise 8: choose x_min minimizing the KS distance (Clauset, Shalizi, Newman 2009).

    Only x_min with at least min_tail nodes in the tail are considered.
    Returns a dict with the best x_min, its alpha, and all candidates for plotting.
    """
    degrees = get_degrees(G)
    candidates = [x for x in np.unique(degrees) if x > 0 and np.sum(degrees >= x) >= min_tail]
    alphas = [mle_alpha(G, x) for x in candidates]
    distances = [ks_distance(G, x, a) for x, a in zip(candidates, alphas)]
    best = int(np.argmin(distances))
    return {
        "x_min": candidates[best],
        "alpha": alphas[best],
        "n_tail": int(np.sum(degrees >= candidates[best])),
        "candidates": candidates,
        "alphas": alphas,
        "distances": distances,
    }


def plot_x_min_scan(G, min_tail=100):
    """Exercise 8: alpha and KS distance as functions of x_min."""
    result = find_x_min(G, min_tail)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, label in [(axes[0], "alphas", "alpha"), (axes[1], "distances", "KS distance")]:
        ax.plot(result["candidates"], result[key])
        ax.axvline(result["x_min"], color="red", linestyle="--")
        ax.set_xlabel("x_min")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs x_min")
    plt.tight_layout()
    return fig


def analyze(G):
    """Run the whole task 4.0 analysis on graph G and print the alpha estimates."""
    _, axes = plt.subplots(1, 4, figsize=(24, 5))
    plot_histogram(G, axes[0])
    plot_loglog_histogram(G, axes[1])
    plot_log_binned_histogram(G, ax=axes[2])
    plot_survival_function(G, axes[3])
    plt.tight_layout()
    plt.show()

    for method, alpha in alpha_regression(G).items():
        print(f"regression ({method}): alpha = {alpha:.2f}")
    plot_alpha_regression(G)
    plt.show()

    result = find_x_min(G)
    print(f"MLE with x_min = 1: alpha = {mle_alpha(G, 1):.2f}")
    print(f"MLE with best x_min = {result['x_min']}: alpha = {result['alpha']:.2f} ({result['n_tail']} nodes in tail)")
    plot_x_min_scan(G)
    plt.show()
