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
    """three plots with fitted lines."""
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
    """MLE of alpha for a known x_min."""
    x = get_degrees(G)
    x = x[x >= x_min]
    return 1 + len(x) / np.sum(np.log(x / x_min))


def ks_distance(G, x_min):
    """Kolmogorov-Smirnov distance between the data (degrees >= x_min)
    and the power law fitted with MLE for this x_min."""
    x = get_degrees(G)
    x = x[x >= x_min]
    alpha = mle_alpha(G, x_min)
    k = np.unique(x)

    D = 0
    for i in range(len(k)):
        data_S = np.mean(x >= k[i])
        model_S = (k[i] / x_min) ** (1 - alpha)
        D = max(D, abs(data_S - model_S))
    return D


def find_x_min(G, x_mins):
    """choose x_min with the smallest KS distance."""
    best_x_min = x_mins[0]
    best_D = ks_distance(G, x_mins[0])
    for i in range(1, len(x_mins)):
        D = ks_distance(G, x_mins[i])
        if D < best_D:
            best_D = D
            best_x_min = x_mins[i]
    return best_x_min, best_D
