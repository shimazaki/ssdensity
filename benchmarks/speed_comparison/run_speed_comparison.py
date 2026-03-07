#!/usr/bin/env python
"""Speed comparison: ssdensity vs standard Python density estimation packages.

Benchmarks execution time (CPU) of ssdensity methods against numpy, scipy,
KDEpy, statsmodels, and scikit-learn on bimodal synthetic data.

Usage:
    python benchmarks/speed_comparison/run_speed_comparison.py [--n-runs 5] [--skip-large]
"""
import argparse
import platform
import sys
import time

import numpy as np

# --- ssdensity (always required) -------------------------------------------
from ssdensity.sshist import sshist
from ssdensity.sskernel import sskernel
from ssdensity.ssvkernel import ssvkernel

# --- Optional packages (skip gracefully if missing) -------------------------
try:
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from KDEpy import FFTKDE
    HAS_KDEPY = True
except ImportError:
    HAS_KDEPY = False

try:
    from statsmodels.nonparametric.kde import KDEUnivariate
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from sklearn.neighbors import KernelDensity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from awkde import GaussianKDE
    HAS_AWKDE = True
except ImportError:
    HAS_AWKDE = False

N_GRID = 256  # common evaluation grid size for KDE methods


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_bimodal(n, seed=42):
    """Bimodal mixture: 0.5 * N(-1,1) + 0.5 * N(1,1)."""
    rng = np.random.RandomState(seed)
    x = np.concatenate([
        rng.normal(-1, 1, n // 2),
        rng.normal(1, 1, n - n // 2),
    ])
    return np.sort(x)


def make_grid(x):
    """Linearly-spaced evaluation grid spanning the data range."""
    margin = 0.2 * (x[-1] - x[0])
    return np.linspace(x[0] - margin, x[-1] + margin, N_GRID)


def scott_bandwidth(x):
    """Scott's rule of thumb: h = n^{-1/5} * std."""
    return len(x) ** (-1.0 / 5.0) * np.std(x, ddof=1)


# ---------------------------------------------------------------------------
# Benchmark wrappers — each returns None (we only care about timing)
# ---------------------------------------------------------------------------

def bench_sshist(x, t_grid):
    sshist(x)


def bench_sskernel(x, t_grid):
    sskernel(x, tin=t_grid, bootstrap=0)


def bench_ssvkernel(x, t_grid):
    ssvkernel(x, tin=t_grid, bootstrap=0)


def bench_np_histogram(x, t_grid):
    np.histogram(x, bins='scott')


def bench_scipy_kde(x, t_grid):
    kde = gaussian_kde(x)
    kde(t_grid)


def bench_kdepy(x, t_grid):
    FFTKDE(bw='silverman').fit(x).evaluate(t_grid)


def bench_statsmodels(x, t_grid):
    kde = KDEUnivariate(x)
    kde.fit(bw='normal_reference')
    kde.evaluate(t_grid)


def bench_sklearn(x, t_grid):
    bw = scott_bandwidth(x)
    kde = KernelDensity(bandwidth=bw, kernel='gaussian')
    kde.fit(x[:, np.newaxis])
    kde.score_samples(t_grid[:, np.newaxis])


def bench_awkde(x, t_grid):
    kde = GaussianKDE(glob_bw='silverman', alpha=0.5, diag_cov=True)
    kde.fit(x[:, np.newaxis])
    kde.predict(t_grid[:, np.newaxis])


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------

def time_func(func, x, t_grid, n_runs):
    """Run func n_runs times, return median wall time in seconds."""
    # Warmup
    func(x, t_grid)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        func(x, t_grid)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
    return np.median(times)


def fmt_time(seconds):
    """Format time with appropriate units."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:>7.1f} us"
    elif seconds < 1.0:
        return f"{seconds * 1e3:>7.1f} ms"
    else:
        return f"{seconds:>7.2f} s "


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Speed comparison: ssdensity vs standard methods")
    parser.add_argument("--n-runs", type=int, default=5,
                        help="Number of timed runs per method (default: 5)")
    parser.add_argument("--skip-large", action="store_true",
                        help="Skip n=100000 dataset")
    args = parser.parse_args()

    # Build method list
    methods = [
        ("sshist", bench_sshist, True),
        ("sskernel", bench_sskernel, True),
        ("ssvkernel", bench_ssvkernel, True),
        ("np.histogram(scott)", bench_np_histogram, True),
        ("scipy.gaussian_kde", bench_scipy_kde, HAS_SCIPY),
        ("KDEpy FFTKDE", bench_kdepy, HAS_KDEPY),
        ("statsmodels KDE", bench_statsmodels, HAS_STATSMODELS),
        ("sklearn KernelDensity", bench_sklearn, HAS_SKLEARN),
        ("awkde (adaptive)", bench_awkde, HAS_AWKDE),
    ]

    # Report missing packages
    missing = [name for name, _, avail in methods if not avail]
    if missing:
        print(f"  WARNING: skipping unavailable packages: {', '.join(missing)}")
        print()

    active_methods = [(name, func) for name, func, avail in methods if avail]

    # Dataset sizes
    sizes = [1_000, 10_000]
    if not args.skip_large:
        sizes.append(100_000)

    # Generate datasets and grids
    datasets = {}
    for n in sizes:
        x = make_bimodal(n)
        t_grid = make_grid(x)
        datasets[n] = (x, t_grid)

    # Run benchmarks
    # results[method_name][n] = median_seconds
    results = {}
    for name, func in active_methods:
        results[name] = {}
        for n in sizes:
            x, t_grid = datasets[n]
            sys.stdout.write(f"\r  Timing {name:<25s} n={n:<8d}")
            sys.stdout.flush()
            median_t = time_func(func, x, t_grid, args.n_runs)
            results[name][n] = median_t
    sys.stdout.write("\r" + " " * 60 + "\r")
    sys.stdout.flush()

    # --- Print results table ---
    name_width = max(len(name) for name, _ in active_methods)
    col_width = 12
    header_sizes = "".join(f"{'n=' + str(n):>{col_width}}" for n in sizes)

    print("=" * 70)
    print("  Speed Comparison: ssdensity vs standard methods")
    print(f"  Python {platform.python_version()}, "
          f"NumPy {np.__version__}, "
          f"n_runs={args.n_runs}")
    print("=" * 70)
    print()
    print(f"  {'Method':<{name_width}}  {header_sizes}")
    print("  " + "-" * (name_width + len(sizes) * col_width + 2))

    for name, _ in active_methods:
        row = f"  {name:<{name_width}}"
        for n in sizes:
            row += f"  {fmt_time(results[name][n])}"
        print(row)

    print("  " + "-" * (name_width + len(sizes) * col_width + 2))
    print()
    print(f"  Grid size: {N_GRID} points (KDE methods)")
    print(f"  sshist: bins selected by MISE cost optimization")
    print(f"  sskernel/ssvkernel: bootstrap=0 (no bootstrap)")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
