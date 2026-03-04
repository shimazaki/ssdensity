#!/usr/bin/env python
"""Benchmark classic vs optimized implementations on Marron-Wand #1 Gaussian.

Prints a markdown table of median wall times (3 warmup + 20 measured runs).
"""

import time
import numpy as np

from ssdensity import (
    sshist, sskernel, ssvkernel,
    sshist_classic, sskernel_classic, ssvkernel_classic,
)


def bench(func, args, kwargs, n_warmup=3, n_runs=20):
    """Return median wall time in seconds over n_runs (after n_warmup)."""
    for _ in range(n_warmup):
        func(*args, **kwargs)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times)


def main():
    # Marron-Wand #1: standard Gaussian, n=1000
    rng = np.random.RandomState(42)
    x = rng.randn(1000)

    benchmarks = [
        ("sshist", sshist_classic, sshist,
         (x,), {}, (x,), {}),
        ("sskernel", sskernel_classic, sskernel,
         (x,), {"bootstrap": 200}, (x,), {"bootstrap": 200}),
        ("ssvkernel", ssvkernel_classic, ssvkernel,
         (x,), {"bootstrap": 50}, (x,), {"bootstrap": 50}),
    ]

    print("| Function | Classic (ms) | Optimized (ms) | Speedup |")
    print("|----------|-------------|----------------|---------|")

    for name, classic_fn, opt_fn, c_args, c_kw, o_args, o_kw in benchmarks:
        t_classic = bench(classic_fn, c_args, c_kw) * 1000
        t_opt = bench(opt_fn, o_args, o_kw) * 1000
        speedup = t_classic / t_opt
        print(f"| `{name}` | {t_classic:.1f} | {t_opt:.1f} | {speedup:.0f}x |")


if __name__ == "__main__":
    main()
