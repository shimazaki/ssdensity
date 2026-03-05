#!/usr/bin/env python
"""Generate density overlay figure: true density + fitted estimates.

Two-panel figure showing representative fits on challenging Marron-Wand densities:
  Left:  #3 Strongly skewed (8 cascading components)
  Right: #14 Smooth comb (6 components)

Methods shown: ssvkernel, KDE diffusion, Zuko NSF.

Figure style matches benchmarks/marron_wand/compare_methods.py:
  - White background, no grid, only left/bottom spines
  - True density as gray fill_between
  - Per-subplot legend with ISE values (monospace)

Usage:
  python benchmarks/performance_comparison/fig_density_overlay.py
  python benchmarks/performance_comparison/fig_density_overlay.py --n-samples 500
  python benchmarks/performance_comparison/fig_density_overlay.py --seed 123
"""
import argparse
import os
import sys
import warnings

# Ensure project root and this directory are importable
_this_dir = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.abspath(os.path.join(_this_dir, '..', '..'))
for _p in (_proj_root, _this_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from run_mise_comparison import (
    MW_DENSITIES, N_GRID,
    mw_sample, mw_support, mw_pdf, compute_ise,
    wrap_ssvkernel, wrap_kde_diffusion, wrap_awkde,
    _method_color,
)


# ── Style (matching marron_wand/compare_methods.py) ──────────────────

def setup_style():
    """Apply clean white style: no grid, only left/bottom spines."""
    plt.rcParams.update({
        'figure.dpi': 150,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '0.4',
        'axes.linewidth': 0.8,
        'axes.grid': False,
        'font.size': 8,
        'axes.titlesize': 11,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
    })


def _style_ax(ax):
    """Remove top/right spines and grid from a single axes."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate density overlay figure')
    parser.add_argument('--n-samples', type=int, default=1000,
                        help='Sample size (default: 1000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    n_samples = args.n_samples
    seed = args.seed

    methods = [
        ('KDE diffusion', wrap_kde_diffusion),
        ('awkde', wrap_awkde),
        ('ssvkernel', wrap_ssvkernel),
    ]

    densities = [MW_DENSITIES[2], MW_DENSITIES[9], MW_DENSITIES[13]]  # #3, #10, #14
    rng = np.random.default_rng(seed)

    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, density in zip(axes, densities):
        x = np.sort(mw_sample(n_samples, density, rng))
        lo, hi = mw_support(density)
        t_grid = np.linspace(lo, hi, N_GRID)
        f_true = mw_pdf(t_grid, density)

        # Data ticks (rug plot along x-axis)
        ax.plot(x, np.zeros_like(x), '|', color='0.3', markersize=6,
                alpha=0.3, zorder=1)

        # True density (gray fill, matching Marron-Wand style)
        ax.fill_between(t_grid, f_true, color='0.85', alpha=0.6, zorder=0.5)

        # Method estimates
        legend_handles = []
        for name, func in methods:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                y_est = func(x, t_grid)
            ise = compute_ise(y_est, t_grid, f_true)
            color = _method_color(name)
            ax.plot(t_grid, y_est, color=color, linewidth=1.5,
                    alpha=0.9, zorder=3)
            label = f'{name:<14s} {ise:.1e}'
            legend_handles.append(
                Line2D([0], [0], color=color, lw=1.5, label=label))

        # Per-subplot legend (monospace, matching Marron-Wand style)
        from matplotlib.patches import Patch
        true_handle = Patch(facecolor='0.85', edgecolor='none', alpha=0.6,
                            label='True density')
        header = Line2D([0], [0], color='none',
                        label=f'{"":14s} ISE')
        ax.legend(handles=[true_handle, header] + legend_handles, loc='upper right',
                  fontsize=7, prop={'family': 'monospace', 'size': 7},
                  framealpha=0.9, edgecolor='none', borderpad=0.4,
                  handlelength=1.8, handletextpad=0.5, labelspacing=0.3)

        ax.set_title(density['name'], fontsize=11, fontweight='bold')
        ax.set_xlim(lo, hi)
        ax.set_ylim(bottom=0, top=f_true.max() * 1.15)
        _style_ax(ax)

    fig.suptitle(
        f'Fitted Density Estimates  (n={n_samples})',
        fontsize=13, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'density_overlay.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Figure saved to {path}')


if __name__ == '__main__':
    main()
