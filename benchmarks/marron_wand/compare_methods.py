#!/usr/bin/env python
"""Marron-Wand benchmark: optimized vs classic ssdensity.

Compare optimized and classic (original) implementations of sskernel,
ssvkernel, and sshist on the 15 Marron-Wand (1992) benchmark densities.

Generates two figures:
  1. compare_methods.png         -- all 6 methods (optimized + classic)
  2. compare_methods_optimized.png -- optimized only (3 methods)

Reference:
  Marron, J. S. and Wand, M. P. (1992). "Exact Mean Integrated Squared Error,"
  The Annals of Statistics, 20(2):712-736.

Usage:
  python benchmarks/marron_wand/compare_methods.py
"""
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from ssdensity import (
    sshist, sskernel, ssvkernel,
    sshist_classic, sskernel_classic, ssvkernel_classic,
)

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# ── Constants ────────────────────────────────────────────────────────

N_SAMPLES = 1000
N_GRID = 1024
N_REPEATS = 3
BS_SSKERNEL = 100
BS_SSVKERNEL = 50
RNG = np.random.default_rng(42)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Colors: optimized (dark, solid) and classic (light, dashed)
CLR_SSHIST_OPT = '#9467bd'     # purple
CLR_SSHIST_CLS = '#c2a5cf'     # light purple
CLR_SSKERNEL_OPT = '#1f77b4'   # blue
CLR_SSKERNEL_CLS = '#6baed6'   # light blue
CLR_SSVKERNEL_OPT = '#ff7f0e'  # orange (fig2) / '#d95f02' (fig1)
CLR_SSVKERNEL_CLS = '#fdae6b'  # light orange

# Fig1 uses slightly different orange to match original notebook
CLR_SSVKERNEL_OPT_FIG1 = '#d95f02'
CLR_SSHIST_OPT_FIG1 = '#7b3294'

# ── 15 Marron-Wand densities ────────────────────────────────────────

MW_DENSITIES = [
    {'name': '#1 Gaussian',
     'weights': [1.0],
     'means': [0.0],
     'sigmas': [1.0]},

    {'name': '#2 Skewed unimodal',
     'weights': [1/5, 1/5, 3/5],
     'means': [0.0, 1/2, 13/12],
     'sigmas': [1.0, 2/3, 5/9]},

    {'name': '#3 Strongly skewed',
     'weights': [1/8]*8,
     'means': [3*((2/3)**k - 1) for k in range(8)],
     'sigmas': [(2/3)**k for k in range(8)]},

    {'name': '#4 Kurtotic unimodal',
     'weights': [2/3, 1/3],
     'means': [0.0, 0.0],
     'sigmas': [1.0, 1/10]},

    {'name': '#5 Outlier',
     'weights': [1/10, 9/10],
     'means': [0.0, 0.0],
     'sigmas': [1.0, 1/10]},

    {'name': '#6 Bimodal',
     'weights': [1/2, 1/2],
     'means': [-1.0, 1.0],
     'sigmas': [2/3, 2/3]},

    {'name': '#7 Separated bimodal',
     'weights': [1/2, 1/2],
     'means': [-3/2, 3/2],
     'sigmas': [1/2, 1/2]},

    {'name': '#8 Asymmetric bimodal',
     'weights': [3/4, 1/4],
     'means': [0.0, 3/2],
     'sigmas': [1.0, 1/3]},

    {'name': '#9 Trimodal',
     'weights': [9/20, 9/20, 1/10],
     'means': [-6/5, 6/5, 0.0],
     'sigmas': [3/5, 3/5, 1/4]},

    {'name': '#10 Claw',
     'weights': [1/2] + [1/10]*5,
     'means': [0.0] + [(k-2)/2 for k in range(5)],
     'sigmas': [1.0] + [1/10]*5},

    {'name': '#11 Double claw',
     'weights': [49/100, 49/100] + [1/350]*7,
     'means': [-1.0, 1.0] + [(k-3)/2 for k in range(7)],
     'sigmas': [2/3, 2/3] + [1/100]*7},

    # Verified against Marron's nmpar.m Matlab source.
    {'name': '#12 Asymmetric claw',
     'weights': [1/2] + [2**(1-l)/31 for l in range(-2, 3)],
     'means': [0.0] + [l + 1/2 for l in range(-2, 3)],
     'sigmas': [1.0] + [2**(-l)/10 for l in range(-2, 3)]},

    {'name': '#13 Asym. double claw',
     'weights': [46/100, 46/100, 1/300, 1/300, 1/300, 7/300, 7/300, 7/300],
     'means': [-1.0, 1.0, -3/2, -1.0, -1/2, 1/2, 1.0, 3/2],
     'sigmas': [2/3, 2/3, 1/100, 1/100, 1/100, 7/100, 7/100, 7/100]},

    {'name': '#14 Smooth comb',
     'weights': [2**(5-k)/63 for k in range(6)],
     'means': [(65 - 96*(1/2)**k)/21 for k in range(6)],
     'sigmas': [(32/63)*(1/2)**k for k in range(6)]},

    {'name': '#15 Discrete comb',
     'weights': [2/7, 2/7, 2/7, 1/21, 1/21, 1/21],
     'means': [(12*k - 15)/7 for k in range(6)],
     'sigmas': [2/7, 2/7, 2/7, 1/21, 1/21, 1/21]},
]

# Verify all weights sum to 1
for _i, _d in enumerate(MW_DENSITIES):
    _s = sum(_d['weights'])
    assert abs(_s - 1.0) < 1e-12, f"#{_i+1} {_d['name']}: weights sum to {_s}"


# ── Helpers ──────────────────────────────────────────────────────────

def mw_pdf(x, density):
    """Evaluate true Marron-Wand density at points x."""
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=float)
    for w, mu, sig in zip(density['weights'], density['means'],
                          density['sigmas']):
        result += (w / (sig * np.sqrt(2 * np.pi))
                   * np.exp(-0.5 * ((x - mu) / sig)**2))
    return result


def mw_sample(n, density, rng):
    """Draw n samples from a Marron-Wand Gaussian mixture."""
    weights = np.array(density['weights'])
    means = np.array(density['means'])
    sigmas = np.array(density['sigmas'])
    components = rng.choice(len(weights), size=n, p=weights/weights.sum())
    return rng.normal(means[components], sigmas[components])


def mw_support(density, margin=5):
    """Return (lo, hi) covering +/-margin*sigma from outermost component."""
    means = np.array(density['means'])
    sigmas = np.array(density['sigmas'])
    lo = np.min(means - margin * sigmas)
    hi = np.max(means + margin * sigmas)
    return lo, hi


def compute_ise(y_est, t_fine, f_true):
    """Integrated squared error via trapezoidal rule."""
    return np.trapezoid((y_est - f_true)**2, t_fine)


def hist_to_density(x, optN, t_fine):
    """Build piecewise-constant histogram density on fine grid."""
    counts, edges = np.histogram(x, bins=optN, density=True)
    idx = np.digitize(t_fine, edges) - 1
    y = np.zeros_like(t_fine)
    mask = (idx >= 0) & (idx < len(counts))
    y[mask] = counts[idx[mask]]
    return y, edges


def timed_run(func, n_repeats=N_REPEATS):
    """Run func() n_repeats times, return (result, median_cpu_seconds)."""
    times = []
    result = None
    for _ in range(n_repeats):
        np.random.seed(0)
        t0 = time.perf_counter()
        result = func()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return result, np.median(times)


# ── Benchmarks ───────────────────────────────────────────────────────

def run_all_benchmarks():
    """Run 15 densities x 6 methods. Returns list of result dicts."""
    results = []

    for i, density in enumerate(MW_DENSITIES):
        print(f'[{i+1:2d}/15] {density["name"]}', end='', flush=True)

        x = mw_sample(N_SAMPLES, density, RNG)
        lo, hi = mw_support(density)
        lo = min(lo, x.min() - 0.5)
        hi = max(hi, x.max() + 0.5)
        t_fine = np.linspace(lo, hi, N_GRID)
        f_true = mw_pdf(t_fine, density)

        rec = {'name': density['name'], 'x': x, 't_fine': t_fine,
               'f_true': f_true}

        # sskernel (optimized)
        res, dt = timed_run(
            lambda: sskernel(x, tin=t_fine, bootstrap=BS_SSKERNEL))
        rec['sskernel'] = {'y': res[0],
                           'ise': compute_ise(res[0], t_fine, f_true),
                           'time': dt}

        # sskernel (classic)
        res, dt = timed_run(
            lambda: sskernel_classic(x, tin=t_fine, bootstrap=BS_SSKERNEL))
        rec['sskernel_classic'] = {'y': res[0],
                                   'ise': compute_ise(res[0], t_fine, f_true),
                                   'time': dt}

        # ssvkernel (optimized)
        res, dt = timed_run(
            lambda: ssvkernel(x, tin=t_fine, bootstrap=BS_SSVKERNEL))
        rec['ssvkernel'] = {'y': res[0],
                            'ise': compute_ise(res[0], t_fine, f_true),
                            'time': dt}

        # ssvkernel (classic)
        res, dt = timed_run(
            lambda: ssvkernel_classic(x, tin=t_fine, bootstrap=BS_SSVKERNEL))
        rec['ssvkernel_classic'] = {'y': res[0],
                                    'ise': compute_ise(res[0], t_fine, f_true),
                                    'time': dt}

        # sshist (optimized)
        res, dt = timed_run(lambda: sshist(x))
        optN = res[0]
        y_hist, hist_edges = hist_to_density(x, optN, t_fine)
        rec['sshist'] = {'y': y_hist, 'edges': hist_edges, 'optN': optN,
                         'ise': compute_ise(y_hist, t_fine, f_true),
                         'time': dt}

        # sshist (classic)
        res, dt = timed_run(lambda: sshist_classic(x))
        optN_c = res[0]
        y_hist_c, hist_edges_c = hist_to_density(x, optN_c, t_fine)
        rec['sshist_classic'] = {'y': y_hist_c, 'edges': hist_edges_c,
                                 'optN': optN_c,
                                 'ise': compute_ise(y_hist_c, t_fine, f_true),
                                 'time': dt}

        results.append(rec)
        print(' done')

    return results


# ── Style ────────────────────────────────────────────────────────────

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


# ── Figure 1: optimized vs classic (6 methods) ──────────────────────

def fig_compare_all(results, path):
    """5x3 grid: all 6 methods with ISE/timing annotations."""
    fig, axes = plt.subplots(5, 3, figsize=(14, 18))

    for rec, ax in zip(results, axes.flat):
        t = rec['t_fine']

        # True density
        ax.fill_between(t, rec['f_true'], color='0.85', alpha=0.6,
                        zorder=0.5)

        # Optimized (solid)
        ax.step(t, rec['sshist']['y'], color=CLR_SSHIST_OPT_FIG1, lw=1.2,
                where='mid', zorder=2)
        ax.plot(t, rec['sskernel']['y'], color=CLR_SSKERNEL_OPT, lw=1.5,
                zorder=3)
        ax.plot(t, rec['ssvkernel']['y'], color=CLR_SSVKERNEL_OPT_FIG1,
                lw=1.5, zorder=4)

        # Classic (dashed)
        ax.step(t, rec['sshist_classic']['y'], color=CLR_SSHIST_CLS, lw=1.2,
                ls='--', where='mid', zorder=5)
        ax.plot(t, rec['sskernel_classic']['y'], color=CLR_SSKERNEL_CLS,
                lw=1.5, ls='--', zorder=6)
        ax.plot(t, rec['ssvkernel_classic']['y'], color=CLR_SSVKERNEL_CLS,
                lw=1.5, ls='--', zorder=7)

        ax.set_title(rec['name'], fontsize=11, fontweight='bold')
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(bottom=0)
        _style_ax(ax)

        # Compact annotation
        methods = [
            ('sshist',     CLR_SSHIST_OPT_FIG1),
            ('sskernel',   CLR_SSKERNEL_OPT),
            ('ssvkernel',  CLR_SSVKERNEL_OPT_FIG1),
        ]
        lines = ['            ISE      ms']
        for mname, _ in methods:
            r_opt = rec[mname]
            r_cls = rec[mname + '_classic']
            lines.append(
                f'{mname:>10s}  {r_opt["ise"]:.1e}  {r_opt["time"]*1000:6.0f}')
            lines.append(
                f'{"classic":>10s}  {r_cls["ise"]:.1e}  {r_cls["time"]*1000:6.0f}')
        txt = '\n'.join(lines)
        ax.text(0.98, 0.98, txt, transform=ax.transAxes,
                fontsize=6.5, fontfamily='monospace',
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='none', alpha=0.9))

    # Shared legend
    LW_LEG = 3.0
    legend_elements = [
        Patch(facecolor='0.85', edgecolor='none', alpha=0.6,
              label='True density'),
        Line2D([0], [0], color=CLR_SSHIST_OPT_FIG1, lw=LW_LEG, ls='-',
               label='sshist (optimized)'),
        Line2D([0], [0], color=CLR_SSHIST_CLS, lw=LW_LEG, ls='--',
               label='sshist (classic)'),
        Line2D([0], [0], color=CLR_SSKERNEL_OPT, lw=LW_LEG, ls='-',
               label='sskernel (optimized)'),
        Line2D([0], [0], color=CLR_SSKERNEL_CLS, lw=LW_LEG, ls='--',
               label='sskernel (classic)'),
        Line2D([0], [0], color=CLR_SSVKERNEL_OPT_FIG1, lw=LW_LEG, ls='-',
               label='ssvkernel (optimized)'),
        Line2D([0], [0], color=CLR_SSVKERNEL_CLS, lw=LW_LEG, ls='--',
               label='ssvkernel (classic)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=10, frameon=True, fancybox=True, handlelength=4.0,
               handleheight=1.5, columnspacing=2.5, borderpad=0.8)

    fig.suptitle(
        'Marron\u2013Wand Benchmark: Optimized vs Classic ssdensity  (n=1000)',
        fontsize=13, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0.05, 1, 0.99])
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Figure saved to {path}')


# ── Figure 2: optimized only (3 methods, per-subplot legend) ────────

def fig_compare_optimized(results, path):
    """5x3 grid: optimized methods only with per-subplot legends."""
    fig, axes = plt.subplots(5, 3, figsize=(14, 18))

    for rec, ax in zip(results, axes.flat):
        t = rec['t_fine']

        # True density
        ax.fill_between(t, rec['f_true'], color='0.85', alpha=0.6,
                        zorder=0.5)

        # Optimized only
        ax.step(t, rec['sshist']['y'], color=CLR_SSHIST_OPT, lw=1.2,
                where='mid', zorder=2)
        ax.plot(t, rec['sskernel']['y'], color=CLR_SSKERNEL_OPT, lw=1.5,
                zorder=3)
        ax.plot(t, rec['ssvkernel']['y'], color=CLR_SSVKERNEL_OPT, lw=1.5,
                zorder=4)

        ax.set_title(rec['name'], fontsize=11, fontweight='bold')
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(bottom=0)
        _style_ax(ax)

        # Per-subplot legend with ISE and timing
        methods = [
            ('sshist',    CLR_SSHIST_OPT,    1.2),
            ('sskernel',  CLR_SSKERNEL_OPT,  1.5),
            ('ssvkernel', CLR_SSVKERNEL_OPT, 1.5),
        ]
        handles = []
        for mname, clr, lw in methods:
            r = rec[mname]
            label = f'{mname:<9s}  {r["ise"]:.1e}  {r["time"]*1000:5.0f}ms'
            handles.append(
                Line2D([0], [0], color=clr, lw=lw, ls='-', label=label))
        header = Line2D([0], [0], color='none',
                        label='            ISE      time')
        ax.legend(handles=[header] + handles, loc='upper right', fontsize=7,
                  prop={'family': 'monospace', 'size': 7},
                  framealpha=0.9, edgecolor='none', borderpad=0.4,
                  handlelength=1.8, handletextpad=0.5, labelspacing=0.3)

    fig.suptitle(
        'Marron\u2013Wand Benchmark: ssdensity  (n=1000)',
        fontsize=13, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Figure saved to {path}')


# ── Summary tables ───────────────────────────────────────────────────

def print_summary(results):
    """Print ISE and wall time tables to stdout."""
    header = (f'{"Density":<25s}  {"sskernel":>10s} {"classic":>10s}'
              f'  {"ssvkernel":>10s} {"classic":>10s}'
              f'  {"sshist":>10s} {"classic":>10s}')
    methods = ['sskernel', 'sskernel_classic', 'ssvkernel',
               'ssvkernel_classic', 'sshist', 'sshist_classic']

    print('\nISE (x1e-3)')
    print(header)
    print('-' * len(header))
    for rec in results:
        vals = [f'{rec[m]["ise"]*1000:10.2f}' for m in methods]
        print(f'{rec["name"]:<25s}  {vals[0]} {vals[1]}'
              f'  {vals[2]} {vals[3]}  {vals[4]} {vals[5]}')

    print('\nWall time (ms)')
    print(header)
    print('-' * len(header))
    for rec in results:
        vals = [f'{rec[m]["time"]*1000:10.1f}' for m in methods]
        print(f'{rec["name"]:<25s}  {vals[0]} {vals[1]}'
              f'  {vals[2]} {vals[3]}  {vals[4]} {vals[5]}')


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print(f'{len(MW_DENSITIES)} Marron-Wand densities, '
          f'n={N_SAMPLES}, {N_REPEATS} repeats')

    results = run_all_benchmarks()

    setup_style()

    fig_compare_all(
        results, os.path.join(OUT_DIR, 'compare_methods.png'))
    fig_compare_optimized(
        results, os.path.join(OUT_DIR, 'compare_methods_optimized.png'))

    print_summary(results)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
