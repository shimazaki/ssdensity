#!/usr/bin/env python
"""MISE performance comparison: ssdensity vs state-of-the-art density estimators.

Evaluates 17 methods on the 15 Marron-Wand (1992) benchmark densities via
Monte Carlo estimation of Median Integrated Squared Error. All methods
use default settings for a fair comparison.

Methods:
  1. sskernel        — ssdensity, L2 cost minimization
  2. ssvkernel       — ssdensity, locally adaptive L2 cost
  3. sshist          — ssdensity, MISE-optimal bin width
  4. KDEpy ISJ       — Improved Sheather-Jones (diffusion bandwidth)
  5. KDEpy Silverman — Silverman rule-of-thumb
  6. scipy kde       — Scott's rule (default)
  7. statsmodels KDE — normal_reference (default)
  8. fastkde         — self-consistent bandwidth (Bernacchia & Pigolotti 2011)
  9. KDE diffusion   — Botev et al. 2010 (full diffusion estimator)
 10. awkde           — adaptive-width KDE (mennthor/awkde, alpha=0.5)
 11. GMM BIC         — sklearn EM + BIC model selection
 12. Bayesian GMM    — sklearn Variational DP
 13. Neural Spline Flow — normflows, Adam optimizer
 14. Zuko NSF        — zuko neural spline flow, Adam optimizer
 15. Zuko UNAF       — zuko UMNN autoregressive flow, Adam optimizer
 16. ARF             — adversarial random forest (arfpy), forde density
 17. TransportMap    — polynomial transport map (TransportMaps), KL fit

Fairness note:
  The Marron-Wand densities are all finite Gaussian mixtures. GMM-based methods
  (GMM BIC, Bayesian GMM) therefore fit the correct parametric family and enjoy
  a structural advantage on this benchmark. Their strong MISE scores here may
  not generalize to non-Gaussian-mixture densities (e.g., uniform, beta,
  log-normal, heavy-tailed). KDE and histogram methods make no parametric
  assumption and are expected to be more robust across arbitrary densities.
  ARF trains a random forest discriminator, which has a structural advantage on
  data that random forests can partition well (e.g., multimodal). TransportMap
  uses polynomial basis, which may underfit highly multimodal densities.

References:
  Marron, J. S. and Wand, M. P. (1992). "Exact Mean Integrated Squared Error,"
  The Annals of Statistics, 20(2):712-736.

Usage:
  python benchmarks/performance_comparison/run_mise_comparison.py [options]
  python benchmarks/performance_comparison/run_mise_comparison.py --quick
  python benchmarks/performance_comparison/run_mise_comparison.py --mc-runs 100 --n-samples 1000
  python benchmarks/performance_comparison/run_mise_comparison.py --no-flow --no-figures
"""
import argparse
import json
import os
import platform
import re
import sys
import time
import warnings

import numpy as np
from tqdm import tqdm

# ── A. Imports + optional package detection ──────────────────────────

from ssdensity.sshist import sshist
from ssdensity.sskernel import sskernel
from ssdensity.ssvkernel import ssvkernel

try:
    from KDEpy import FFTKDE
    HAS_KDEPY = True
except ImportError:
    HAS_KDEPY = False

try:
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from statsmodels.nonparametric.kde import KDEUnivariate
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import fastkde
    HAS_FASTKDE = True
except ImportError:
    HAS_FASTKDE = False

try:
    from kde_diffusion import kde1d
    from scipy.interpolate import interp1d
    HAS_KDEDIFFUSION = True
except ImportError:
    HAS_KDEDIFFUSION = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    from awkde import GaussianKDE
    HAS_AWKDE = True
except ImportError:
    HAS_AWKDE = False

try:
    import torch
    import normflows as nf
    HAS_NORMFLOWS = True
except ImportError:
    HAS_NORMFLOWS = False

try:
    import torch
    import zuko
    HAS_ZUKO = True
except ImportError:
    HAS_ZUKO = False

# arfpy needs np.in1d monkey-patch for NumPy 2.0+
try:
    import numpy as _np_check
    if not hasattr(_np_check, 'in1d'):
        _np_check.in1d = _np_check.isin
    from arfpy import arf as arf_module
    import pandas as pd
    HAS_ARF = True
except ImportError:
    HAS_ARF = False

try:
    import TransportMaps as TM
    import TransportMaps.Maps as TM_MAPS
    import TransportMaps.Distributions as TM_DIST
    import TransportMaps.KL as TM_KL
    import logging as _tm_logging
    TM.setLogLevel(_tm_logging.WARNING)
    HAS_TRANSPORTMAPS = True
except ImportError:
    HAS_TRANSPORTMAPS = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ── B. Constants ─────────────────────────────────────────────────────

N_GRID = 1024       # evaluation grid points
MW_MARGIN = 5       # support margin in sigma units
GMM_MAX_K = 15      # max components for BIC search
NSF_LAYERS = 3      # number of NSF flow layers
NSF_HIDDEN = 32     # hidden units per layer
NSF_BINS = 8        # rational-quadratic spline bins
NSF_EPOCHS = 300    # training epochs
NSF_LR = 5e-4       # learning rate
NSF_BATCH = 256     # batch size

ZUKO_TRANSFORMS = 3   # number of spline transforms
ZUKO_HIDDEN = (32, 32) # hidden features per transform
ZUKO_BINS = 8          # rational-quadratic spline bins
ZUKO_EPOCHS = 300      # training epochs
ZUKO_LR = 5e-4         # learning rate
ZUKO_BATCH = 256       # batch size

# Zuko UNAF (UMNN-based flow)
UNAF_TRANSFORMS = 8
UNAF_HIDDEN = (64, 64)
UNAF_SIGNAL = 16
UNAF_EPOCHS = 300
UNAF_LR = 5e-4
UNAF_BATCH = 256

# TransportMap
TM_ORDER = 10       # polynomial order
TM_TOL = 1e-4       # optimization tolerance


# ── C. Marron-Wand density definitions ───────────────────────────────

MW_DENSITIES = [
    # 1. Gaussian
    {'name': '#1 Gaussian',
     'weights': [1.0],
     'means': [0.0],
     'sigmas': [1.0]},

    # 2. Skewed unimodal
    {'name': '#2 Skewed unimodal',
     'weights': [1/5, 1/5, 3/5],
     'means': [0.0, 1/2, 13/12],
     'sigmas': [1.0, 2/3, 5/9]},

    # 3. Strongly skewed
    {'name': '#3 Strongly skewed',
     'weights': [1/8]*8,
     'means': [3*((2/3)**k - 1) for k in range(8)],
     'sigmas': [(2/3)**k for k in range(8)]},

    # 4. Kurtotic unimodal
    {'name': '#4 Kurtotic unimodal',
     'weights': [2/3, 1/3],
     'means': [0.0, 0.0],
     'sigmas': [1.0, 1/10]},

    # 5. Outlier
    {'name': '#5 Outlier',
     'weights': [1/10, 9/10],
     'means': [0.0, 0.0],
     'sigmas': [1.0, 1/10]},

    # 6. Bimodal
    {'name': '#6 Bimodal',
     'weights': [1/2, 1/2],
     'means': [-1.0, 1.0],
     'sigmas': [2/3, 2/3]},

    # 7. Separated bimodal
    {'name': '#7 Separated bimodal',
     'weights': [1/2, 1/2],
     'means': [-3/2, 3/2],
     'sigmas': [1/2, 1/2]},

    # 8. Asymmetric bimodal
    {'name': '#8 Asymmetric bimodal',
     'weights': [3/4, 1/4],
     'means': [0.0, 3/2],
     'sigmas': [1.0, 1/3]},

    # 9. Trimodal
    {'name': '#9 Trimodal',
     'weights': [9/20, 9/20, 1/10],
     'means': [-6/5, 6/5, 0.0],
     'sigmas': [3/5, 3/5, 1/4]},

    # 10. Claw
    {'name': '#10 Claw',
     'weights': [1/2] + [1/10]*5,
     'means': [0.0] + [(k - 2)/2 for k in range(5)],
     'sigmas': [1.0] + [1/10]*5},

    # 11. Double claw
    {'name': '#11 Double claw',
     'weights': [49/100, 49/100] + [1/350]*7,
     'means': [-1.0, 1.0] + [(k - 3)/2 for k in range(7)],
     'sigmas': [2/3, 2/3] + [1/100]*7},

    # 12. Asymmetric claw
    # 1/2 N(0,1) + sum_{l=-2}^{2} [2^{1-l}/31] N(l+1/2, (2^{-l}/10)^2)
    {'name': '#12 Asymmetric claw',
     'weights': [1/2] + [2**(1 - l)/31 for l in range(-2, 3)],
     'means': [0.0] + [l + 1/2 for l in range(-2, 3)],
     'sigmas': [1.0] + [2**(-l)/10 for l in range(-2, 3)]},

    # 13. Asymmetric double claw
    {'name': '#13 Asym. double claw',
     'weights': [46/100, 46/100, 1/300, 1/300, 1/300, 7/300, 7/300, 7/300],
     'means': [-1.0, 1.0, -3/2, -1.0, -1/2, 1/2, 1.0, 3/2],
     'sigmas': [2/3, 2/3, 1/100, 1/100, 1/100, 7/100, 7/100, 7/100]},

    # 14. Smooth comb
    {'name': '#14 Smooth comb',
     'weights': [2**(5 - k)/63 for k in range(6)],
     'means': [(65 - 96*(1/2)**k)/21 for k in range(6)],
     'sigmas': [(32/63)*(1/2)**k for k in range(6)]},

    # 15. Discrete comb
    {'name': '#15 Discrete comb',
     'weights': [2/7, 2/7, 2/7, 1/21, 1/21, 1/21],
     'means': [(12*k - 15)/7 for k in range(6)],
     'sigmas': [2/7, 2/7, 2/7, 1/21, 1/21, 1/21]},
]


def mw_pdf(x, density):
    """Evaluate true Marron-Wand density at points x."""
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=float)
    for w, mu, sig in zip(density['weights'], density['means'], density['sigmas']):
        result += w / (sig * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sig)**2)
    return result


def mw_sample(n, density, rng):
    """Draw n samples from a Marron-Wand Gaussian mixture."""
    weights = np.array(density['weights'])
    means = np.array(density['means'])
    sigmas = np.array(density['sigmas'])
    components = rng.choice(len(weights), size=n, p=weights / weights.sum())
    return rng.normal(means[components], sigmas[components])


def mw_support(density, margin=MW_MARGIN):
    """Return (lo, hi) covering +/-margin*sigma from outermost component."""
    means = np.array(density['means'])
    sigmas = np.array(density['sigmas'])
    lo = np.min(means - margin * sigmas)
    hi = np.max(means + margin * sigmas)
    return lo, hi


# ── D. Helpers ───────────────────────────────────────────────────────

def compute_ise(y_est, t_grid, f_true):
    """Integrated Squared Error via trapezoidal rule."""
    return np.trapezoid((y_est - f_true)**2, t_grid)


def hist_to_density(x, n_bins, t_grid):
    """Piecewise-constant histogram density evaluated on t_grid."""
    counts, edges = np.histogram(x, bins=n_bins, density=True)
    idx = np.digitize(t_grid, edges) - 1
    y = np.zeros_like(t_grid)
    mask = (idx >= 0) & (idx < len(counts))
    y[mask] = counts[idx[mask]]
    return y


def gmm_density(model, t_grid):
    """Evaluate fitted GMM density on t_grid."""
    log_prob = model.score_samples(t_grid.reshape(-1, 1))
    return np.exp(log_prob)


# ── E. Method wrappers ──────────────────────────────────────────────
# Each: (x_sorted, t_grid) -> y_est
# Return NaN array on failure.

def wrap_sskernel(x, t_grid):
    y, t, optw, W, C, confb95, yb = sskernel(x, tin=t_grid, bootstrap=0)
    return y


def wrap_ssvkernel(x, t_grid):
    y, t, optw, gs, C, confb95, yb = ssvkernel(x, tin=t_grid, bootstrap=0)
    return y


def wrap_sshist(x, t_grid):
    optN, optD, edges, C, Ns = sshist(x)
    return hist_to_density(x, optN, t_grid)


def wrap_kdepy_isj(x, t_grid):
    y = FFTKDE(bw='ISJ').fit(x).evaluate(t_grid)
    return y


def wrap_kdepy_silverman(x, t_grid):
    y = FFTKDE(bw='silverman').fit(x).evaluate(t_grid)
    return y


def wrap_scipy_kde(x, t_grid):
    kde = gaussian_kde(x)
    return kde(t_grid)


def wrap_statsmodels_kde(x, t_grid):
    kde = KDEUnivariate(x)
    kde.fit(bw='normal_reference')
    return kde.evaluate(t_grid)


def wrap_fastkde(x, t_grid):
    result = fastkde.pdf(x)
    coords = list(result.coords)
    grid = result.coords[coords[0]].values
    density = result.values
    # Interpolate onto our evaluation grid
    y = np.interp(t_grid, grid, density, left=0.0, right=0.0)
    return y


def wrap_kde_diffusion(x, t_grid):
    lo, hi = t_grid[0], t_grid[-1]
    density, grid, bw = kde1d(x, n=2048, limits=(lo, hi))
    # Interpolate onto our evaluation grid
    y = np.interp(t_grid, grid, density, left=0.0, right=0.0)
    return y


def wrap_gmm_bic(x, t_grid):
    best_model = None
    best_bic = np.inf
    X = x.reshape(-1, 1)
    for k in range(1, GMM_MAX_K + 1):
        gm = GaussianMixture(n_components=k, random_state=0)
        gm.fit(X)
        bic = gm.bic(X)
        if bic < best_bic:
            best_bic = bic
            best_model = gm
    return gmm_density(best_model, t_grid)


def wrap_bayesian_gmm(x, t_grid):
    bgm = BayesianGaussianMixture(
        n_components=GMM_MAX_K,
        random_state=0,
    )
    bgm.fit(x.reshape(-1, 1))
    return gmm_density(bgm, t_grid)


def _get_torch_device():
    """Return best available torch device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _torch_device_tag():
    """Return '(gpu)' or '(cpu)' for labeling flow methods."""
    if torch.cuda.is_available():
        return '(gpu)'
    return '(cpu)'


def wrap_nsf(x, t_grid):
    """Neural Spline Flow (1D) with normflows.

    Uses GPU if available; falls back to single-threaded CPU.
    Timing uses perf_counter (wall time) to capture GPU compute.
    """
    device = _get_torch_device()
    if device.type == 'cpu':
        torch.set_num_threads(1)

    base = nf.distributions.DiagGaussian(1)

    flows = []
    for _ in range(NSF_LAYERS):
        flows.append(nf.flows.AutoregressiveRationalQuadraticSpline(
            1, NSF_BINS, NSF_HIDDEN))
        flows.append(nf.flows.LULinearPermute(1))

    model = nf.NormalizingFlow(base, flows).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=NSF_LR)

    x_tensor = torch.tensor(x.reshape(-1, 1), dtype=torch.float32, device=device)
    n = len(x)

    model.train()
    for _ in range(NSF_EPOCHS):
        # Mini-batch
        idx = torch.randperm(n, device=device)[:NSF_BATCH]
        loss = model.forward_kld(x_tensor[idx])
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    t_tensor = torch.tensor(t_grid.reshape(-1, 1), dtype=torch.float32, device=device)
    with torch.no_grad():
        log_prob = model.log_prob(t_tensor)
        y = torch.exp(log_prob).cpu().numpy().flatten()
    # Clamp negatives from numerical noise
    y = np.maximum(y, 0.0)
    return y


def wrap_awkde(x, t_grid):
    """Adaptive-width KDE (mennthor/awkde).

    Uses Silverman global bandwidth with alpha=0.5 local adaptation.
    """
    kde = GaussianKDE(glob_bw='silverman', alpha=0.5, diag_cov=True)
    kde.fit(x[:, np.newaxis])
    y = kde.predict(t_grid[:, np.newaxis])
    y = np.maximum(y, 0.0)
    return y


def wrap_zuko_nsf(x, t_grid):
    """Neural Spline Flow (1D) with zuko.

    Uses GPU if available; falls back to single-threaded CPU.
    Data is standardized before training (spline transforms are defined
    over a bounded interval).
    """
    device = _get_torch_device()
    if device.type == 'cpu':
        torch.set_num_threads(1)

    # Standardize data (zuko spline transforms cover [-5, 5])
    x_mean, x_std = float(np.mean(x)), float(np.std(x))
    if x_std < 1e-12:
        x_std = 1.0
    x_normed = (x - x_mean) / x_std

    flow = zuko.flows.NSF(
        features=1, context=0,
        transforms=ZUKO_TRANSFORMS,
        hidden_features=ZUKO_HIDDEN,
        bins=ZUKO_BINS,
    ).to(device)

    optimizer = torch.optim.Adam(flow.parameters(), lr=ZUKO_LR)
    x_tensor = torch.tensor(
        x_normed.reshape(-1, 1), dtype=torch.float32, device=device)
    n = len(x)

    flow.train()
    for _ in range(ZUKO_EPOCHS):
        idx = torch.randperm(n, device=device)[:ZUKO_BATCH]
        loss = -flow().log_prob(x_tensor[idx]).mean()
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    flow.eval()
    t_normed = (t_grid - x_mean) / x_std
    t_tensor = torch.tensor(
        t_normed.reshape(-1, 1), dtype=torch.float32, device=device)
    with torch.no_grad():
        log_prob = flow().log_prob(t_tensor)
        # Density in standardized space; divide by x_std for original space
        y = torch.exp(log_prob).cpu().numpy().flatten() / x_std
    y = np.maximum(y, 0.0)
    return y


def wrap_zuko_unaf(x, t_grid):
    """UMNN Autoregressive Flow (1D) with zuko.

    Uses GPU if available; falls back to single-threaded CPU.
    Data is standardized before training.
    """
    device = _get_torch_device()
    if device.type == 'cpu':
        torch.set_num_threads(1)

    x_mean, x_std = float(np.mean(x)), float(np.std(x))
    if x_std < 1e-12:
        x_std = 1.0
    x_normed = (x - x_mean) / x_std

    flow = zuko.flows.UNAF(
        features=1, context=0,
        transforms=UNAF_TRANSFORMS,
        hidden_features=UNAF_HIDDEN,
        signal=UNAF_SIGNAL,
    ).to(device)

    optimizer = torch.optim.Adam(flow.parameters(), lr=UNAF_LR)
    x_tensor = torch.tensor(
        x_normed.reshape(-1, 1), dtype=torch.float32, device=device)
    n = len(x)

    flow.train()
    for _ in range(UNAF_EPOCHS):
        idx = torch.randperm(n, device=device)[:UNAF_BATCH]
        loss = -flow().log_prob(x_tensor[idx]).mean()
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    flow.eval()
    t_normed = (t_grid - x_mean) / x_std
    t_tensor = torch.tensor(
        t_normed.reshape(-1, 1), dtype=torch.float32, device=device)
    with torch.no_grad():
        log_prob = flow().log_prob(t_tensor)
        y = torch.exp(log_prob).cpu().numpy().flatten() / x_std
    y = np.maximum(y, 0.0)
    return y


def wrap_arf(x, t_grid):
    """Adversarial Random Forest density estimation (arfpy).

    Fits ARF, estimates leaf-level truncated normal params via forde(),
    then evaluates density at t_grid by averaging across trees.
    """
    from scipy.stats import truncnorm

    df = pd.DataFrame({'x': x})
    my_arf = arf_module.arf(x=df, verbose=False)
    params = my_arf.forde()

    cnt = params['cnt']
    forest = params['forest']
    n_real = len(x)

    # Apply ORIGINAL data to forest for correct leaf weights
    orig_leaf_ids = forest.apply(x.reshape(-1, 1))
    query_leaf_ids = forest.apply(t_grid.reshape(-1, 1))
    n_trees = forest.n_estimators
    y = np.zeros(len(t_grid))

    for tree_idx in range(n_trees):
        tree_cnt = cnt[cnt['tree'] == tree_idx].set_index('nodeid')
        q_leaves = query_leaf_ids[:, tree_idx]
        o_leaves = orig_leaf_ids[:, tree_idx]

        # Count original data per leaf
        leaf_counts = {}
        for lid in np.unique(o_leaves):
            leaf_counts[lid] = np.sum(o_leaves == lid)

        for leaf_id in np.unique(q_leaves):
            if leaf_id not in tree_cnt.index:
                continue
            row = tree_cnt.loc[leaf_id]
            mu, sd = float(row['mean']), float(row['sd'])
            lo, hi = float(row['min']), float(row['max'])
            mask = q_leaves == leaf_id

            leaf_weight = leaf_counts.get(leaf_id, 0) / n_real
            if leaf_weight == 0:
                continue

            if np.isinf(lo):
                lo = mu - 10 * sd
            if np.isinf(hi):
                hi = mu + 10 * sd
            if sd < 1e-12:
                sd = 1e-12

            a, b = (lo - mu) / sd, (hi - mu) / sd
            pdf_vals = truncnorm.pdf(t_grid[mask], a, b, loc=mu, scale=sd)
            y[mask] += leaf_weight * pdf_vals

    y /= n_trees
    y = np.maximum(y, 0.0)
    return y


def wrap_transportmaps(x, t_grid):
    """Transport map density estimation (TransportMaps).

    Constructs a monotone polynomial transport map pushing data to
    standard normal, then evaluates the pullback density.
    """
    rho = TM_DIST.StandardNormalDistribution(1)

    # Scale data to [-4, 4] range
    xmin, xmax = x.min(), x.max()
    margin = 0.1 * (xmax - xmin)
    xmin -= margin
    xmax += margin
    a_coeff = 4.0 * (xmin + xmax) / (xmin - xmax)
    b_coeff = 8.0 / (xmax - xmin)
    x_scaled = a_coeff + b_coeff * x

    # Build transport map
    S = TM_MAPS.assemble_IsotropicIntegratedExponentialTriangularTransportMap(
        1, TM_ORDER, 'total')
    pull = TM_DIST.PullBackParametricTransportMapDistribution(S, rho)

    # Fit by minimizing KL divergence
    pi = TM_DIST.DistributionFromSamples(x_scaled.reshape(-1, 1))
    TM_KL.minimize_kl_divergence(
        pi, pull,
        qtype=0, qparams=len(x),
        tol=TM_TOL, ders=2)

    # Evaluate density
    t_scaled = a_coeff + b_coeff * t_grid
    push = TM_DIST.PullBackTransportMapDistribution(S, rho)
    log_pdf = push.log_pdf(t_scaled.reshape(-1, 1))
    y = np.exp(log_pdf) * abs(b_coeff)
    y = np.maximum(y, 0.0)
    return y


# ── F. Method registry ──────────────────────────────────────────────

def build_method_list(include_normflows=True, include_zuko=True,
                      include_gmm=False, include_zuko_unaf=True,
                      include_arf=True, include_transportmaps=True):
    """Build list of (name, wrapper, available) tuples."""
    methods = [
        ('sskernel',        wrap_sskernel,        True),
        ('ssvkernel',       wrap_ssvkernel,       True),
        ('sshist',          wrap_sshist,          True),
        ('KDEpy ISJ',       wrap_kdepy_isj,       HAS_KDEPY),
        ('KDEpy Silverman', wrap_kdepy_silverman, HAS_KDEPY),
        ('scipy kde',       wrap_scipy_kde,       HAS_SCIPY),
        ('statsmodels KDE', wrap_statsmodels_kde, HAS_STATSMODELS),
        ('fastkde',         wrap_fastkde,         HAS_FASTKDE),
        ('KDE diffusion',   wrap_kde_diffusion,   HAS_KDEDIFFUSION),
        ('awkde',            wrap_awkde,            HAS_AWKDE),
    ]
    if include_gmm:
        methods.append(('GMM BIC',         wrap_gmm_bic,         HAS_SKLEARN))
        methods.append(('Bayesian GMM',    wrap_bayesian_gmm,    HAS_SKLEARN))
    tag = _torch_device_tag() if (HAS_NORMFLOWS or HAS_ZUKO) else '(cpu)'
    if include_normflows:
        methods.append((f'Neural Spline Flow {tag}', wrap_nsf, HAS_NORMFLOWS))
    if include_zuko:
        methods.append((f'Zuko NSF {tag}',        wrap_zuko_nsf,        HAS_ZUKO))
    if include_zuko_unaf:
        methods.append((f'Zuko UNAF {tag}', wrap_zuko_unaf, HAS_ZUKO))
    if include_arf:
        methods.append(('ARF', wrap_arf, HAS_ARF))
    if include_transportmaps:
        methods.append(('TransportMap', wrap_transportmaps, HAS_TRANSPORTMAPS))
    return methods


# ── G0. Per-method result caching ─────────────────────────────────────

CACHE_VERSION = 1


def _nan_to_json(v):
    """Convert NaN values in nested dicts/lists to None for JSON."""
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, dict):
        return {k: _nan_to_json(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_nan_to_json(x) for x in v]
    return v


def _json_to_nan(v):
    """Convert None values back to float('nan') after JSON load."""
    if v is None:
        return float('nan')
    if isinstance(v, dict):
        return {k: _json_to_nan(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_to_nan(x) for x in v]
    return v


def _sanitize_method_name(name):
    """Lowercase, replace non-alphanumeric with underscore."""
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def _cache_path(cache_dir, method_name, n_samples, mc_runs):
    """Build cache file path for one method."""
    safe = _sanitize_method_name(method_name)
    return os.path.join(cache_dir, f'{safe}_n{n_samples}_mc{mc_runs}.json')


def save_method_cache(cache_dir, name, method_results, n_samples, mc_runs):
    """Write one method's results to a JSON cache file."""
    os.makedirs(cache_dir, exist_ok=True)
    data = {
        'cache_version': CACHE_VERSION,
        'method_name': name,
        'n_samples': n_samples,
        'mc_runs': mc_runs,
        'n_grid': N_GRID,
        'seed': 42,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'densities': _nan_to_json(method_results),
    }
    path = _cache_path(cache_dir, name, n_samples, mc_runs)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_method_cache(cache_dir, name, n_samples, mc_runs, densities):
    """Load + validate cache for one method. Returns dict or None."""
    path = _cache_path(cache_dir, name, n_samples, mc_runs)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    # Validate metadata
    if data.get('cache_version') != CACHE_VERSION:
        return None
    if data.get('method_name') != name:
        return None
    if data.get('n_samples') != n_samples:
        return None
    if data.get('mc_runs') != mc_runs:
        return None
    if data.get('n_grid') != N_GRID:
        return None

    cached = data.get('densities')
    if not isinstance(cached, dict):
        return None

    # Check all density keys present with correct array lengths
    for d in densities:
        dname = d['name']
        if dname not in cached:
            return None
        entry = cached[dname]
        if not isinstance(entry, dict):
            return None
        for key in ('ise', 'wall_time', 'cpu_time'):
            if key not in entry:
                return None
            if not isinstance(entry[key], list) or len(entry[key]) != mc_runs:
                return None

    return _json_to_nan(cached)


# ── G. Monte Carlo loop ─────────────────────────────────────────────

def run_mc(mc_runs, n_samples, methods, densities, verbose=True,
           cache_dir=None, force_rerun=False, force_methods=None):
    """Run MC simulation. Returns dict of results per method per density.

    If cache_dir is set, loads previously computed per-method results from
    JSON cache files and only re-runs methods that are missing or forced.

    Returns:
        results: {method_name: {density_name: {'ise': [...], 'time': [...]}}}
    """
    if force_methods is None:
        force_methods = set()
    else:
        force_methods = set(force_methods)

    n_densities = len(densities)
    total = n_densities * mc_runs

    # Initialize results storage
    results = {}
    for name, _, _ in methods:
        results[name] = {}
        for d in densities:
            results[name][d['name']] = {'ise': [], 'wall_time': [], 'cpu_time': []}

    # ── Try loading from cache ──
    cached_methods = set()
    if cache_dir and not force_rerun:
        for name, _, avail in methods:
            if not avail:
                continue
            if name in force_methods:
                continue
            cached = load_method_cache(
                cache_dir, name, n_samples, mc_runs, densities)
            if cached is not None:
                results[name] = cached
                cached_methods.add(name)

    # Determine which methods still need computation
    compute_methods = [(n, f, a) for n, f, a in methods
                       if a and n not in cached_methods]

    if verbose and cached_methods:
        print(f'  Cache: loaded {len(cached_methods)} method(s) from '
              f'{cache_dir}')
        for m in sorted(cached_methods):
            print(f'    ✓ {m}')
    if verbose and compute_methods:
        print(f'  Computing {len(compute_methods)} method(s)...')

    # If everything is cached, skip MC loop entirely
    if not compute_methods:
        if verbose:
            print('  All methods loaded from cache — skipping MC loop.')
        return results

    # Master seed sequence for reproducibility
    ss = np.random.SeedSequence(42)
    run_seeds = ss.spawn(mc_runs)

    n_compute = len(compute_methods)
    total_steps = n_densities * mc_runs * n_compute
    pbar = tqdm(total=total_steps, desc='  MC', unit='fit',
                disable=not verbose, dynamic_ncols=True)
    for run_idx in range(mc_runs):
        rng = np.random.default_rng(run_seeds[run_idx])

        for d_idx, density in enumerate(densities):
            # Draw sample (same for all methods this run)
            # NOTE: mw_sample must be called even if some methods are cached,
            # to keep RNG state synchronized across runs.
            x_raw = mw_sample(n_samples, density, rng)
            x_sorted = np.sort(x_raw)

            # Evaluation grid covering true support
            lo, hi = mw_support(density)
            lo = min(lo, x_sorted[0] - 0.5)
            hi = max(hi, x_sorted[-1] + 0.5)
            t_grid = np.linspace(lo, hi, N_GRID)
            f_true = mw_pdf(t_grid, density)

            for name, func, avail in compute_methods:
                pbar.set_postfix_str(
                    f'run {run_idx+1}/{mc_runs}  '
                    f'{density["name"]}  {name}', refresh=True)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        t0_wall = time.perf_counter()
                        t0_cpu = time.process_time()
                        y_est = func(x_sorted, t_grid)
                        elapsed_cpu = time.process_time() - t0_cpu
                        elapsed_wall = time.perf_counter() - t0_wall
                    ise = compute_ise(y_est, t_grid, f_true)
                    results[name][density['name']]['ise'].append(ise)
                    results[name][density['name']]['wall_time'].append(elapsed_wall)
                    results[name][density['name']]['cpu_time'].append(elapsed_cpu)
                except Exception:
                    results[name][density['name']]['ise'].append(np.nan)
                    results[name][density['name']]['wall_time'].append(np.nan)
                    results[name][density['name']]['cpu_time'].append(np.nan)
                pbar.update(1)
    pbar.close()

    # ── Save newly computed methods to cache ──
    if cache_dir:
        for name, _, _ in compute_methods:
            save_method_cache(
                cache_dir, name, results[name], n_samples, mc_runs)
        if verbose:
            print(f'  Cache: saved {len(compute_methods)} method(s) to '
                  f'{cache_dir}')

    return results


# ── H. Aggregation and output ────────────────────────────────────────

def aggregate(results, densities, methods):
    """Compute median ISE, std(ISE), median times per method per density.

    Returns:
        agg: {method: {density: {'median_ise': float, 'std_ise': float,
                                  'median_wall_time': float,
                                  'median_cpu_time': float, 'n_ok': int},
                        '_pooled_median_ise': float}}
    """
    agg = {}
    for name, _, avail in methods:
        if not avail:
            continue
        agg[name] = {}
        all_ise = []
        for d in densities:
            dname = d['name']
            ise_arr = np.array(results[name][dname]['ise'])
            wall_arr = np.array(results[name][dname]['wall_time'])
            cpu_arr = np.array(results[name][dname]['cpu_time'])
            ok = ~np.isnan(ise_arr)
            all_ise.extend(ise_arr[ok].tolist())
            if ok.sum() > 0:
                agg[name][dname] = {
                    'median_ise': float(np.median(ise_arr[ok])),
                    'std_ise': float(np.std(ise_arr[ok])),
                    'median_wall_time': float(np.median(wall_arr[ok])),
                    'median_cpu_time': float(np.median(cpu_arr[ok])),
                    'n_ok': int(ok.sum()),
                }
            else:
                agg[name][dname] = {
                    'median_ise': float('nan'),
                    'std_ise': float('nan'),
                    'median_wall_time': float('nan'),
                    'median_cpu_time': float('nan'),
                    'n_ok': 0,
                }
        agg[name]['_pooled_median_ise'] = (
            float(np.median(all_ise)) if all_ise else float('nan'))
    return agg


def compute_ranks(results, densities, methods):
    """Compute ISE ranks per (run, density) for each method.

    For every MC run and every density, the methods are ranked by their ISE
    (lower = rank 1). This gives mc_runs * n_densities rank samples per method,
    enabling meaningful distributional summaries.

    Returns:
        ranks: {method: list of float ranks}  (length = mc_runs * n_densities)
    """
    method_names = [name for name, _, avail in methods if avail]
    n_methods = len(method_names)
    dnames = [d['name'] for d in densities]

    # Determine number of MC runs from data
    n_runs = len(results[method_names[0]][dnames[0]]['ise'])

    ranks = {m: [] for m in method_names}
    for d_idx, dname in enumerate(dnames):
        for run_idx in range(n_runs):
            # Collect ISE for this (density, run) across all methods
            ise_vals = []
            for m in method_names:
                ise_list = results[m][dname]['ise']
                ise_vals.append(ise_list[run_idx] if run_idx < len(ise_list)
                                else np.nan)
            ise_arr = np.array(ise_vals)

            # Rank: NaN gets worst rank
            order = np.argsort(ise_arr)
            rank_arr = np.empty(n_methods, dtype=float)
            rank_arr[:] = np.nan
            r = 1
            for idx in order:
                if np.isnan(ise_arr[idx]):
                    rank_arr[idx] = n_methods
                else:
                    rank_arr[idx] = r
                    r += 1
            for i, m in enumerate(method_names):
                ranks[m].append(rank_arr[i])
    return ranks


def collect_ise_values(results, densities, methods):
    """Collect all raw ISE values per method across densities and MC runs.

    Returns:
        ise_vals: {method: np.array of ISE values}  (NaNs excluded)
    """
    method_names = [name for name, _, avail in methods if avail]
    dnames = [d['name'] for d in densities]

    ise_vals = {}
    for m in method_names:
        all_ise = []
        for dname in dnames:
            all_ise.extend(results[m][dname]['ise'])
        arr = np.array(all_ise)
        ise_vals[m] = arr[~np.isnan(arr)]
    return ise_vals


def print_tables(agg, ranks, densities, methods, mc_runs, n_samples):
    """Print MISE table, time table, and mean ranks to console."""
    active = [name for name, _, avail in methods if avail]
    dnames = [d['name'] for d in densities]

    col_w = 13
    name_w = max(len(n) for n in active)
    name_w = max(name_w, 22)

    # Header
    print('=' * 80)
    print(f'  MISE Performance Comparison  (mc_runs={mc_runs}, n_samples={n_samples})')
    print(f'  Python {platform.python_version()}, NumPy {np.__version__}')
    print('=' * 80)

    # ── Median ISE table ──
    print()
    print('Median ISE (x1e-3)')
    print(f'  {"Density":<{name_w}}', end='')
    for m in active:
        print(f'  {m:>{col_w}}', end='')
    print()
    print('  ' + '-' * (name_w + len(active) * (col_w + 2)))

    for dname in dnames:
        print(f'  {dname:<{name_w}}', end='')
        for m in active:
            v = agg[m][dname]['median_ise']
            if np.isnan(v):
                print(f'  {"FAIL":>{col_w}}', end='')
            else:
                print(f'  {v*1000:{col_w}.2f}', end='')
        print()

    # Pooled median row
    print('  ' + '-' * (name_w + len(active) * (col_w + 2)))
    print(f'  {"Pooled median":<{name_w}}', end='')
    for m in active:
        med_v = agg[m]['_pooled_median_ise']
        print(f'  {med_v*1000:{col_w}.2f}', end='')
    print()

    # ── Time tables ──
    for time_key, time_label in [('median_wall_time', 'Median wall time (ms)'),
                                  ('median_cpu_time', 'Median CPU time (ms)')]:
        print()
        print(time_label)
        print(f'  {"Density":<{name_w}}', end='')
        for m in active:
            print(f'  {m:>{col_w}}', end='')
        print()
        print('  ' + '-' * (name_w + len(active) * (col_w + 2)))

        for dname in dnames:
            print(f'  {dname:<{name_w}}', end='')
            for m in active:
                v = agg[m][dname][time_key]
                if np.isnan(v):
                    print(f'  {"FAIL":>{col_w}}', end='')
                else:
                    print(f'  {v*1000:{col_w}.1f}', end='')
            print()

        # Mean time row
        print('  ' + '-' * (name_w + len(active) * (col_w + 2)))
        print(f'  {"Mean":<{name_w}}', end='')
        for m in active:
            vals = [agg[m][dn][time_key] for dn in dnames]
            mean_v = np.nanmean(vals)
            print(f'  {mean_v*1000:{col_w}.1f}', end='')
        print()

    # ── Rank summary ──
    print()
    print('MISE Ranks (lower is better)')
    print(f'  {"Method":<{name_w}}  {"Mean":>6}  {"Median":>6}  {"Best":>5}  {"Worst":>5}')
    print('  ' + '-' * (name_w + 30))
    rank_items = [(m, ranks[m]) for m in active]
    rank_items.sort(key=lambda x: np.mean(x[1]))
    for m, r in rank_items:
        r = np.array(r)
        print(f'  {m:<{name_w}}  {np.mean(r):6.2f}  {np.median(r):6.1f}'
              f'  {np.min(r):5.0f}  {np.max(r):5.0f}')
    print()

    # Fairness note
    has_gmm = any('GMM' in m for m in active)
    if has_gmm:
        print('  Note: Marron-Wand densities are Gaussian mixtures by construction.')
        print('  GMM methods fit the correct parametric family and have a structural')
        print('  advantage here that may not generalize to arbitrary densities.')
        print()


def save_json(agg, ranks, results, densities, methods, mc_runs, n_samples,
              out_dir):
    """Save results to JSON, including raw ISE values and density definitions."""
    active = [name for name, _, avail in methods if avail]
    dnames = [d['name'] for d in densities]

    data = {
        'metadata': {
            'mc_runs': mc_runs,
            'n_samples': n_samples,
            'n_grid': N_GRID,
            'python': platform.python_version(),
            'numpy': np.__version__,
            'platform': platform.platform(),
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        },
        'methods': active,
        'densities': dnames,
        'density_defs': [
            {'name': d['name'],
             'weights': d['weights'],
             'means': d['means'],
             'sigmas': d['sigmas']}
            for d in densities
        ],
        'median_ise': {},
        'std_ise': {},
        'median_wall_time': {},
        'median_cpu_time': {},
        'overall_median_ise': {},
        'mean_rank': {},
        'ranks': {},
        'raw_ise': {},
    }

    for m in active:
        data['median_ise'][m] = {dn: agg[m][dn]['median_ise'] for dn in dnames}
        data['std_ise'][m] = {dn: agg[m][dn]['std_ise'] for dn in dnames}
        data['median_wall_time'][m] = {
            dn: agg[m][dn]['median_wall_time'] for dn in dnames}
        data['median_cpu_time'][m] = {
            dn: agg[m][dn]['median_cpu_time'] for dn in dnames}
        data['overall_median_ise'][m] = agg[m]['_pooled_median_ise']
        data['mean_rank'][m] = float(np.mean(ranks[m]))
        data['ranks'][m] = [float(r) for r in ranks[m]]
        data['raw_ise'][m] = {
            dn: [float(v) for v in results[m][dn]['ise']]
            for dn in dnames
        }

    path = os.path.join(out_dir, f'mise_results_n{n_samples}.json')
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f'  Results saved to {path}')
    return path


# ── J. Figures ───────────────────────────────────────────────────────

# Color palette: distinctive colors for methods
METHOD_COLORS = {
    'sskernel':           '#1f77b4',  # blue
    'ssvkernel':          '#ff7f0e',  # orange
    'sshist':             '#9467bd',  # purple
    'KDEpy ISJ':          '#2ca02c',  # green
    'KDEpy Silverman':    '#8c564b',  # brown
    'scipy kde':          '#d62728',  # red
    'statsmodels KDE':    '#e377c2',  # pink
    'fastkde':            '#393b79',  # dark indigo
    'KDE diffusion':      '#637939',  # dark sage
    'GMM BIC':            '#17becf',  # cyan
    'Bayesian GMM':       '#bcbd22',  # olive
    'awkde':              '#843c39',  # dark brick
    'Neural Spline Flow': '#7f7f7f',  # gray
    'Zuko NSF':           '#b5cf6b',  # lime green
    'Zuko UNAF':          '#9edae5',  # light teal
    'ARF':                '#aec7e8',  # light blue
    'TransportMap':       '#c49c94',  # tan
}

METHOD_MARKERS = {
    'sskernel':           'o',
    'ssvkernel':          's',
    'sshist':             '^',
    'KDEpy ISJ':          'D',
    'KDEpy Silverman':    'v',
    'scipy kde':          'P',
    'statsmodels KDE':    'X',
    'fastkde':            'd',
    'KDE diffusion':      '8',
    'GMM BIC':            'h',
    'Bayesian GMM':       'p',
    'awkde':              'H',
    'Neural Spline Flow': '*',
    'Zuko NSF':           '2',
    'Zuko UNAF':          '<',
    'ARF':                '>',
    'TransportMap':       'P',
}


def _method_color(name):
    """Look up color, stripping device tag like '(cpu)' or '(gpu)' if needed."""
    if name in METHOD_COLORS:
        return METHOD_COLORS[name]
    base = name.rsplit(' (', 1)[0]
    return METHOD_COLORS.get(base, '#333333')


def _method_marker(name):
    """Look up marker, stripping device tag like '(cpu)' or '(gpu)' if needed."""
    if name in METHOD_MARKERS:
        return METHOD_MARKERS[name]
    base = name.rsplit(' (', 1)[0]
    return METHOD_MARKERS.get(base, 'o')


def _scatter_speed_on_ax(ax, agg, densities, methods, n_samples, time_key,
                         xlabel, title=None):
    """Draw accuracy-vs-speed scatter on a single axes.

    Parameters
    ----------
    time_key : str
        'median_wall_time' or 'median_cpu_time'
    xlabel : str
        Label for the x-axis.
    title : str, optional
        Panel title.  Defaults to 'n = {n_samples}'.
    """
    active = [name for name, _, avail in methods if avail]
    dnames = [d['name'] for d in densities]

    xs, ys = [], []
    for m in active:
        pooled_ise = agg[m]['_pooled_median_ise']
        mean_time = np.nanmean([agg[m][dn][time_key] for dn in dnames])
        if np.isnan(pooled_ise) or np.isnan(mean_time):
            continue
        xs.append(mean_time * 1000)
        ys.append(pooled_ise)
        ax.scatter(mean_time * 1000, pooled_ise,
                   c=_method_color(m),
                   marker=_method_marker(m),
                   s=120, zorder=5, edgecolors='white', linewidths=0.5)
        ax.annotate(m, (mean_time * 1000, pooled_ise),
                    textcoords='offset points', xytext=(8, 4),
                    fontsize=8, color=_method_color(m))

    ax.set_xscale('log')
    ax.set_yscale('log')
    if xs:
        pad = 2.0
        ax.set_xlim(min(xs) / pad, max(xs) * pad)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Median ISE', fontsize=10)
    ax.set_title(title or f'n = {n_samples}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')

    # Lower-left "better" annotation
    ax.annotate('better', xy=(0.02, 0.02), xycoords='axes fraction',
                fontsize=9, color='gray', fontstyle='italic',
                ha='left', va='bottom')
    ax.annotate('', xy=(0.02, 0.10), xycoords='axes fraction',
                xytext=(0.12, 0.10),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
    ax.annotate('', xy=(0.02, 0.10), xycoords='axes fraction',
                xytext=(0.02, 0.20),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))


def fig_accuracy_vs_speed(agg, densities, methods, out_dir, n_samples):
    """Two-panel scatter: CPU time (left) and wall time (right) vs ISE."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    _scatter_speed_on_ax(axes[0], agg, densities, methods, n_samples,
                         'median_cpu_time',
                         'Median CPU time per fit+evaluate (ms)',
                         title='CPU time')
    _scatter_speed_on_ax(axes[1], agg, densities, methods, n_samples,
                         'median_wall_time',
                         'Median wall time per fit+evaluate (ms)',
                         title='Wall time')

    fig.suptitle(f'Accuracy vs Speed  (n={n_samples})', fontsize=13,
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, 'accuracy_vs_speed.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Figure saved to {path}')


def _heatmap_on_ax(ax, agg, densities, methods, n_samples):
    """Draw MISE heatmap on a single axes using seaborn."""
    active = [name for name, _, avail in methods if avail]
    # Sort methods by pooled median ISE (best first)
    active.sort(key=lambda m: agg[m]['_pooled_median_ise'])
    dnames = [d['name'] for d in densities]
    short_names = [dn.replace('#', '') for dn in dnames]

    mat = np.full((len(active), len(dnames)), np.nan)
    for i, m in enumerate(active):
        for j, dn in enumerate(dnames):
            mat[i, j] = agg[m][dn]['median_ise']

    with np.errstate(divide='ignore', invalid='ignore'):
        log_mat = np.log10(mat)

    # Annotation: MISE x 1000
    annot = np.empty_like(mat, dtype=object)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            annot[i, j] = f'{v*1000:.1f}' if not np.isnan(v) else ''

    sns.heatmap(log_mat, ax=ax, cmap='YlOrRd', annot=annot, fmt='',
                annot_kws={'fontsize': 6}, linewidths=0.3, linecolor='white',
                xticklabels=short_names, yticklabels=active,
                cbar_kws={'label': 'log10(Median ISE)', 'shrink': 0.8})

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right',
                       fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    ax.set_title(f'n = {n_samples}', fontsize=11, fontweight='bold')


def fig_mise_heatmap(agg, densities, methods, out_dir, n_samples):
    """Heatmap: methods x densities, color = log10(MISE)."""
    fig, ax = plt.subplots(figsize=(14, 6))
    _heatmap_on_ax(ax, agg, densities, methods, n_samples)
    fig.suptitle('Median ISE Heatmap (x1e-3)', fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(out_dir, 'mise_heatmap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Figure saved to {path}')


def _violin_on_ax(ax, ise_vals, methods, n_samples):
    """Draw ISE violin plot on a single axes using seaborn (log scale).

    Each violin shows the distribution of log10(ISE) values across MC runs
    and densities. A white horizontal line marks the median inside each violin.
    """
    active = [name for name, _, avail in methods if avail]

    # Sort by median ISE (lower = better, leftmost)
    medians = {m: np.median(ise_vals[m]) for m in active if m in ise_vals
               and len(ise_vals[m]) > 0}
    order = sorted(medians, key=lambda m: medians[m])
    n_methods = len(order)

    # Build long-form lists for seaborn (no pandas needed)
    x_vals = []
    y_vals = []
    for m in order:
        for v in ise_vals[m]:
            if v > 0:
                x_vals.append(m)
                y_vals.append(np.log10(v))

    # Gradient palette: best (green) → worst (red)
    cmap = plt.colormaps['RdYlGn_r']
    palette = {m: cmap(i / max(n_methods - 1, 1)) for i, m in enumerate(order)}

    sns.violinplot(x=x_vals, y=y_vals, hue=x_vals, order=order,
                   palette=palette, hue_order=order, legend=False,
                   inner=None, linewidth=0.8, saturation=0.9,
                   density_norm='width', ax=ax)

    # Overlay white horizontal median line inside each violin
    for i, m in enumerate(order):
        log_vals = np.log10(ise_vals[m][ise_vals[m] > 0])
        med = np.median(log_vals)
        ax.plot([i - 0.15, i + 0.15], [med, med],
                color='white', linewidth=2, solid_capstyle='round', zorder=4)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('')
    ax.set_ylabel('log₁₀(ISE)', fontsize=10)
    ax.set_title(f'n = {n_samples}', fontsize=11, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)


def _add_violin_legend(fig):
    """Add legend explaining median line marker."""
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0.5], color='gray', linewidth=2, label='Median'),
    ]
    fig.legend(handles=legend_elements, loc='upper right',
               fontsize=9, framealpha=0.9, edgecolor='0.7',
               bbox_to_anchor=(0.98, 0.98))


def fig_ise_violin(ise_vals, methods, out_dir, n_samples):
    """Violin plot: distribution of raw ISE values across runs x densities."""
    fig, ax = plt.subplots(figsize=(10, 6))
    _violin_on_ax(ax, ise_vals, methods, n_samples)
    _add_violin_legend(fig)
    fig.suptitle('ISE Distribution  (per run × density, sorted by median)',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, 'ise_violin.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Figure saved to {path}')


# ── J2. Multi-panel figures ──────────────────────────────────────────

def fig_accuracy_vs_speed_multi(all_agg, densities, methods, out_dir,
                                sample_sizes):
    """Grid of scatter panels: rows = sample sizes, columns = CPU / wall time."""
    nrows = len(sample_sizes)
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 7 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    for row, ns in enumerate(sample_sizes):
        _scatter_speed_on_ax(axes[row, 0], all_agg[ns], densities, methods, ns,
                             'median_cpu_time',
                             'Median CPU time per fit+evaluate (ms)',
                             title=f'CPU time (n={ns})')
        _scatter_speed_on_ax(axes[row, 1], all_agg[ns], densities, methods, ns,
                             'median_wall_time',
                             'Median wall time per fit+evaluate (ms)',
                             title=f'Wall time (n={ns})')

    fig.suptitle('Accuracy vs Speed', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, 'accuracy_vs_speed.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Figure saved to {path}')


def fig_mise_heatmap_multi(all_agg, densities, methods, out_dir, sample_sizes):
    """Stacked heatmaps: one row per sample size."""
    nrows = len(sample_sizes)
    fig, axes = plt.subplots(nrows, 1, figsize=(14, 5 * nrows + 1))
    if nrows == 1:
        axes = [axes]

    for ax, ns in zip(axes, sample_sizes):
        _heatmap_on_ax(ax, all_agg[ns], densities, methods, ns)

    fig.suptitle('Median ISE Heatmap (x1e-3)', fontsize=13, fontweight='bold',
                 y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, 'mise_heatmap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Figure saved to {path}')


def fig_ise_violin_multi(all_ise, methods, out_dir, sample_sizes):
    """Stacked ISE violin plots (one row per sample size)."""
    nrows = len(sample_sizes)
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 6 * nrows))
    if nrows == 1:
        axes = [axes]

    for ax, ns in zip(axes, sample_sizes):
        _violin_on_ax(ax, all_ise[ns], methods, ns)

    _add_violin_legend(fig)
    fig.suptitle('ISE Distribution  (per run × density, sorted by median)',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, 'ise_violin.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Figure saved to {path}')


# ── K. CLI ───────────────────────────────────────────────────────────

def estimate_flow_time(sample_sizes, mc_runs, methods):
    """Time a single flow fit per sample size and print projected total."""
    flow_methods = [(n, f) for n, f, a in methods
                    if a and (n.startswith('Neural Spline Flow')
                              or n.startswith('Zuko NSF')
                              or n.startswith('Zuko UNAF')
                              or n == 'ARF'
                              or n == 'TransportMap')]
    if not flow_methods:
        return
    print('  Estimating flow training time...')
    density = MW_DENSITIES[0]  # use Gaussian for timing
    rng = np.random.default_rng(99)
    for ns in sample_sizes:
        x = np.sort(mw_sample(ns, density, rng))
        lo, hi = mw_support(density)
        t_grid = np.linspace(lo, hi, N_GRID)
        for name, func in flow_methods:
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                func(x, t_grid)
            single = time.perf_counter() - t0
            total = single * mc_runs * len(MW_DENSITIES)
            print(f'    {name} n={ns}: {single:.1f}s/fit → '
                  f'~{total/60:.0f} min total '
                  f'({mc_runs} runs × {len(MW_DENSITIES)} densities)')
    print()


def main():
    parser = argparse.ArgumentParser(
        description='MISE performance comparison on Marron-Wand densities')
    parser.add_argument('--mc-runs', type=int, default=50,
                        help='Monte Carlo runs per density (default: 50)')
    parser.add_argument('--n-samples', type=int, nargs='+', default=[1000],
                        help='Sample sizes (default: 1000; use multiple for panels, '
                             'e.g. --n-samples 200 1000)')
    parser.add_argument('--no-figures', action='store_true',
                        help='Skip figure generation')
    parser.add_argument('--no-flow', action='store_true',
                        help='Skip all normalizing flow methods (normflows + Zuko)')
    parser.add_argument('--no-normflows', action='store_true',
                        help='Skip normflows Neural Spline Flow')
    parser.add_argument('--no-zuko', action='store_true',
                        help='Skip Zuko Neural Spline Flow')
    parser.add_argument('--include-gmm', action='store_true',
                        help='Include GMM methods (excluded by default; '
                             'unfair structural advantage on MW densities)')
    parser.add_argument('--no-zuko-unaf', action='store_true',
                        help='Skip Zuko UNAF (UMNN-based flow)')
    parser.add_argument('--no-arf', action='store_true',
                        help='Skip ARF (adversarial random forest)')
    parser.add_argument('--no-transportmaps', action='store_true',
                        help='Skip TransportMap density estimation')
    parser.add_argument('--quick', action='store_true',
                        help='Quick run: 5 MC runs, 200 samples, no slow methods')
    parser.add_argument('--force-rerun', action='store_true',
                        help='Ignore all caches, recompute everything')
    parser.add_argument('--force-method', action='append', default=[],
                        metavar='METHOD',
                        help='Force rerun of specific method(s) by exact name '
                             '(repeatable)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable caching entirely (legacy behavior)')
    args = parser.parse_args()

    if args.quick:
        args.mc_runs = 5
        args.n_samples = [200]
        args.no_flow = True
        args.no_arf = True
        args.no_transportmaps = True

    sample_sizes = args.n_samples

    # Output directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    fig_dir = os.path.join(base_dir, 'fig')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # Cache directory (None disables caching)
    cache_dir = None if args.no_cache else os.path.join(data_dir, 'cache')

    # Verify MW density weights
    for d in MW_DENSITIES:
        s = sum(d['weights'])
        assert abs(s - 1.0) < 1e-12, f"{d['name']}: weights sum to {s}"

    # Build method list
    inc_normflows = not (args.no_flow or args.no_normflows)
    inc_zuko = not (args.no_flow or args.no_zuko)
    inc_zuko_unaf = not (args.no_flow or args.no_zuko_unaf)
    inc_arf = not args.no_arf
    inc_tm = not args.no_transportmaps
    methods = build_method_list(include_normflows=inc_normflows,
                                include_zuko=inc_zuko,
                                include_gmm=args.include_gmm,
                                include_zuko_unaf=inc_zuko_unaf,
                                include_arf=inc_arf,
                                include_transportmaps=inc_tm)

    active = [(n, f, a) for n, f, a in methods if a]
    missing = [n for n, f, a in methods if not a]

    print()
    print(f'  MISE Performance Comparison')
    print(f'  mc_runs={args.mc_runs}, n_samples={sample_sizes}, '
          f'n_grid={N_GRID}')
    print(f'  Methods: {len(active)} active'
          + (f', {len(missing)} skipped ({", ".join(missing)})' if missing else ''))
    print()

    # Flow timing estimate (before committing to full MC loop)
    has_flow = any((n.startswith('Neural Spline Flow')
                    or n.startswith('Zuko NSF')
                    or n.startswith('Zuko UNAF')
                    or n == 'ARF'
                    or n == 'TransportMap') and a
                   for n, _, a in active)
    if has_flow:
        estimate_flow_time(sample_sizes, args.mc_runs, active)

    # Run MC loop for each sample size
    all_agg = {}
    all_ranks = {}
    all_ise = {}
    for n_samples in sample_sizes:
        print(f'  === n_samples = {n_samples} ===')
        t_start = time.time()
        results = run_mc(args.mc_runs, n_samples, active,
                         MW_DENSITIES, verbose=True,
                         cache_dir=cache_dir,
                         force_rerun=args.force_rerun,
                         force_methods=args.force_method)
        elapsed_total = time.time() - t_start
        print(f'  Total MC time: {elapsed_total:.1f}s')
        print()

        agg = aggregate(results, MW_DENSITIES, active)
        ranks = compute_ranks(results, MW_DENSITIES, active)
        ise_vals = collect_ise_values(results, MW_DENSITIES, active)
        all_agg[n_samples] = agg
        all_ranks[n_samples] = ranks
        all_ise[n_samples] = ise_vals

        # Console output
        print_tables(agg, ranks, MW_DENSITIES, active, args.mc_runs, n_samples)

        # JSON output
        save_json(agg, ranks, results, MW_DENSITIES, active, args.mc_runs,
                  n_samples, data_dir)

    # Figures
    if not args.no_figures and HAS_MATPLOTLIB:
        print()
        print('Generating figures...')
        if len(sample_sizes) >= 2:
            # Multi-panel figures (one column per sample size)
            fig_accuracy_vs_speed_multi(
                all_agg, MW_DENSITIES, active, fig_dir, sample_sizes)
            fig_mise_heatmap_multi(
                all_agg, MW_DENSITIES, active, fig_dir, sample_sizes)
            fig_ise_violin_multi(
                all_ise, active, fig_dir, sample_sizes)
        else:
            n_samples = sample_sizes[0]
            fig_accuracy_vs_speed(
                all_agg[n_samples], MW_DENSITIES, active, fig_dir, n_samples)
            fig_mise_heatmap(
                all_agg[n_samples], MW_DENSITIES, active, fig_dir, n_samples)
            fig_ise_violin(
                all_ise[n_samples], active, fig_dir, n_samples)
        print()
    elif args.no_figures:
        print('  Figures skipped (--no-figures)')
    elif not HAS_MATPLOTLIB:
        print('  Figures skipped (matplotlib not available)')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
