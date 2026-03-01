# ssdensity

Optimal fixed or locally adaptive kernel density estimation for 1-D data.

> Forked from [AdaptiveKDE](https://github.com/shimazaki/AdaptiveKDE)
> ([PyPI: adaptivekde](https://pypi.org/project/adaptivekde/)) with
> significant performance optimizations. Original (unoptimized) code
> is preserved in the `classic` subpackage.

## Installation

```
pip install ssdensity
```

Or from source:

```
pip install -e .
```

Requires Python >= 3.9 and NumPy >= 1.24.

## Quick start

```python
import numpy as np
from ssdensity import sshist, sskernel, ssvkernel

x = np.concatenate([np.random.randn(300) - 1, np.random.randn(300) + 1])

optN, optD, edges, C, N = sshist(x)
y, t, optw, W, C, confb95, yb = sskernel(x)
y, t, optw, gs, C, confb95, yb = ssvkernel(x)
```

## Methods

1. **sshist** -- optimal number of histogram bins, minimizing the L2 norm
   between the histogram and the underlying distribution.
2. **sskernel** -- kernel density estimation with a single
   globally-optimized bandwidth.
3. **ssvkernel** -- kernel density estimation with a locally variable
   bandwidth.

Each function also has a `_classic` variant that preserves the original,
unoptimized reference implementation with identical signatures and return
values:

```python
from ssdensity import sshist_classic, sskernel_classic, ssvkernel_classic
```

The optimized versions are the default and recommended for normal use.
The `_classic` variants are provided for reference and reproducibility.

## Benchmark

![Marron-Wand benchmark](tests/compare_methods_optimized.png)

Comparison of all three methods on the 15 Marron-Wand densities (1000 samples
each). See `tests/` for the benchmark notebook.

## Tutorial

Papers and slides are available in the `tutorial/` directory.

## References

- H. Shimazaki and S. Shinomoto, "A method for selecting the bin size of
  a time histogram," *Neural Computation* 19(6): 1503-1527, 2007.
  [doi:10.1162/neco.2007.19.6.1503](https://doi.org/10.1162/neco.2007.19.6.1503)

- H. Shimazaki and S. Shinomoto, "Kernel bandwidth optimization in spike
  rate estimation," *Journal of Computational Neuroscience* 29(1-2):
  171-182, 2010.
  [doi:10.1007/s10827-009-0180-4](https://doi.org/10.1007/s10827-009-0180-4)

## Authors

- Hideaki Shimazaki (shimazaki.hideaki.8x@kyoto-u.jp) -- [shimazaki](https://github.com/shimazaki) on GitHub
- Lee A.D. Cooper (cooperle@gmail.com) -- [cooperlab](https://github.com/cooperlab) on GitHub
- Subhasis Ray (ray.subhasis@gmail.com)

## License

Apache License 2.0. See [LICENSE.txt](LICENSE.txt) for details.
