from .sshist import sshist
from .sskernel import sskernel
from .ssvkernel import ssvkernel
from .classic import sshist_classic, sskernel_classic, ssvkernel_classic

try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version("ssdensity")
except Exception:
    __version__ = "0.0.0.dev0+unknown"

__all__ = ('sshist', 'sskernel', 'ssvkernel',
           'sshist_classic', 'sskernel_classic', 'ssvkernel_classic')
