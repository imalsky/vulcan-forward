"""vulcan-forward: the VULCAN 3 chemistry to ExoJAX spectrum forward engine, differentiable end to end.

Import ``vulcan_chem`` before anything from exojax: it sets the import-frozen ``VULCAN3_*`` variables and jax x64, and
raises if exojax or a conflicting vulcan3 got there first::

    from vulcan_forward import constants, vulcan_chem, interp_map, exojax_rt

Data is never bundled: set ``$VULCAN_FORWARD_DATA`` or call ``paths.set_data_root``. Modules, methods and sources are in
docs/method.md.
"""

from vulcan_forward._version import __version__

__all__ = ["__version__"]
