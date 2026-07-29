"""vulcan-forward: the shared VULCAN-JAX -> ExoJAX forward-model engine.

Chains live VULCAN-JAX photochemical kinetics into an ExoJAX transmission or
emission spectrum, differentiably. This package is the single implementation
shared by the retrieval framework (vulcan-retrieval) and the JWST observation
planner (vulcan-jwst-tool); neither depends on the other.

    constants    shared physics constants + the default molecule/opacity table
    paths        the external data-root contract (line lists, opacity cache, CIA)
    vulcan_chem  chemistry driver: theta -> converged VMR (jvp-differentiable)
    interp_map   chemistry grid -> RT grid log-pressure interpolation
    exojax_rt    opacities + CIA + ArtTransPure/ArtEmisPure -> depth or flux

IMPORT ORDER IS LOAD-BEARING. ``vulcan_chem`` must be imported before anything
from exojax: it sets the VULCAN_JAX_* import-frozen env vars and enables jax
x64, both of which are read once at first import. It raises if exojax (or a
conflicting vulcan_jax) got there first rather than producing a subtly wrong
model, so the canonical order is:

    from vulcan_forward import constants, vulcan_chem, interp_map, exojax_rt

Data is never bundled: line lists, the offline opacity cache, and the two CIA
tables are tens of gigabytes. Point the engine at them with
$VULCAN_FORWARD_DATA or ``paths.set_data_root(...)``; nothing touches the
filesystem until a path is actually needed, and then it fails loudly with the
offending value and the remedy.
"""
from vulcan_forward._version import __version__

__all__ = ["__version__"]
