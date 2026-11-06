"""The chemistry parameter layout and the per-solve diagnostics; ``vulcan_chem`` exports them.

Imported after ``vulcan_chem`` has set the environment and jax x64.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Iterable

    from jax import Array
    from jax.typing import ArrayLike

# Optional chemistry parameters beyond [lnZ, c_o, lnKzz], in their positional order.
CHEM_EXTRAS = ("s_o", "kzz_slope")


def _check_extras(extras: Iterable[str]) -> tuple[str, ...]:
    """Refuse optional entries that are unknown, repeated or out of order."""
    extras = tuple(extras)
    bad = [e for e in extras if e not in CHEM_EXTRAS]
    if bad or len(set(extras)) != len(extras):
        raise ValueError(f"chem_extras {extras!r}: expected a subset of {CHEM_EXTRAS} with no repeats")
    if list(extras) != [e for e in CHEM_EXTRAS if e in extras]:
        raise ValueError(f"chem_extras {extras!r} must keep the order {CHEM_EXTRAS}")
    return extras


class ChemParams(NamedTuple):
    """Named chemistry parameters, a JAX pytree; the positional vector (``params_from_vector``) is also accepted.

    Attributes
    ----------
    lnZ : float
        ln of the multiple of the baseline metallicity.
    c_o : float
        ln of the C/O multiple, at fixed oxygen.
    lnKzz : float
        ln of the eddy-diffusion multiple.
    tp : tuple
        The ``tp_eval`` hook's parameter block, or one uniform temperature offset, K, without a hook.
    s_o : float
        ln of the S/O multiple, on top of lnZ.
    kzz_slope : float
        The Kzz power-law slope.
    """

    lnZ: float | Array = 0.0
    c_o: float | Array = 0.0
    lnKzz: float | Array = 0.0
    tp: tuple[float, ...] | Array = ()
    s_o: float | Array = 0.0
    kzz_slope: float | Array = 0.0

    def to_vector(self, extras: Iterable[str] = ()) -> Array:
        """Return the positional vector ``[lnZ, c_o, lnKzz, *extras, *tp]``.

        Parameters
        ----------
        extras : sequence of str, optional
            The optional entries the layout carries, the model's ``chem_extras``.

        Returns
        -------
        jax.Array
            The vector.
        """
        extras = _check_extras(extras)
        return jnp.asarray(
            [self.lnZ, self.c_o, self.lnKzz, *(getattr(self, e) for e in extras), *self.tp], dtype=jnp.float64
        )


def params_from_vector(
    theta: ArrayLike, n_tp_params: int, *, has_tp_eval: bool, extras: Iterable[str] = ()
) -> ChemParams:
    """Read a positional ``theta`` into ``ChemParams``.

    ``theta[0:3]`` is ``[lnZ, c_o, lnKzz]``, then the entries named by ``extras``, then the temperature block.

    Parameters
    ----------
    theta : array-like
        The vector.
    n_tp_params : int
        Temperature entries when the model has a ``tp_eval`` hook.
    has_tp_eval : bool
        Whether it has one; without one the block is a single uniform offset.
    extras : sequence of str, optional
        The optional entries the layout carries, in ``CHEM_EXTRAS`` order.

    Returns
    -------
    ChemParams
        The named parameters.
    """
    extras = _check_extras(extras)
    v = jnp.asarray(theta, dtype=jnp.float64)
    k = 3 + len(extras)
    tp = v[k : k + n_tp_params] if has_tp_eval else v[k : k + 1]
    kw = {e: v[3 + i] for i, e in enumerate(extras)}
    return ChemParams(lnZ=v[0], c_o=v[1], lnKzz=v[2], tp=tp, **kw)


class ConvDiag(NamedTuple):
    """Per-solve convergence diagnostics read off the runner's final carry.

    Every field rides the primal carry, so reading them is free inside a jvp chain. ``longdy`` and ``longdydt`` carry a
    tangent, so callers on an AD path ``stop_gradient`` them.

    ``accept_count`` alone is not a convergence test: the runtime budget or a non-finite state can end the runner well
    under the cap. ``conv_normal`` is the runner's two-branch certification (tight yconv_cri/slope_cri or loose
    yconv_min/slope_min, and the photo-flux gate, the geometry term and the column element budget) recomputed at the
    exit state.

    Attributes
    ----------
    accept_count : jax.Array, int32
        Accepted steps taken.
    longdy : jax.Array
        The runner's convergence metric at exit.
    longdydt : jax.Array
        ``longdy`` per lookback time.
    conv_normal : jax.Array, bool
        The certification at exit.
    aflux_change : jax.Array
        Maximum relative actinic-flux change at exit.
    conv_branch : jax.Array, int32
        Certified branch: 1 tight, 2 loose, 0 not certified.
    cell_species : jax.Array, int32
        Species index of the cell that sets ``longdy``.
    cell_layer : jax.Array, int32
        Layer index of that cell, 0 at the bottom.
    cell_vmr : jax.Array
        Mixing ratio of that cell at exit.
    t : jax.Array
        Integration time at exit, s.
    dt : jax.Array
        Step size at exit, s.
    tangent_longdy : jax.Array
        ``converged_y_jvp`` only: the tangent's ``longdy`` at exit; NaN on the primal path.
    """

    accept_count: jnp.ndarray
    longdy: jnp.ndarray
    longdydt: jnp.ndarray
    conv_normal: jnp.ndarray
    aflux_change: jnp.ndarray
    conv_branch: jnp.ndarray
    cell_species: jnp.ndarray
    cell_layer: jnp.ndarray
    cell_vmr: jnp.ndarray
    t: jnp.ndarray
    dt: jnp.ndarray
    tangent_longdy: jnp.ndarray
