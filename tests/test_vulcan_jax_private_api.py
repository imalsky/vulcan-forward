"""The private VULCAN-JAX surface this engine reaches into must stay callable.

`vulcan_chem.build_chem_model` calls a few PRIVATE methods on
`vulcan_jax.outer_loop.OuterLoop` (leading underscore, no compatibility promise).
That is a deliberate coupling -- the engine needs the runner's own packing and
refresh kernels to seed a carry consistent with what the loop maintains -- but it
means a refactor inside VULCAN-JAX can break this repo without either project's
tests noticing.

That is not hypothetical. VULCAN-JAX dropped the unused `var` parameter from
`OuterLoop._build_refresh_static`, and `vulcan_chem.py` kept passing it, so every
FRESH forward-model run raised

    TypeError: _build_refresh_static() takes 2 positional arguments but 3 given

while cached spectra kept loading. All three sibling suites stayed green: none of
them calls `build_chem_model`, and the single test anywhere that does wraps it in
`except Exception: pytest.skip(...)`, which turns such a TypeError into a pass.

This test needs no data, no network and no solve: it reads the call sites out of
the source and checks each one against the live signature.
"""
from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

vulcan_jax = pytest.importorskip(
    "vulcan_jax", reason="VULCAN-JAX is a hard dependency of this engine")

from vulcan_jax.outer_loop import OuterLoop           # noqa: E402
from vulcan_forward import vulcan_chem                # noqa: E402


def _private_calls_on(receiver: str):
    """Every `<receiver>._name(...)` call in vulcan_chem.py, with its arity."""
    src = pathlib.Path(inspect.getfile(vulcan_chem)).read_text()
    out = []
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr.startswith("_")
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == receiver):
            out.append((node.func.attr, len(node.args), len(node.keywords),
                        node.lineno))
    return out


def test_there_are_private_reach_ins_to_check():
    """Guard the guard: if the AST scan silently matched nothing, every
    assertion below would vacuously pass and this file would be worthless."""
    calls = _private_calls_on("integ")
    assert calls, ("found no integ._private(...) calls in vulcan_chem.py -- "
                   "either the coupling was removed (delete this test) or the "
                   "scan broke (fix it), but do not leave it passing on zero")


@pytest.mark.parametrize("name,npos,nkw,lineno", _private_calls_on("integ"))
def test_private_call_matches_the_live_signature(name, npos, nkw, lineno):
    """Each private method we call must exist and accept the arity we pass."""
    attr = getattr(OuterLoop, name, None)
    if attr is None:
        # Instance attributes assigned in __init__ (e.g. _runner) are not class
        # attributes, so absence here is not automatically a failure. Only flag
        # names that VULCAN-JAX does not define anywhere in the class body.
        src = inspect.getsource(OuterLoop)
        assert f"self.{name}" in src, (
            f"vulcan_chem.py:{lineno} calls integ.{name}(), which "
            f"OuterLoop neither defines as a method nor assigns as an "
            f"attribute. The VULCAN-JAX private API moved under us.")
        return
    if not callable(attr):
        return
    sig = inspect.signature(attr)
    params = [p for n, p in sig.parameters.items() if n != "self"]
    star = any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params)
    positional = [p for p in params
                  if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    required = sum(1 for p in positional if p.default is inspect.Parameter.empty)
    assert star or npos <= len(positional), (
        f"vulcan_chem.py:{lineno} calls integ.{name}() with {npos} positional "
        f"argument(s), but OuterLoop.{name}{sig} accepts at most "
        f"{len(positional)}. This is exactly the _build_refresh_static "
        f"breakage: a VULCAN-JAX signature change that no suite covers.")
    assert npos + nkw >= required or star, (
        f"vulcan_chem.py:{lineno} calls integ.{name}() with {npos}+{nkw} "
        f"argument(s), but OuterLoop.{name}{sig} requires {required}.")
