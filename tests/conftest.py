"""Shared pytest setup."""
import os

# Tests run on CPU unless the caller sets JAX_PLATFORMS: on a GPU host the
# CUDA plugin would otherwise open every visible card in each worker.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
