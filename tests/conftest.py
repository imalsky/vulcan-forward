"""Shared pytest setup."""

import os

# Tests run on CPU unless the caller sets JAX_PLATFORMS: on a GPU host the
# CUDA plugin would otherwise open every visible card in each worker.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# vulcan3 freezes its network at its first import, and a few tests reach vulcan3 before any model is built (a helper
# that reads its photolysis table, for one). The suite solves one chemistry, so name it here; every profile names the
# same pair, and vulcan_chem.select_network refuses any other.
os.environ.setdefault("VULCAN3_NETWORK", "thermo/SNCHO_photo_network_2025.txt")
os.environ.setdefault("VULCAN3_ATOM_LIST", "H,O,C,N,S")
