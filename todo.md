# todo

Open, owed or undecided work for vulcan-forward; the evidence is in `notes.md`.

1. [owed] Release 0.30.0 (the v5 engine options) to TestPyPI and tag it: vulcan-retrieval floors `vulcan-forward>=0.30.0`.

## 1. Release 0.30.0

Evidence: notes "Current state" (0.30.0 on main: the sulfur knob and the Kzz power
law as optional entries of the chemistry layout, `lnM` and the grey deck in the RT).
The version is `src/vulcan_forward/_version.py`; the floors in `pyproject.toml` are
unchanged. vulcan-retrieval's `requirements-hpc.txt` pins a commit, so the NAS env
needs the pushed commit, not the release; check what vulcan-retrieval's CI installs
before relying on the release there. Release with the `vulcan-release` skill after the
push. Done when the tag and the TestPyPI release exist and vulcan-retrieval's
`validate_env` passes against the installed 0.30.0.
