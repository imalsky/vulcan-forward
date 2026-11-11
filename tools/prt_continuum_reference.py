"""Write the petitRADTRANS references with the continuum on: tests/data/prt_ref_cont_*.npz.

Three cases on H2O-poor columns, where the continuum sets the windows: transmission with H2-H2 and H2-He CIA,
transmission with CIA and H2 and He Rayleigh, and emission with CIA. Both codes read the same files: the engine's
ExoMolOP H2O k-table and its HITRAN 2011 CIA tables, which this script converts into pRT's CIA format inside a private
input_data tree (pRT ships other CIA tables). Rayleigh is each code's own cross-section. Each fixture stores the column,
pRT's spectrum with and without the continuum, and its provenance. Two steps, from the repository root:

    <pRT-python> tools/prt_continuum_reference.py prt <work-dir>
    <engine-python> tools/prt_continuum_reference.py stats

The first runs pRT 3.4.0 in its own environment, which has no vulcan_forward, and reads the data root from
$VULCAN_FORWARD_DATA. The second measures the agreement through the test body in tests/test_e2e_rt_reference.py and
records it, with the tolerances at three times the achieved agreement.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "data"
H2O = "1H2-16O__POKAZATEL"
CIA = {"H2--H2": "H2-H2_2011.cia", "H2--He": "H2-He_2011.cia"}
# pRT finds a CIA table by its directory and name pattern; the names say what the file holds.
CIA_PRT_NAME = {
    "H2--H2": "H2--H2-NatAbund__HITRAN2011.DeltaWavenumber1_1-500mu",
    "H2--He": "H2--He-NatAbund__HITRAN2011.DeltaWavenumber1_0.5-500mu",
}
HE_OVER_H2 = 0.19384  # the He/H2 ratio of the WASP-39b fixture's column (Tsai et al. 2023)
MOLMASS = {"H2": 2.016, "He": 4.0026, "H2O": 18.015}  # g/mol, for the column's mean molecular weight
TRANS = {
    "nlayer": 100,
    "p_top_bar": 1.0e-6,
    "p_btm_bar": 1.0e2,
    "p_ref_bar": 0.8995,
    "rp_cm": 9.1438268e9,
    "gs_cgs": 426.0,
    "rstar_cm": 0.932 * 6.957e10,
    "T_top_K": 800.0,
    "T_btm_K": 1800.0,
    "vmr_h2o": 1.0e-5,
    "wl_um": [1.0, 5.0],
    "variable_gravity": True,
}
# pRT's emission runs at constant g: the engine emulates it with rp_cm = 100 R_Jup, where the hydrostatic
# g-variation over the column is below 1e-4 relative, as in the lines-only emission fixture.
EMIS = {
    "nlayer": 80,
    "p_top_bar": 1.0e-6,
    "p_btm_bar": 1.0e2,
    "p_ref_bar": 1.0e-3,
    "rp_cm": 100 * 7.1492e9,
    "gs_cgs": 1.0e3,
    "rstar_cm": 0.932 * 6.957e10,
    "T_top_K": 900.0,
    "T_btm_K": 2200.0,
    "vmr_h2o": 1.0e-4,
    "wl_um": [1.0, 5.0],
    "variable_gravity": False,
}
# fixture -> (observable, column, Rayleigh on)
CASES = {
    "prt_ref_cont_cia_trans.npz": ("transmission", TRANS, False),
    "prt_ref_cont_cia_rayleigh_trans.npz": ("transmission", TRANS, True),
    "prt_ref_cont_cia_emission.npz": ("emission", EMIS, False),
}
ESTIMATOR = (
    "r = engine/interp(pRT) over the mask; mean_ratio=mean(r); rms_pct=std(r/mean-1)*100; "
    "max_dev_pct=max|r/mean-1|*100; max_abs_dev_from_unity_pct=max|r-1|*100"
)
TOL_FACTOR = 3.0
# mean_ratio_abs bounds the drift of the mean from its recorded value, with the floor of the lines-only fixtures
MEAN_TOL_FLOOR = 1.0e-4


def column(cfg: dict) -> dict:
    """Return the column on pRT's pressure grid: T linear in log10 P, H2O fixed, H2 and He in a fixed ratio."""
    p = np.logspace(np.log10(cfg["p_top_bar"]), np.log10(cfg["p_btm_bar"]), cfg["nlayer"])
    x = (np.log10(p) - np.log10(cfg["p_top_bar"])) / np.log10(cfg["p_btm_bar"] / cfg["p_top_bar"])
    h2 = (1.0 - cfg["vmr_h2o"]) / (1.0 + HE_OVER_H2)
    mmw = h2 * MOLMASS["H2"] + h2 * HE_OVER_H2 * MOLMASS["He"] + cfg["vmr_h2o"] * MOLMASS["H2O"]
    return {
        "p_bar": p,
        "T_K": cfg["T_top_K"] + (cfg["T_btm_K"] - cfg["T_top_K"]) * x,
        "vmr_h2o": np.full(p.size, cfg["vmr_h2o"]),
        "vmr_h2": np.full(p.size, h2),
        "vmr_he": np.full(p.size, h2 * HE_OVER_H2),
        "mmw": np.full(p.size, mmw),
    }


def _read_hitran_cia(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return wavenumber (cm^-1), temperature (K) and k (cm^5 molecule^-2, [T, nu]) of a rectangular HITRAN file."""
    lines = path.read_text().splitlines()
    nnu = int(lines[0].split()[3])
    blocks = [lines[i : i + nnu + 1] for i in range(0, len(lines), nnu + 1)]
    temps = np.array([float(b[0].split()[4]) for b in blocks])
    data = np.array([[row.split() for row in b[1:]] for b in blocks], dtype=float)
    if not np.all(data[:, :, 0] == data[0, :, 0]):
        raise ValueError(f"{path}: the temperature blocks have different wavenumber grids")
    return data[0, :, 0], temps, data[:, :, 1]


def build_input_data(work: Path, data_root: Path) -> dict:
    """Build pRT's input_data tree in ``work``: the engine's H2O k-table and its CIA tables in pRT's format."""
    from petitRADTRANS import physical_constants as cst
    from petitRADTRANS.__file_conversion import write_cia_opacities
    from petitRADTRANS.chemistry.utils import get_species_molar_mass
    from petitRADTRANS.opacities.opacities import CIAOpacity, CorrelatedKOpacity
    from petitRADTRANS.radtrans import Radtrans

    kdir = work / "opacities" / "lines" / "correlated_k" / "H2O" / "1H2-16O"
    kdir.mkdir(parents=True, exist_ok=True)
    ktable = data_root / "exomolop" / "H2O.ktable.h5"
    link = kdir / f"{H2O}.R1000_0.3-50mu.ktable.petitRADTRANS.h5"
    if not link.exists():
        link.symlink_to(ktable)
    CorrelatedKOpacity.find(species=H2O, path_input_data=str(work), search_online=False)
    record = {"H2O": {"file": ktable.name, "sha256": hashlib.sha256(ktable.read_bytes()).hexdigest()}}
    for pair, name in CIA.items():
        src = data_root / "opacity_cache" / name
        nu, temps, k = _read_hitran_cia(src)
        cdir = work / "opacities" / "continuum" / "collision_induced_absorptions" / pair / f"{pair}-NatAbund"
        cdir.mkdir(parents=True, exist_ok=True)
        out = cdir / f"{CIA_PRT_NAME[pair]}.ciatable.petitRADTRANS.h5"
        # pRT's alpha is per amagat^2: its _interpolate_cia divides by the Loschmidt number squared
        write_cia_opacities(str(out), pair.split("--"), nu, 1.0 / nu[::-1], k * cst.L0**2, temps, description=name)
        found = CIAOpacity.find(species=pair, path_input_data=str(work), search_online=False)
        if Path(found).resolve() != out.resolve():
            raise RuntimeError(f"pRT resolves {pair} to {found}, not {out}")
        # Round trip at a table node: pRT's CIA opacity per gram must be k n_a n_b / rho.
        mols, weight, lam, t_grid, a_grid = Radtrans.load_cia_opacity(str(out))
        it, inu = temps.size // 2, nu.size // 3
        p_cgs, T, mmw, mf = np.array([1.0e6]), np.array([temps[it]]), np.array([2.3]), (0.5, 0.25)
        kappa = Radtrans._interpolate_cia(
            {"molecules": mols, "weight": weight, "lambda": lam, "temperature": t_grid, "alpha": a_grid},
            mf[0] * mf[1],
            p_cgs,
            T,
            np.array([cst.c * nu[inu]]),
            mmw,
        )
        rho = p_cgs * mmw * cst.amu / (cst.kB * T)
        n_a, n_b = (f * rho / (get_species_molar_mass(m) * cst.amu) for f, m in zip(mf, mols, strict=True))
        want = k[it, inu] * n_a * n_b / rho
        if not np.allclose(kappa.ravel(), want, rtol=1e-9):
            raise RuntimeError(f"{pair}: pRT's CIA opacity {kappa.ravel()} is not k n_a n_b / rho = {want}")
        record[pair] = {"file": name, "sha256": hashlib.sha256(src.read_bytes()).hexdigest(), "prt_file": out.name}
    return record


def run_prt(work: Path, kind: str, cfg: dict, col: dict, continuum: bool, rayleigh: bool) -> dict:
    """Return pRT's spectrum on the column, sorted by wavelength: depth (ppm) or flux per cm^-1 and per cm."""
    from petitRADTRANS.chemistry.utils import get_species_molar_mass
    from petitRADTRANS.radtrans import Radtrans

    rt = Radtrans(
        pressures=col["p_bar"],
        line_species=[H2O],
        rayleigh_species=["H2", "He"] if rayleigh else [],
        gas_continuum_contributors=list(CIA) if continuum else [],
        cloud_species=[],
        wavelength_boundaries=cfg["wl_um"],
        line_opacity_mode="c-k",
        scattering_in_emission=False,
        path_input_data=str(work),
    )
    # Mass fractions with pRT's own molar masses, so its number densities are vmr * n, as the engine's.
    mf = {H2O: col["vmr_h2o"] * get_species_molar_mass(H2O) / col["mmw"]}
    mf["H2"] = col["vmr_h2"] * get_species_molar_mass("H2") / col["mmw"]
    mf["He"] = col["vmr_he"] * get_species_molar_mass("He") / col["mmw"]
    if kind == "transmission":
        wl, rad, _ = rt.calculate_transit_radii(
            temperatures=col["T_K"],
            mass_fractions=mf,
            mean_molar_masses=col["mmw"],
            reference_gravity=cfg["gs_cgs"],
            planet_radius=cfg["rp_cm"],
            reference_pressure=cfg["p_ref_bar"],
            variable_gravity=cfg["variable_gravity"],
        )
        o = np.argsort(wl)
        return {"wl_um": wl[o] * 1e4, "depth_ppm": (rad[o] / cfg["rstar_cm"]) ** 2 * 1e6}
    wl, fl, _ = rt.calculate_flux(
        temperatures=col["T_K"],
        mass_fractions=mf,
        mean_molar_masses=col["mmw"],
        reference_gravity=cfg["gs_cgs"],
        frequencies_to_wavelengths=True,
    )
    o = np.argsort(wl)
    nutilde = 1.0 / wl[o]  # cm^-1
    return {"wl_um": wl[o] * 1e4, "flux_lambda": fl[o], "flux_per_cm1": fl[o] / nutilde**2}


def write_prt(work: Path) -> None:
    """Run every case in pRT and write the fixtures without their statistics."""
    data_root = Path(os.environ["VULCAN_FORWARD_DATA"])
    files = build_input_data(work, data_root)
    text = Path(__file__).read_text()
    for name, (kind, cfg, rayleigh) in CASES.items():
        col = column(cfg)
        full = run_prt(work, kind, cfg, col, continuum=True, rayleigh=rayleigh)
        lines = run_prt(work, kind, cfg, col, continuum=False, rayleigh=False)
        arrays = {"wl_um": full["wl_um"], **col}
        if kind == "transmission":
            arrays |= {"prt_depth_ppm": full["depth_ppm"], "prt_depth_lines_ppm": lines["depth_ppm"]}
        else:
            arrays |= {
                "prt_flux_per_cm1": full["flux_per_cm1"],
                "prt_flux_lambda": full["flux_lambda"],
                "prt_flux_lines_per_cm1": lines["flux_per_cm1"],
            }
        meta = {
            "case": name.removeprefix("prt_ref_").removesuffix(".npz"),
            "prt_version": version("petitRADTRANS"),
            "generated": dt.datetime.now(dt.timezone.utc).date().isoformat(),
            "opacity": files,
            "config": {
                **cfg,
                "observable": kind,
                "continuum": "H2-H2 and H2-He CIA" + (", H2 and He Rayleigh" if rayleigh else ", no Rayleigh"),
                "use_rayleigh": rayleigh,
                "he_over_h2": HE_OVER_H2,
                "mask": "wl > lo*1.01 and wl < hi*0.99",
            },
            "cia": "both codes: HITRAN 2011 (Richard et al. 2012) H2-H2 and H2-He, the engine's files, converted to "
            "pRT's format (k * Loschmidt^2, cm^-1 amagat^-2); pRT interpolates log10 alpha linearly in T and alpha "
            "linearly in wavelength, exojax log10 k linearly in T at the table point above each band centre",
            "rayleigh": "pRT 3.4.0 built-in H2 (Dalgarno & Williams 1962) and He cross-sections; the engine exojax "
            "polarizabilities (H2 0.804e-24, He 0.21e-24 cm^3) with King factor 1.0, a pure lambda^-4 form"
            if rayleigh
            else "off in both codes",
            "scattering": "off in emission in both codes" if kind == "emission" else "Rayleigh as extinction",
            "estimator": ESTIMATOR,
            "prt_script": "tools/prt_continuum_reference.py prt <work-dir>",
            "prt_script_text": text,
            "exo_method": "build_rt_model and build_emis_model public API, through the test body "
            "(tests/test_e2e_rt_reference.py, _continuum_ratio); stats recorded by "
            "tools/prt_continuum_reference.py stats",
            "bundle_record": "none: first generated by tools/prt_continuum_reference.py, no earlier A/B bundle",
        }
        if kind == "emission":
            meta["units"] = (
                "prt_flux_per_cm1 = pRT calculate_flux(frequencies_to_wavelengths=True) output DIVIDED by "
                "nu_tilde^2: pRT returns flux per cm of wavelength, this engine per cm^-1. prt_flux_lambda carries "
                "the original array."
            )
        meta_bytes = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
        np.savez_compressed(OUT / name, meta=meta_bytes, **arrays)
        print("wrote", OUT / name)


def write_stats() -> None:
    """Measure each fixture's agreement through the test body and record it with its tolerances."""
    spec = importlib.util.spec_from_file_location("e2e", REPO / "tests" / "test_e2e_rt_reference.py")
    test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test)
    for name in CASES:
        r = test._continuum_ratio(name)
        m = float(np.mean(r))
        stats = {
            "mean_ratio": m,
            "rms_pct": float(np.std(r / m - 1.0) * 100.0),
            "max_dev_pct": float(np.max(np.abs(r / m - 1.0)) * 100.0),
            "max_abs_dev_from_unity_pct": float(np.max(np.abs(r - 1.0)) * 100.0),
        }
        tol = {k: TOL_FACTOR * v for k, v in stats.items() if k != "mean_ratio"}
        tol["mean_ratio_abs"] = TOL_FACTOR * abs(m - 1.0) + MEAN_TOL_FLOOR
        z = dict(np.load(OUT / name))
        meta = json.loads(bytes(z["meta"]))
        meta |= {"stats": stats, "tol": tol}
        z["meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
        np.savez_compressed(OUT / name, **z)
        print(name, json.dumps(stats))


if __name__ == "__main__":
    if sys.argv[1:2] == ["prt"] and len(sys.argv) == 3:
        write_prt(Path(sys.argv[2]))
    elif sys.argv[1:] == ["stats"]:
        write_stats()
    else:
        raise SystemExit(__doc__)
