# Polarimetric UWB RT Project Structure

- `rt_core/geometry.py`: Plane/material definitions, normals, ray-plane intersections, mirror point utilities.
- `rt_core/rays.py`: Ray data model and reflection-direction transport helper.
- `rt_core/polarization.py`: s/p basis construction, basis projection matrices, Fresnel reflection models (PEC/dielectric), optional depolarization layer.
- `rt_core/antenna.py`: Antenna local frame and H/V or R/L port-basis projection to direction-dependent transverse basis.
- `rt_core/tracer.py`: Deterministic specular path enumeration (0/1/2 bounce), delay and per-frequency 2x2 transfer matrix accumulation, metadata generation.
- `rt_io/hdf5_io.py`: Contracted HDF5 schema for scenarios/cases/paths, full metadata serialization, load/reconstruction, and roundtrip self-test.
- `analysis/ctf_cir.py`: Step-2 basis conversion + CTF synthesis + CIR/PDP extraction with window/nfft/cache options.
- `analysis/xpd_stats.py`: Step-3/4 XPD estimation (tap/path/subband/parity) and conditional Normal(mu,sigma) fitting + JSON export.
- `analysis/path_matching.py`: Optional H/V independent-run path matching and 2x2 matrix reconstruction with tolerance-based fallback.
- `scenarios/C0_free_space.py`, `scenarios/A1_los_min_reflect.py`, `scenarios/A2_pec_plane.py`, `scenarios/A3_dihedral.py`, `scenarios/A4_dielectric.py`, `scenarios/A5_depol_stress.py`: scenario builders with `build_scene`, `build_sweep_params`, `run_case`.
- `scenarios/runner.py`: full sweep execution, HDF5 export, key plot generation, and markdown validation report.
- `plots/p0_p13.py`: P0~P13 plotting utilities (PNG/PDF) and scenario summary charts.
- `tests/*.py`: unit tests for tracer, HDF5 contract, path matching, Step-2 synthesis, and Step-3/4 XPD logic.
