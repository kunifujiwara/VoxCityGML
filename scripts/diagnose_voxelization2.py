"""Voxelization unevenness diagnosis, v2.

v1's slice-based truth oracle failed (subdivided soup does not close into
loops), so truth is now a MeshLib winding-number SDF computed on a fine
lattice whose ORIGIN I CONTROL -- robust to triangle soup and, by
construction, immune to the phase bug under test.

Adds a calibration step: MeshLib's sampling convention (cell centre vs cell
corner) is measured on a synthetic box before anything depends on it.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import binary_fill_holes
from scipy.spatial import cKDTree

import meshlib.mrmeshpy as mr
import meshlib.mrmeshnumpy as mrnp

import os

# Data artefacts (plateau_raw.pkl, kanda_city.pkl) are NOT in the repo -- they
# are multi-MB pickles from the CityMesher diagnosis session.  Point
# VOXCITYGML_DIAG_DATA at the directory holding them.
SP = Path(os.environ.get(
    "VOXCITYGML_DIAG_DATA",
    r"C:\Users\kunih\AppData\Local\Temp\claude\c--Users-kunih-OneDrive-00-Codes-python-CityMesher\c28d7136-0bdf-4af2-a57f-cf6510459177\scratchpad"))
OUT = Path(os.environ.get("VOXCITYGML_DIAG_OUT", str(SP)))
MS = 2.0
FINE = 0.25

raw = pickle.loads((SP / "plateau_raw.pkl").read_bytes())
polys = raw["polys"]
city = pickle.loads((SP / "kanda_city.pkl").read_bytes())
vox = np.asarray(city.voxels.classes)
rows, cols, nz = vox.shape
dem_n = np.nan_to_num(np.asarray(city.dem.elevation, float)
                      - np.nanmin(city.dem.elevation))
NON_GROUND = {0, -3, -2}
k_ground = np.zeros((rows, cols), int)
for r in range(rows):
    for c in range(cols):
        k = 0
        while k < nz and int(vox[r, c, k]) not in NON_GROUND:
            k += 1
        k_ground[r, c] = k
offset = k_ground * MS - dem_n
solid = vox == -3
report = {}


def sdf_volume(verts, faces, origin, dims, vs):
    """Winding-number SDF on a lattice at MY origin.  Returns bool inside[]."""
    v64 = np.ascontiguousarray(verts, np.float64)
    shift = v64.min(axis=0)
    ml = mrnp.meshFromFacesVerts(np.ascontiguousarray(faces, np.int32),
                                 (v64 - shift).astype(np.float32))
    p = mr.MeshToDistanceVolumeParams()
    o = np.asarray(origin, float) - shift
    p.vol.origin = mr.Vector3f(o[0], o[1], o[2])
    p.vol.voxelSize = mr.Vector3f.diagonal(vs)
    p.vol.dimensions = mr.Vector3i(int(dims[0]), int(dims[1]), int(dims[2]))
    p.dist.signMode = mr.SignDetectionMode.HoleWindingRule
    p.dist.maxDistSq = (8 * vs) ** 2
    sv = mr.meshToDistanceVolume(mr.MeshPart(ml), p)
    sdf = mrnp.getNumpy3Darray(sv)                 # (nx, ny, nz)
    return binary_fill_holes(np.nan_to_num(sdf, nan=1.0) <= 0.0)


# ---- calibration: does MeshLib sample at origin+(i+0.5)vs or origin+i*vs? --
box = trimesh.creation.box(extents=[10, 10, 10])
box.apply_translation([5, 5, 5])                   # solid occupies [0,10]^3
ins = sdf_volume(box.vertices, box.faces, origin=(-3.0, -3.0, -3.0),
                 dims=(17, 17, 17), vs=1.0)
ix = np.nonzero(ins.any(axis=(1, 2)))[0]
# centre convention: samples at -3+(i+0.5) -> inside for i in [3..12]
# corner convention: samples at -3+i     -> inside for i in [3..13]
conv = "centre" if (ix.min(), ix.max()) == (3, 12) else (
    "corner" if (ix.min(), ix.max()) == (3, 13) else f"odd {ix.min()},{ix.max()}")
print(f"calibration: MeshLib SDF samples at cell {conv} "
      f"(inside x-indices {ix.min()}..{ix.max()})")
report["sampling_convention"] = conv
HALF = 0.5 if conv == "centre" else 0.0


class Oracle:
    """Fine winding SDF over a window, aligned to the scene lattice."""

    def __init__(self, verts, faces, lo, hi):
        self.lo = np.floor(np.asarray(lo) / MS) * MS - 2 * MS
        hi = np.ceil(np.asarray(hi) / MS) * MS + 2 * MS
        self.dims = np.ceil((hi - self.lo) / FINE).astype(int) + 1
        self.inside = sdf_volume(verts, faces, self.lo, self.dims, FINE)

    def contains(self, pts):
        idx = np.floor((np.asarray(pts) - self.lo) / FINE - HALF + 0.5)
        idx = idx.astype(int)
        ok = np.all((idx >= 0) & (idx < self.dims), axis=1)
        out = np.zeros(len(pts), bool)
        out[ok] = self.inside[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
        return out


# ---- pick tall interior buildings; mask cells contested by neighbours ------
from collections import defaultdict
groups = defaultdict(list)
for o in polys:
    groups[o.object_id].append(o)
builds, bb = {}, {}
for oid, os_ in groups.items():
    t = np.concatenate([o.vertices[o.faces] for o in os_])
    v = t.reshape(-1, 3)
    builds[oid] = (v, np.arange(len(v)).reshape(-1, 3))
    bb[oid] = (v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max(),
               v[:, 2].min(), v[:, 2].max())
cands = sorted(((b[5] - b[4], oid) for oid, b in bb.items()
                if b[0] > 12 and b[1] < 188 and b[2] > 12 and b[3] < 188),
               reverse=True)
picks = [oid for _, oid in cands[:6]]
print(f"analysing {len(picks)} tall interior buildings")

# surface samples of OTHER buildings, for contested-cell masking
samples = {}
for oid in picks:
    others = np.concatenate([builds[o][0][builds[o][1]].reshape(-1, 3)
                             for o in builds if o != oid])
    m = trimesh.Trimesh(others, np.arange(len(others)).reshape(-1, 3),
                        process=False)
    pts, _ = trimesh.sample.sample_surface(m, 400_000)
    samples[oid] = cKDTree(pts)

# ---- voxcitygml pipeline pieces -------------------------------------------
import sys
sys.path.insert(0, r"C:\Users\kunih\OneDrive\00_Codes\python\VoxCityGML")
from voxcitygml.voxelizer3d import (Grid3DParams, _overlay_surface_shell,
                                    _voxelize_meshlib_levelset,
                                    _voxelize_meshlib_winding,
                                    _voxelize_building_solid)
from voxcitygml.watertight import make_watertight_mesh

results = {}
for oid in picks:
    v, f = builds[oid]
    x0, x1, y0, y1, z0, z1 = bb[oid]
    R = {"height": round(z1 - z0, 1),
         "footprint": [round(x1 - x0, 1), round(y1 - y0, 1)]}
    orc = Oracle(v, f, (x0, y0, min(z0, 0) - 2), (x1, y1, z1))

    # window in stored-grid indices
    c0 = max(int(x0 / MS) - 2, 0); c1 = min(int(x1 / MS) + 3, cols)
    r0 = max(int(y0 / MS) - 2, 0); r1 = min(int(y1 / MS) + 3, rows)
    woff = offset[r0:r1, c0:c1]
    k0 = max(int((z0 + woff.min()) / MS) - 2, 0)
    k1 = min(int((z1 + woff.max()) / MS) + 3, nz)
    S = solid[r0:r1, c0:c1, k0:k1].copy()
    nR, nC, nK = S.shape
    ccx = (np.arange(c0, c1) + 0.5) * MS
    ccy = (np.arange(r0, r1) + 0.5) * MS
    X, Y = np.meshgrid(ccx, ccy, indexing="xy")          # (nR, nC)
    Z = ((np.arange(k0, k1) + 0.5) * MS)[None, None, :] \
        - woff[:, :, None]                               # (nR, nC, nK)
    P = np.stack([np.broadcast_to(X[:, :, None], S.shape),
                  np.broadcast_to(Y[:, :, None], S.shape), Z], axis=-1)
    flat = P.reshape(-1, 3)
    contested = (samples[oid].query(flat, k=1)[0] < 2.0).reshape(S.shape)
    ok_cells = ~contested

    # ---- B: sub-voxel shift fit of STORED solid vs truth -------------------
    Sm = S & ok_cells
    best, iou0 = (-1.0, None), None
    for sx in np.arange(-1, 1.01, 0.25):
        for sy in np.arange(-1, 1.01, 0.25):
            for sz in np.arange(-1, 1.01, 0.25):
                V = orc.contains(flat - [sx, sy, sz]).reshape(S.shape) & ok_cells
                inter = (Sm & V).sum(); union = (Sm | V).sum()
                iou = inter / union if union else 0.0
                if sx == 0 and sy == 0 and sz == 0:
                    iou0 = iou
                if iou > best[0]:
                    best = (iou, (sx, sy, sz))
    R["B"] = {"iou_as_is": round(float(iou0), 3),
              "iou_best": round(float(best[0]), 3),
              "shift_m": list(best[1]), "n_stored": int(Sm.sum())}
    print(f"B {oid[-8:]}: h={R['height']:5.1f} m  IoU {iou0:.3f} -> "
          f"{best[0]:.3f} at ({best[1][0]:+.2f},{best[1][1]:+.2f},"
          f"{best[1][2]:+.2f}) m")

    # excess after best shift: how far outside truth do stored cells sit?
    Vb = orc.contains(flat - list(best[1])).reshape(S.shape) & ok_cells
    ex = Sm & ~Vb
    R["B"]["excess_frac_after_shift"] = round(float(ex.sum() / max(Sm.sum(), 1)), 3)

    # ---- D: per-column top excess (uncontested columns) --------------------
    dcounts = {}
    for iR in range(nR):
        for iC in range(nC):
            if contested[iR, iC].any():
                continue
            ks = np.nonzero(S[iR, iC])[0]
            if not len(ks):
                continue
            zs = np.arange(0, int(z1) + 4, 0.5)
            colpts = np.column_stack([np.full(len(zs), X[iR, iC]),
                                      np.full(len(zs), Y[iR, iC]), zs])
            inside_col = orc.contains(colpts)
            if not inside_col.any():
                continue
            true_top = zs[inside_col].max()
            exp_k = int(np.ceil((true_top + woff[iR, iC]) / MS)) - 1 - k0
            d = int(ks.max() - exp_k)
            dcounts[d] = dcounts.get(d, 0) + 1
    R["D_top_excess"] = dict(sorted(dcounts.items()))
    print(f"D {oid[-8:]}: top-layer excess {R['D_top_excess']}")

    # ---- C: their pipeline in a clean aligned frame ------------------------
    gx0 = MS * np.floor(x0 / MS) - 3 * MS; gx1 = MS * np.ceil(x1 / MS) + 3 * MS
    gy0 = MS * np.floor(y0 / MS) - 3 * MS; gy1 = MS * np.ceil(y1 / MS) + 3 * MS
    gz0 = MS * np.floor(min(z0, 0) / MS) - 3 * MS
    gz1 = MS * np.ceil(z1 / MS) + 3 * MS
    nC_ = int(round((gx1 - gx0) / MS)); nR_ = int(round((gy1 - gy0) / MS))
    nK_ = int(round((gz1 - gz0) / MS))
    gp = Grid3DParams(n_rows=nR_, n_cols=nC_, n_z=nK_, min_x=gx0, max_x=gx1,
                      min_y=gy0, max_y=gy1, min_z=gz0, max_z=gz1, voxel_size=MS)
    tm0 = trimesh.Trimesh(v.copy(), f.copy(), process=True)
    wt = make_watertight_mesh(np.asarray(tm0.vertices, float),
                              np.asarray(tm0.faces), voxel_size=MS)
    # resculpting metric: how far did watertighting move the surface?
    wpts = np.asarray(wt.vertices)
    dev = cKDTree(trimesh.sample.sample_surface(
        trimesh.Trimesh(v, f, process=False), 200_000)[0]).query(wpts, k=1)[0]
    R["C_wt"] = {"method": wt.method, "watertight": bool(wt.is_watertight),
                 "surface_dev_p95_m": round(float(np.percentile(dev, 95)), 3),
                 "surface_dev_max_m": round(float(dev.max()), 3)}

    g_ls = np.zeros((nR_, nC_, nK_), np.int32)
    _voxelize_meshlib_levelset(wt.vertices, wt.faces, gp, g_ls, -3, True)
    g_sh = g_ls.copy()
    _overlay_surface_shell(np.asarray(wt.vertices, float),
                           np.asarray(wt.faces), gp, g_sh, -3, False,
                           occupancy_threshold=0.0)
    ls = (g_ls == -3)[::-1]          # their row0=north -> ascending y
    sh = (g_sh == -3)[::-1]
    # truth on the same lattice
    cx = gx0 + (np.arange(nC_) + 0.5) * MS
    cy = gy0 + (np.arange(nR_) + 0.5) * MS
    cz = gz0 + (np.arange(nK_) + 0.5) * MS
    G = np.stack(np.meshgrid(cy, cx, cz, indexing="ij"), axis=-1)[..., [1, 0, 2]]
    Vt = orc.contains(G.reshape(-1, 3)).reshape(nR_, nC_, nK_)
    inter = (ls & Vt).sum(); union = (ls | Vt).sum()
    # measured displacement of the fill: best sub-voxel shift of truth onto fill
    bestC = (-1.0, None)
    for sx in np.arange(-1, 1.01, 0.25):
        for sy in np.arange(-1, 1.01, 0.25):
            for sz in np.arange(-1, 1.01, 0.25):
                Vs = orc.contains(G.reshape(-1, 3)
                                  - [sx, sy, sz]).reshape(nR_, nC_, nK_)
                i2 = (ls & Vs).sum(); u2 = (ls | Vs).sum()
                iou = i2 / u2 if u2 else 0
                if iou > bestC[0]:
                    bestC = (iou, (sx, sy, sz))
    # predicted stamp displacement from the actual MeshLib lattice origin
    v64 = np.ascontiguousarray(wt.vertices, np.float64)
    shift = v64.min(axis=0)
    ml = mrnp.meshFromFacesVerts(np.ascontiguousarray(wt.faces, np.int32),
                                 (v64 - shift).astype(np.float32))
    pp = mr.MeshToVolumeParams()
    pp.type = mr.MeshToVolumeParams.Type.Signed
    pp.surfaceOffset = 3
    pp.voxelSize = mr.Vector3f.diagonal(MS)
    oxf = mr.AffineXf3f(); pp.outXf = oxf
    mr.meshToVolume(mr.MeshPart(ml), pp)
    org = np.array([oxf.b.x, oxf.b.y, oxf.b.z]) + shift
    gmin = np.array([gp.min_x, 0.0, gp.min_z])
    pred = []
    for ax, om, gm in ((0, org[0], gp.min_x), (1, org[1], None),
                       (2, org[2], gp.min_z)):
        if ax == 1:
            # y axis: their stamp uses row = (max_y - y)/vs, same phase math
            ph = ((gp.max_y - om) / MS - 0.5) % 1.0
            pred.append(round((0.5 - ph) % 1.0 - (1.0 if ((0.5 - ph) % 1.0) > 0.5 else 0.0), 3))
        else:
            ph = ((om - gm) / MS + 0.5) % 1.0
            d = (ph - 0.5) % 1.0
            pred.append(round(MS * (d if d <= 0.5 else d - 1.0), 3))
    R["C"] = {"levelset_iou_vs_truth": round(float(inter / union), 3),
              "fill_shift_measured_m": list(bestC[1]),
              "fill_shift_iou": round(float(bestC[0]), 3),
              "n_fill": int(ls.sum()), "n_truth": int(Vt.sum()),
              "shell_added": int((sh & ~ls).sum()),
              "sdf_origin_phase_m": [round(float(((org[i] - gmin[i]) % MS)), 3)
                                     for i in range(3)]}
    print(f"C {oid[-8:]}: wt={wt.method:<10s} dev_p95 "
          f"{R['C_wt']['surface_dev_p95_m']:.2f} m | levelset IoU "
          f"{inter/union:.3f} -> {bestC[0]:.3f} at "
          f"({bestC[1][0]:+.2f},{bestC[1][1]:+.2f},{bestC[1][2]:+.2f}) m | "
          f"origin phase {R['C']['sdf_origin_phase_m']} | shell +"
          f"{(sh & ~ls).sum()} on {ls.sum()}")

    # ---- E: the NEW building seam (2026-08-11 alignment fix) --------------
    # Part C above reproduces the OLD path (watertight -> levelset -> shell at
    # threshold 0) and is kept as the "before" reference.  Part C does NOT
    # exercise the fix: it calls _voxelize_meshlib_levelset directly, which
    # the fix deliberately left alone.  This block runs what buildings
    # actually go through now -- grid-aligned winding SDF on the RAW mesh
    # (no watertight resculpting) plus the surface shell at 0.5 -- measured
    # against the same truth oracle on the same lattice, so C and E are
    # directly comparable.
    g_fill = np.zeros((nR_, nC_, nK_), np.int32)
    _voxelize_meshlib_winding(np.asarray(v, float), np.asarray(f), gp,
                              g_fill, -3, True, align_origin=True)
    g_new = g_fill.copy()
    _voxelize_building_solid(np.asarray(v, float), np.asarray(f), gp,
                             g_new, -3, True, shell_threshold=0.5)
    nf = (g_fill == -3)[::-1]
    nw = (g_new == -3)[::-1]
    iE = (nf & Vt).sum(); uE = (nf | Vt).sum()
    bestE = (-1.0, None)
    for sx in np.arange(-1, 1.01, 0.25):
        for sy in np.arange(-1, 1.01, 0.25):
            for sz in np.arange(-1, 1.01, 0.25):
                Vs = orc.contains(G.reshape(-1, 3)
                                  - [sx, sy, sz]).reshape(nR_, nC_, nK_)
                i2 = (nf & Vs).sum(); u2 = (nf | Vs).sum()
                iou = i2 / u2 if u2 else 0
                if iou > bestE[0]:
                    bestE = (iou, (sx, sy, sz))
    shell_add = int((nw & ~nf).sum())
    R["E_new_seam"] = {
        "fill_iou_vs_truth": round(float(iE / uE), 3),
        "fill_shift_measured_m": list(bestE[1]),
        "fill_shift_iou": round(float(bestE[0]), 3),
        "n_fill": int(nf.sum()), "n_truth": int(Vt.sum()),
        "shell_added": shell_add,
        "shell_added_pct_of_fill": round(
            100.0 * shell_add / max(int(nf.sum()), 1), 1),
    }
    print(f"E {oid[-8:]}: NEW seam   fill IoU {iE / uE:.3f} -> {bestE[0]:.3f} "
          f"at ({bestE[1][0]:+.2f},{bestE[1][1]:+.2f},{bestE[1][2]:+.2f}) m | "
          f"shell +{shell_add} on {int(nf.sum())} "
          f"({R['E_new_seam']['shell_added_pct_of_fill']:.1f}%)")

    results[oid] = R

report["buildings"] = results
(OUT / "voxelization_diagnosis2.json").write_text(json.dumps(report, indent=2))
print(f"\nwrote {OUT / 'voxelization_diagnosis2.json'}")

# ---- acceptance summary over all analysed buildings ------------------------
print("\n=== ACCEPTANCE (new seam, Part E) ===")
shifts = [tuple(r["E_new_seam"]["fill_shift_measured_m"]) for r in results.values()]
ious = [r["E_new_seam"]["fill_iou_vs_truth"] for r in results.values()]
pcts = [r["E_new_seam"]["shell_added_pct_of_fill"] for r in results.values()]
gains = [r["E_new_seam"]["fill_shift_iou"] - r["E_new_seam"]["fill_iou_vs_truth"]
         for r in results.values()]
old_shifts = [tuple(r["C"]["fill_shift_measured_m"]) for r in results.values()]
print(f"  OLD path best-fit fill shifts : {sorted(set(old_shifts))}")
print(f"  NEW seam best-fit fill shifts : {sorted(set(shifts))}")
print(f"    all zero?                   : {all(s == (0.0, 0.0, 0.0) for s in shifts)}")
print(f"  NEW fill IoU vs truth         : min {min(ious):.3f}  max {max(ious):.3f}")
print(f"    all >= 0.90?                : {all(i >= 0.90 for i in ious)}")
print(f"    max IoU gain from shifting  : {max(gains):.3f}")
print(f"  shell added, % of fill        : min {min(pcts):.1f}%  max {max(pcts):.1f}%")
print(f"    all <= 10%?                 : {all(p <= 10.0 for p in pcts)}")
