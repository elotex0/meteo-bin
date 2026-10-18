import sys
import json
import os
import gc
import warnings
from zoneinfo import ZoneInfo

import cfgrib
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, cKDTree

from encoding_config import NODATA_U8, LEVELS, ENCODING

warnings.simplefilter(action='ignore', category=FutureWarning)

# ------------------------------
# Eingabe-/Ausgabe
# ------------------------------
data_dir = sys.argv[1]          # z.B. "data/t2m"
output_dir = sys.argv[2]        # z.B. "icon/t2m"
var_type = sys.argv[3].lower()  # 't2m', 'tp', 'tp_acc', 'ww', 'cape_ml', 'wind', 'snow'
grid_dir = sys.argv[4] if len(sys.argv) > 4 else "data/grid"

os.makedirs(output_dir, exist_ok=True)

if var_type not in ENCODING:
    print(f"Unbekannter var_type {var_type}")
    sys.exit(1)
enc = ENCODING[var_type]

# ICON (global) läuft auf dem nativen Dreiecksgitter (R03B07, ~13 km) - die
# Gitterkoordinaten kommen aus den CLAT-/CLON-GRIB2-Dateien (ICON-GLOBAL-Gitter!).
clat_path = os.path.join(grid_dir, "clat.grib2")
clon_path = os.path.join(grid_dir, "clon.grib2")

for gp in (clat_path, clon_path):
    if not os.path.exists(gp):
        raise FileNotFoundError(f"Grid-Datei nicht gefunden: {gp}")

# ------------------------------
# Ausgaberaster: nur Europa (gleicher Ausschnitt wie ICON-EU)
# ------------------------------
# regular-lat-lon, EPSG:4326, row0 = Norden, Werte = Pixelzentren.
# MUSS zu GRID in der Meta-Datei passen (dort wird die Dateigröße geprüft).
# Auflösung 0.125 Grad (~13.9 km) passt zur nativen Auflösung von ICON global.
# Breite/Höhe werden aus Extent und Auflösung berechnet.
# Der Ausschnitt darf die Datumsgrenze (+-180) nicht überqueren.
GRID_LON_MIN, GRID_LAT_MIN, GRID_LON_MAX, GRID_LAT_MAX = -23.5, 29.5, 62.5, 70.5
GRID_RES = 0.125

GRID_WIDTH = int(round((GRID_LON_MAX - GRID_LON_MIN) / GRID_RES)) + 1    # 689
GRID_HEIGHT = int(round((GRID_LAT_MAX - GRID_LAT_MIN) / GRID_RES)) + 1   # 329
if (abs((GRID_WIDTH - 1) * GRID_RES - (GRID_LON_MAX - GRID_LON_MIN)) > 1e-9 or
        abs((GRID_HEIGHT - 1) * GRID_RES - (GRID_LAT_MAX - GRID_LAT_MIN)) > 1e-9):
    raise ValueError("Extent ist kein ganzzahliges Vielfaches von GRID_RES")

lon_t = np.linspace(GRID_LON_MIN, GRID_LON_MAX, GRID_WIDTH)
lat_t = np.linspace(GRID_LAT_MAX, GRID_LAT_MIN, GRID_HEIGHT)   # Nord -> Süd
lon_tt, lat_tt = np.meshgrid(lon_t, lat_t)                     # Shape (HEIGHT, WIDTH)
n_target = GRID_WIDTH * GRID_HEIGHT


def to_xyz(lon_deg, lat_deg):
    """Grad -> Einheitsvektoren auf der Kugel (n, 3)."""
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    return np.column_stack((
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ))


def encode_uint8(data, enc):
    """float-Array (NaN erlaubt) -> uint8-Array (255 = NoData)."""
    nan_mask = ~np.isfinite(data)
    vmin, vmax, tr = enc["vmin"], enc["vmax"], enc["transform"]
    d = np.where(nan_mask, vmin, data).astype(np.float64)

    if tr == "raw":
        q = np.clip(np.rint(d), 0, LEVELS)
    else:
        d = np.clip(d, vmin, vmax)
        if tr == "linear":
            u = (d - vmin) / (vmax - vmin)
        elif tr == "sqrt":
            u = np.sqrt((d - vmin) / (vmax - vmin))
        elif tr == "ssqrt":
            u = (np.sign(d) * np.sqrt(np.abs(d) / vmax) + 1.0) / 2.0
        else:
            raise ValueError(tr)
        q = np.rint(u * LEVELS)

    q = q.astype(np.uint8)
    q[nan_mask] = NODATA_U8
    return q


def _load_grid_coords(path):
    """Lädt Gitterkoordinaten aus CLAT-/CLON-GRIB2-Datei (in Grad)."""
    ds_grid = cfgrib.open_dataset(path)
    varname = list(ds_grid.data_vars)[0]
    values = np.asarray(ds_grid[varname].values).ravel()
    ds_grid.close()

    if np.nanmax(np.abs(values)) <= (np.pi + 0.1):
        values = np.rad2deg(values)
    return values


# ------------------------------
# Gitter laden + auf Europa zuschneiden
# ------------------------------
lats_all = _load_grid_coords(clat_path)
lons_all = _load_grid_coords(clon_path)
lons_all = np.where(lons_all > 180.0, lons_all - 360.0, lons_all)
n_grid = lats_all.shape[0]

if not (np.all(np.isfinite(lats_all)) and np.all(np.isfinite(lons_all))):
    raise RuntimeError("clat/clon enthalten NaN - Grid-Dateien prüfen")

# Puffer um den Ausschnitt: Die Triangulation wird nur für diesen Teil gebaut
# (statt für die ganze Erde). Am Rand des Puffers ist sie unzuverlässig,
# deshalb muss er deutlich größer sein als der Punktabstand (~0.12 Grad).
MARGIN = 3.0  # Grad
sel = (
    (lons_all >= GRID_LON_MIN - MARGIN) & (lons_all <= GRID_LON_MAX + MARGIN) &
    (lats_all >= GRID_LAT_MIN - MARGIN) & (lats_all <= GRID_LAT_MAX + MARGIN)
)
sel_idx = np.nonzero(sel)[0]          # Indizes ins volle Gitter (für die Daten)
n_sel = sel_idx.size

print(f"Gitterpunkte nach Zuschnitt: {n_sel} von {n_grid}")
if n_sel < 3:
    raise RuntimeError("Zu wenige Gitterpunkte im Ausschnitt - stimmen clat/clon?")

xyz = to_xyz(lons_all[sel_idx], lats_all[sel_idx])
target_xyz = to_xyz(lon_tt.ravel(), lat_tt.ravel())

# Cache-Dateien (Interpolationsgewichte hängen nur von Gitter + Zielraster ab)
_tag = (f"{GRID_WIDTH}x{GRID_HEIGHT}_{GRID_LON_MIN}_{GRID_LAT_MIN}_"
        f"{GRID_LON_MAX}_{GRID_LAT_MAX}_{n_grid}_{n_sel}")
nn_cache = os.path.join(grid_dir, f"interp_globeu_nn_{_tag}.npy")
bary_cache = os.path.join(grid_dir, f"interp_globeu_bary_{_tag}.npz")


def get_nearest():
    """Nächster (zugeschnittener) Gitterpunkt für jeden Zielpixel (n_target,)."""
    if os.path.exists(nn_cache):
        print("Lade Nearest-Neighbor aus Cache ...")
        return np.load(nn_cache)
    print("Berechne Nearest-Neighbor ...")
    tree = cKDTree(xyz)
    _, nn = tree.query(target_xyz, workers=-1)
    nn = nn.astype(np.int32)
    np.save(nn_cache, nn)
    return nn


def build_barycentric(nn):
    """Sphärische Delaunay-Triangulation (= konvexe Hülle der Punkte auf der
    Einheitskugel). Für jeden Zielpixel wird das umgebende Dreieck unter den
    Dreiecken des nächsten Gitterpunkts gesucht und baryzentrisch interpoliert.
    Rückgabe: verts (n_target, 3) int32, weights (n_target, 3) float32.
    Indizes beziehen sich auf den zugeschnittenen Punktsatz (sel_idx)."""
    if os.path.exists(bary_cache):
        print("Lade Interpolationsgewichte aus Cache ...")
        z = np.load(bary_cache)
        return z["verts"], z["w"]

    print("Baue sphärische Triangulation auf ...")
    hull = ConvexHull(xyz, qhull_options="Qt Qbb")
    simp = hull.simplices.astype(np.int32)             # (n_tri, 3)
    n_tri = simp.shape[0]

    # Vertex -> angrenzende Dreiecke (auf feste Breite aufgefüllt mit -1)
    tv = simp.ravel()
    tid = np.repeat(np.arange(n_tri, dtype=np.int32), 3)
    order = np.argsort(tv, kind="stable")
    tv_s, tid_s = tv[order], tid[order]
    counts = np.bincount(tv, minlength=n_sel)
    starts = np.cumsum(counts) - counts
    pos = np.arange(tv_s.size) - starts[tv_s]
    cand = np.full((n_sel, int(counts.max())), -1, dtype=np.int32)
    cand[tv_s, pos] = tid_s
    del tv, tid, order, tv_s, tid_s, pos
    print(f"Dreiecke: {n_tri}, max. Dreiecke pro Punkt: {cand.shape[1]}")

    verts_out = np.empty((n_target, 3), dtype=np.int32)
    w_out = np.empty((n_target, 3), dtype=np.float32)
    n_fallback = 0
    CH = 100_000

    for s in range(0, n_target, CH):
        e = min(s + CH, n_target)
        p = target_xyz[s:e]                              # (m, 3)
        v = nn[s:e]
        ct = cand[v]                                     # (m, D)
        valid = ct >= 0
        tri_v = simp[np.where(valid, ct, 0)]             # (m, D, 3)
        V = xyz[tri_v]                                   # (m, D, 3 Ecken, 3 xyz)

        # Löse  c0*V0 + c1*V1 + c2*V2 = p
        M = np.swapaxes(V, -1, -2)                       # (m, D, 3, 3)
        c = np.linalg.solve(M, p[:, None, :, None])[..., 0]   # (m, D, 3)

        c_min = c.min(axis=-1)
        c_sum = c.sum(axis=-1)
        score = np.where(valid & (c_sum > 0), c_min, -np.inf)
        best = score.argmax(axis=1)
        ar = np.arange(e - s)
        best_score = score[ar, best]

        cb = c[ar, best]                                 # (m, 3)
        vb = tri_v[ar, best]                             # (m, 3)
        ok = best_score >= -1e-6

        cb = np.clip(cb, 0.0, None)
        ssum = cb.sum(axis=1, keepdims=True)
        ssum[ssum == 0] = 1.0
        cb = cb / ssum

        # Fallback: nächster Punkt (numerische Randfälle)
        bad = ~ok
        if bad.any():
            n_fallback += int(bad.sum())
            vb[bad] = v[bad][:, None]
            cb[bad] = np.array([1.0, 0.0, 0.0])

        verts_out[s:e] = vb
        w_out[s:e] = cb.astype(np.float32)

    print(f"Fallback auf Nearest-Neighbor: {n_fallback} Pixel")
    np.savez(bary_cache, verts=verts_out, w=w_out)
    return verts_out, w_out


nn_idx = get_nearest()
if var_type == "ww":
    # Wettercodes dürfen nicht gemittelt werden -> nächster Gitterpunkt.
    verts = weights = None
else:
    verts, weights = build_barycentric(nn_idx)


def to_target_grid(values_native):
    """Werte auf dem zugeschnittenen Dreiecksgitter (1D, n_sel) -> Zielraster
    (HEIGHT, WIDTH), row0 = Norden."""
    values_native = np.asarray(values_native, dtype=np.float64)

    if var_type == "ww":
        return values_native[nn_idx].reshape(GRID_HEIGHT, GRID_WIDTH)

    out = np.einsum("ni,ni->n", values_native[verts], weights.astype(np.float64))
    return out.reshape(GRID_HEIGHT, GRID_WIDTH)


# ------------------------------
# Dateien durchgehen
# ------------------------------
all_files_global = sorted([f for f in os.listdir(data_dir) if f.endswith(".grib2")])

meta = {
    "var": var_type,
    "unit": enc["unit"],
    "dtype": "uint8",
    "nodata": NODATA_U8,
    "levels": LEVELS,
    "transform": enc["transform"],
    "vmin": enc["vmin"],
    "vmax": enc["vmax"],
    "decode": {
        "linear": "v = vmin + (q/254) * (vmax - vmin)",
        "sqrt": "v = vmin + (q/254)^2 * (vmax - vmin)",
        "ssqrt": "u = q/254*2 - 1; v = sign(u) * u^2 * vmax",
        "raw": "v = q",
    }[enc["transform"]],
    "orientation": "row0 = Norden, row-major, uint8",
    "projection": "EPSG:4326",
    "extent_4326": [GRID_LON_MIN, GRID_LAT_MIN, GRID_LON_MAX, GRID_LAT_MAX],
    "width": GRID_WIDTH,
    "height": GRID_HEIGHT,
    "files": [],
}

for filename in all_files_global:
    path = os.path.join(data_dir, filename)
    ds = cfgrib.open_dataset(path)

    # --------------------------
    # Daten je Typ (auf dem Dreiecksgitter)
    # --------------------------
    if var_type == "t2m":
        if "t2m" not in ds:
            print(f"Keine t2m in {filename}")
            ds.close()
            continue
        data = ds["t2m"].values - 273.15
    elif var_type == "tp":
        if "tprate" not in ds:
            print(f"Keine tprate in {filename}")
            ds.close()
            continue
        data = ds["tprate"].values
    elif var_type == "tp_acc":
        if "tp" not in ds:
            print(f"Keine tp in {filename}")
            ds.close()
            continue
        data = ds["tp"].values
    elif var_type == "ww":
        varname = next((vn for vn in ds.data_vars if vn.lower() in ["ww", "weather"]), None)
        if varname is None:
            print(f"Keine WW in {filename}")
            ds.close()
            continue
        data = ds[varname].values
    elif var_type == "cape_ml":
        if "CAPE_ML" not in ds:
            print(f"Keine CAPE_ML in {filename}")
            ds.close()
            continue
        data = ds["CAPE_ML"].values
        data[data < 0] = np.nan
    # dbz_cmax gibt es nur in ICON-D2, nicht in ICON global -> Zweig entfernt.
    elif var_type == "wind":
        if "fg10" not in ds:
            print(f"Keine fg10 in {filename}")
            ds.close()
            continue
        data = ds["fg10"].values * 3.6  # m/s -> km/h
        data[data < 0] = np.nan
    elif var_type == "snow":
        if "sde" not in ds:
            print(f"Keine sde in {filename}")
            ds.close()
            continue
        data = ds["sde"].values * 100  # -> cm
        data[data < 0] = np.nan
    else:
        print(f"Unbekannter var_type: {var_type}")
        ds.close()
        continue

    data = np.asarray(data)
    if data.ndim == 3:
        data = data[0]
    data = data.ravel()

    # --------------------------
    # Zeiten
    # --------------------------
    run_time_utc = pd.to_datetime(ds["time"].values) if "time" in ds else None
    if "valid_time" in ds:
        valid_time_raw = ds["valid_time"].values
        valid_time_utc = pd.to_datetime(valid_time_raw[0]) if np.ndim(valid_time_raw) > 0 else pd.to_datetime(valid_time_raw)
    else:
        step = pd.to_timedelta(ds["step"].values[0]) if "step" in ds else pd.to_timedelta(0)
        valid_time_utc = run_time_utc + step if run_time_utc is not None else None

    if valid_time_utc is None:
        print(f"{filename}: keine Gültigkeitszeit ermittelbar, überspringe")
        ds.close()
        continue

    valid_time_local = valid_time_utc.tz_localize("UTC").astimezone(ZoneInfo("Europe/Berlin"))

    ds.close()

    # --------------------------
    # Dreiecksgitter -> Zielraster -> uint8
    # --------------------------
    if data.shape[0] != n_grid:
        print(f"{filename}: Anzahl Datenpunkte ({data.shape[0]}) passt nicht zum Grid ({n_grid}), überspringe")
        continue

    grid_data = to_target_grid(data[sel_idx])     # row0 = Norden
    q = encode_uint8(grid_data, enc)

    outname = f"{var_type}_{valid_time_local:%Y%m%d_%H%M}"
    out_path = os.path.join(output_dir, outname)
    np.ascontiguousarray(q, dtype=np.uint8).tofile(out_path)

    meta["files"].append({
        "file": outname,
        "valid_time_utc": valid_time_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "valid_time_local": valid_time_local.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })

    print(f"{filename} -> {outname} ({GRID_WIDTH}x{GRID_HEIGHT}, {q.nbytes} Bytes)")

    del data, grid_data, q
    gc.collect()

print("META: " + json.dumps(meta, ensure_ascii=False))
print("Fertig!")