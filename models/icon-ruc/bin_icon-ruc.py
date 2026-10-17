import sys
import json
import os
import gc
import warnings
from zoneinfo import ZoneInfo

import cfgrib
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, cKDTree

from encoding_config import NODATA_U8, LEVELS, ENCODING

warnings.simplefilter(action='ignore', category=FutureWarning)

# ------------------------------
# Eingabe-/Ausgabe
# ------------------------------
data_dir = sys.argv[1]        # z.B. "data/t2m"
output_dir = sys.argv[2]      # z.B. "iconruc/t2m"
var_type = sys.argv[3].lower()  # 't2m', 'ww', 'tp', 'tp_acc', 'cape_ml', 'dbz_cmax', 'wind', 'srh3km', 'ehi', 'scp', ...
grid_dir = sys.argv[4] if len(sys.argv) > 4 else "data/grid"
cape_dir = sys.argv[5] if len(sys.argv) > 5 else None
wshearu_dir = sys.argv[6] if len(sys.argv) > 6 else None
wshearv_dir = sys.argv[7] if len(sys.argv) > 7 else None

os.makedirs(output_dir, exist_ok=True)

if var_type not in ENCODING:
    print(f"Unbekannter var_type {var_type}")
    sys.exit(1)
enc = ENCODING[var_type]

# ICON-RUC läuft auf dem nativen Dreiecksgitter - die Gitterkoordinaten
# müssen separat aus den von DWD bereitgestellten CLAT-/CLON-GRIB2-Dateien
# geladen werden.
clat_path = os.path.join(grid_dir, "clat.grib2")
clon_path = os.path.join(grid_dir, "clon.grib2")

for gp in (clat_path, clon_path):
    if not os.path.exists(gp):
        raise FileNotFoundError(f"Grid-Datei nicht gefunden: {gp}")

# ------------------------------
# Ausgaberaster (identisch zum ICON-D2-Raster)
# ------------------------------
# regular-lat-lon, EPSG:4326, row0 = Norden, Werte = Pixelzentren.
# MUSS zu GRID in generate_metadata.py passen (dort wird die Dateigröße geprüft).
GRID_LON_MIN, GRID_LAT_MIN, GRID_LON_MAX, GRID_LAT_MAX = -3.94, 43.18, 20.34, 58.08
GRID_WIDTH, GRID_HEIGHT = 1215, 746

# Ziel-Pixel: Zeilen von Nord nach Süd, Spalten von West nach Ost
lon_t = np.linspace(GRID_LON_MIN, GRID_LON_MAX, GRID_WIDTH)
lat_t = np.linspace(GRID_LAT_MAX, GRID_LAT_MIN, GRID_HEIGHT)
lon_tt, lat_tt = np.meshgrid(lon_t, lat_t)          # Shape (HEIGHT, WIDTH)

# Für Triangulation/Nearest-Neighbor wird die Länge mit cos(Breite) skaliert,
# damit die Dreiecke in der (lon, lat)-Ebene annähernd unverzerrt sind.
COS_LAT0 = np.cos(np.radians(0.5 * (GRID_LAT_MIN + GRID_LAT_MAX)))
target_points = np.column_stack((lon_tt.ravel() * COS_LAT0, lat_tt.ravel()))
n_target = target_points.shape[0]


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
# Gitter laden + Interpolation einmalig vorbereiten
# ------------------------------
lats = _load_grid_coords(clat_path)
lons = _load_grid_coords(clon_path)
n_grid = lats.shape[0]

points_all = np.column_stack((lons * COS_LAT0, lats))

finite_mask = np.all(np.isfinite(points_all), axis=1)
n_bad = int((~finite_mask).sum())
if n_bad:
    print(f"Warnung: {n_bad} nicht-endliche Gitterpunkte gefunden, werden entfernt.")
    valid_idx = np.nonzero(finite_mask)[0]
    points = points_all[valid_idx]
else:
    valid_idx = None
    points = points_all

print(f"Gitterpunkte: {points.shape[0]} (finite)")
print("Baue Triangulation für Interpolation & Hüllen-Maske auf ...")
tri = Delaunay(points, qhull_options="Qbb Qc Qz Qt")

# Für jeden Zielpixel: umgebendes Dreieck + baryzentrische Gewichte (einmalig).
# Danach ist die Interpolation pro Datei nur noch eine gewichtete Summe.
simplex = tri.find_simplex(target_points)
inside = simplex >= 0
_s = simplex[inside]
_T = tri.transform[_s]                                    # (n, 3, 2)
_delta = target_points[inside] - _T[:, 2]
_b = np.einsum("nij,nj->ni", _T[:, :2], _delta)           # (n, 2)
_w = np.column_stack((_b, 1.0 - _b.sum(axis=1)))          # (n, 3)
_verts = tri.simplices[_s]                                # (n, 3)

# entartete Dreiecke (NaN-Gewichte) wie "außerhalb" behandeln
_ok = np.all(np.isfinite(_w), axis=1)
if not np.all(_ok):
    _idx_inside = np.nonzero(inside)[0]
    inside[_idx_inside[~_ok]] = False
    _verts = _verts[_ok]
    _w = _w[_ok]

inside_2d = inside.reshape(GRID_HEIGHT, GRID_WIDTH)

nn_idx = None
if var_type == "ww":
    # Wettercodes dürfen nicht gemittelt werden -> nächster Gitterpunkt
    _, nn_idx = cKDTree(points).query(target_points)


def to_target_grid(values_native):
    """Werte auf dem Dreiecksgitter (1D) -> Zielraster (HEIGHT, WIDTH), row0 = Norden.
    Außerhalb der Gitter-Hülle = NaN."""
    if valid_idx is not None:
        values_native = values_native[valid_idx]
    values_native = np.asarray(values_native, dtype=np.float64)

    out = np.full(n_target, np.nan, dtype=np.float64)
    if nn_idx is not None:
        out[inside] = values_native[nn_idx[inside]]
    else:
        out[inside] = np.einsum("ni,ni->n", values_native[_verts], _w)
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
    elif var_type == "dbz_cmax":
        if "DBZ_CMAX" not in ds:
            print(f"Keine DBZ_CMAX in {filename}")
            ds.close()
            continue
        data = ds["DBZ_CMAX"].values
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
    elif var_type == "srh3km":
        if "hlcy" not in ds:
            print(f"Keine hlcy in {filename}")
            ds.close()
            continue
        data = ds["hlcy"].values
    elif var_type == "ehi":
        if cape_dir is None:
            print("Kein cape_dir angegeben")
            ds.close()
            continue
        suffix = filename.replace("srh3km_", "")
        cape_path = os.path.join(cape_dir, f"cape_ml_{suffix}")
        if not os.path.exists(cape_path):
            print(f"Kein CAPE-File: {cape_path}")
            ds.close()
            continue
        ds_cape = cfgrib.open_dataset(cape_path)
        if "hlcy" not in ds or "CAPE_ML" not in ds_cape:
            print("Variablen fehlen")
            ds.close()
            ds_cape.close()
            continue
        srh = ds["hlcy"].values
        cape = ds_cape["CAPE_ML"].values
        cape[cape < 0] = 0
        data = (cape * srh) / 160000.0
        ds_cape.close()
    elif var_type == "scp":
        if None in [cape_dir, wshearu_dir, wshearv_dir]:
            print("SCP benötigt cape_dir, wshearu_dir, wshearv_dir")
            ds.close()
            continue
        suffix = filename.replace("srh3km_", "")
        cape_path = os.path.join(cape_dir, f"cape_ml_{suffix}")
        wshearu_path = os.path.join(wshearu_dir, f"wshear_u_{suffix}")
        wshearv_path = os.path.join(wshearv_dir, f"wshear_v_{suffix}")

        missing = False
        for p in [cape_path, wshearu_path, wshearv_path]:
            if not os.path.exists(p):
                print(f"Datei nicht gefunden: {p}")
                missing = True
        if missing:
            ds.close()
            continue

        ds_cape = cfgrib.open_dataset(cape_path)
        ds_wshearu = cfgrib.open_dataset(wshearu_path)
        ds_wshearv = cfgrib.open_dataset(wshearv_path)

        if "hlcy" not in ds:
            print(f"hlcy fehlt in {filename}")
            ds.close()
            ds_cape.close()
            ds_wshearu.close()
            ds_wshearv.close()
            continue

        srh = ds["hlcy"].values
        cape = ds_cape["CAPE_ML"].values
        bs06 = np.sqrt(ds_wshearu["WSHEAR_U"].values**2 + ds_wshearv["WSHEAR_V"].values**2)

        cape[cape < 0] = 0
        cape_term = cape / 1000
        srh_term = np.maximum(0, srh) / 50
        ebs_term = np.where(bs06 < 10, 0.0, np.where(bs06 > 20, 1.0, bs06 / 20))

        data = np.round(cape_term * srh_term * ebs_term, 2)
        data[data < 0] = np.nan

        ds_cape.close()
        ds_wshearu.close()
        ds_wshearv.close()
    else:
        print(f"Unbekannter var_type: {var_type}")
        ds.close()
        continue

    data = np.asarray(data).squeeze()

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
    if data.shape != (n_grid,):
        print(f"{filename}: Datenform {data.shape} passt nicht zum Grid ({n_grid},), überspringe")
        continue

    grid_data = to_target_grid(data)          # row0 = Norden, NaN außerhalb der Hülle
    if var_type == "ww":
        nat = np.unique(data[np.isfinite(data)])
        gr = np.unique(grid_data[np.isfinite(grid_data)])
        print("native:", nat)
        print("grid:  ", gr)
        print("nur im Grid:", np.setdiff1d(gr, nat))
    # --- Ende Debug ---
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