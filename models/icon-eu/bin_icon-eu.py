import sys
import json
import os
import gc
import warnings
from zoneinfo import ZoneInfo

import cfgrib
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from encoding_config import NODATA_U8, LEVELS, ENCODING

warnings.simplefilter(action='ignore', category=FutureWarning)

# ------------------------------
# Eingabe-/Ausgabe
# ------------------------------
data_dir = sys.argv[1]        # z.B. "data/t2m"
output_dir = sys.argv[2]      # z.B. "iconeu/t2m"
var_type = sys.argv[3]        # 't2m', 't850', 'ww', 'tp', 'tp_acc', 'wind', ...
os.makedirs(output_dir, exist_ok=True)

if var_type not in ENCODING:
    print(f"Unbekannter var_type {var_type}")
    sys.exit(1)
enc = ENCODING[var_type]

# ------------------------------
# Ausgaberaster (identisch zum ICON-D2-Raster)
# ------------------------------
# regular-lat-lon, EPSG:4326, row0 = Norden, Werte = Pixelzentren.
# MUSS zu GRID in generate_metadata.py passen (dort wird die Dateigröße geprüft).
GRID_LON_MIN, GRID_LAT_MIN, GRID_LON_MAX, GRID_LAT_MAX = -3.94, 43.18, 20.34, 58.08
GRID_WIDTH, GRID_HEIGHT = 1215, 746

lon_t = np.linspace(GRID_LON_MIN, GRID_LON_MAX, GRID_WIDTH)
lat_t = np.linspace(GRID_LAT_MAX, GRID_LAT_MIN, GRID_HEIGHT)   # Nord -> Süd
lat_tt, lon_tt = np.meshgrid(lat_t, lon_t, indexing="ij")      # Shape (HEIGHT, WIDTH)
target_points = np.column_stack((lat_tt.ravel(), lon_tt.ravel()))


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


def to_target_grid(data, lon, lat, method):
    """Reguläres Modellgitter (lat, lon) -> Ausgaberaster (HEIGHT, WIDTH), row0 = Norden.
    Punkte außerhalb des Modellgitters = NaN."""
    # Längen in -180..180 bringen und beide Achsen aufsteigend sortieren
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    order_lon = np.argsort(lon)
    order_lat = np.argsort(lat)
    lon = lon[order_lon]
    lat = lat[order_lat]
    data = data[np.ix_(order_lat, order_lon)]

    interp = RegularGridInterpolator(
        (lat, lon), data, method=method, bounds_error=False, fill_value=np.nan
    )
    return interp(target_points).reshape(GRID_HEIGHT, GRID_WIDTH)


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

    valid_time_utc_override = None

    # ------------------------------
    # Daten je Typ
    # ------------------------------
    if var_type == "t2m":
        if "t2m" not in ds:
            print(f"Keine t2m in {filename}")
            ds.close()
            continue
        data = ds["t2m"].values - 273.15
    elif var_type == "t850":
        if "t" not in ds:
            print(f"Keine t in {filename} ds.keys(): {list(ds.keys())}")
            ds.close()
            continue
        data = ds["t"].values - 273.15
    elif var_type == "ww":
        varname = next((vn for vn in ds.data_vars if vn.lower() in ["ww", "weather"]), None)
        if varname is None:
            print(f"Keine WW in {filename}")
            ds.close()
            continue
        data = ds[varname].values
    elif var_type == "tp":
        if "tp" not in ds:
            print(f"Keine tp in {filename}")
            ds.close()
            continue

        idx_now = all_files_global.index(filename)
        if idx_now >= 78:
            print(f"Datei {filename}: Überspringe stündlichen Niederschlag (3h Schritte)")
            ds.close()
            continue
        if idx_now + 1 >= len(all_files_global):
            print(f"{filename}: keine folgende Datei -> 1h-Niederschlag nicht berechenbar, überspringe")
            ds.close()
            continue

        next_path = os.path.join(data_dir, all_files_global[idx_now + 1])
        ds_next = cfgrib.open_dataset(next_path)

        if "tp" not in ds_next:
            print(f"Keine tp in {all_files_global[idx_now + 1]}")
            ds_next.close()
            ds.close()
            continue

        tp_now = ds["tp"].values
        tp_next = ds_next["tp"].values
        data = tp_next - tp_now

        vt_next_raw = ds_next["valid_time"].values
        valid_time_utc_override = pd.to_datetime(vt_next_raw[0]) if np.ndim(vt_next_raw) > 0 else pd.to_datetime(vt_next_raw)
        ds_next.close()
    elif var_type == "tp_acc":
        tp_var = next((vn for vn in ["tp", "tot_prec"] if vn in ds), None)
        if tp_var is None:
            print(f"Keine Niederschlagsvariable in {filename}")
            ds.close()
            continue
        lon_tmp = ds["longitude"].values
        lat_tmp = ds["latitude"].values
        tp_all = ds[tp_var].values
        if tp_all.ndim == 1:
            ny, nx = len(lat_tmp), len(lon_tmp)
            data = tp_all.reshape(ny, nx)
        elif tp_all.ndim == 3:
            data = tp_all[3] - tp_all[0] if tp_all.shape[0] > 1 else tp_all[0]
        else:
            data = tp_all
    elif var_type == "dbz_cmax":
        if "DBZ_CMAX" not in ds:
            print(f"Keine DBZ_CMAX in {filename} ds.keys(): {list(ds.keys())}")
            ds.close()
            continue
        data = ds["DBZ_CMAX"].values
    elif var_type == "wind":
        if "fg10" not in ds:
            print(f"Keine passende Windvariable in {filename} ds.keys(): {list(ds.keys())}")
            ds.close()
            continue
        data = ds["fg10"].values
        data[data < 0] = np.nan
        data = data * 3.6  # m/s -> km/h
    elif var_type == "snow":
        if "sde" not in ds:
            print(f"Keine sde-Variable in {filename}")
            ds.close()
            continue
        data = ds["sde"].values
        data[data < 0] = np.nan
        data = data * 100  # -> cm
    elif var_type == "twater":
        if "TWATER" not in ds:
            print(f"Keine TWATER-Variable in {filename}")
            ds.close()
            continue
        data = ds["TWATER"].values
        data[data < 0] = np.nan
    elif var_type == "snowfall":
        if "SNOWLMT" not in ds:
            print(f"Keine SNOWLMT-Variable in {filename} ds.keys(): {list(ds.keys())}")
            ds.close()
            continue
        data = ds["SNOWLMT"].values
        data[data < 0] = np.nan
    elif var_type == "pmsl":
        if "prmsl" not in ds:
            print(f"Keine prmsl-Variable in {filename} ds.keys(): {list(ds.keys())}")
            ds.close()
            continue
        data = ds["prmsl"].values / 100  # Pa -> hPa
        data[data < 0] = np.nan
    else:
        print(f"Unbekannter var_type {var_type}")
        ds.close()
        continue

    data = np.asarray(data)
    if data.ndim == 3:
        data = data[0]

    # --------------------------
    # Zeiten
    # --------------------------
    lon = np.asarray(ds["longitude"].values, dtype=np.float64)
    lat = np.asarray(ds["latitude"].values, dtype=np.float64)
    run_time_utc = pd.to_datetime(ds["time"].values) if "time" in ds else None

    if valid_time_utc_override is not None:
        valid_time_utc = valid_time_utc_override
    elif "valid_time" in ds:
        valid_time_raw = ds["valid_time"].values
        valid_time_utc = pd.to_datetime(valid_time_raw[0]) if np.ndim(valid_time_raw) > 0 else pd.to_datetime(valid_time_raw)
    else:
        step = pd.to_timedelta(ds["step"].values[0])
        valid_time_utc = run_time_utc + step
    valid_time_local = valid_time_utc.tz_localize("UTC").astimezone(ZoneInfo("Europe/Berlin"))

    ds.close()

    # --------------------------
    # Modellgitter -> Ausgaberaster -> uint8
    # --------------------------
    if not (lon.ndim == 1 and lat.ndim == 1 and data.shape == (lat.size, lon.size)):
        print(f"{filename}: Datenform {data.shape} passt nicht zum lat/lon-Gitter "
              f"({lat.size}, {lon.size}), überspringe")
        continue

    method = "nearest" if var_type == "ww" else "linear"
    grid_data = to_target_grid(data, lon, lat, method)     # row0 = Norden
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