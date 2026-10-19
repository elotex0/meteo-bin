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
output_dir = sys.argv[2]      # z.B. "gfs/t2m"
var_type = sys.argv[3]        # 't2m', 'wind', 'tp', 'tp_acc'
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


def normalize_grid(data, lon, lat):
    """GFS liefert Längen 0..360 und Breiten von Nord nach Süd.
    -> Längen auf -180..180 umklappen, beide Achsen aufsteigend sortieren."""
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    order_lon = np.argsort(lon)
    order_lat = np.argsort(lat)
    return data[np.ix_(order_lat, order_lon)], lon[order_lon], lat[order_lat]


def to_target_grid(data, lon, lat):
    """Reguläres Quellgitter (lat, lon beliebig sortiert, lon 0..360 erlaubt)
    -> Ausgaberaster (HEIGHT, WIDTH), row0 = Norden, bilinear.
    Außerhalb des Quellgitters = NaN."""
    data, lon, lat = normalize_grid(data, lon, lat)
    interp = RegularGridInterpolator(
        (lat, lon), data, method="linear", bounds_error=False, fill_value=np.nan
    )
    return interp(target_points).reshape(GRID_HEIGHT, GRID_WIDTH)


def get_valid_time(ds):
    """Gültigkeitszeit (UTC) und Modelllauf-Zeit (UTC) aus dem Dataset."""
    run_time_utc = pd.to_datetime(ds["time"].values) if "time" in ds else None
    if "valid_time" in ds:
        valid_time_raw = ds["valid_time"].values
        valid_time_utc = pd.to_datetime(valid_time_raw[0]) if np.ndim(valid_time_raw) > 0 else pd.to_datetime(valid_time_raw)
    else:
        step = pd.to_timedelta(ds["step"].values[0])
        valid_time_utc = run_time_utc + step
    return run_time_utc, valid_time_utc


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


def save_field(data, lon, lat, valid_time_utc):
    """Ein Feld -> Ausgaberaster -> uint8-Datei + Eintrag in meta. Rückgabe: Dateiname."""
    valid_time_local = valid_time_utc.tz_localize("UTC").astimezone(ZoneInfo("Europe/Berlin"))

    grid_data = to_target_grid(data, lon, lat)      # row0 = Norden
    q = encode_uint8(grid_data, enc)

    outname = f"{var_type}_{valid_time_local:%Y%m%d_%H%M}"
    out_path = os.path.join(output_dir, outname)
    np.ascontiguousarray(q, dtype=np.uint8).tofile(out_path)

    meta["files"].append({
        "file": outname,
        "valid_time_utc": valid_time_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "valid_time_local": valid_time_local.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    return outname


# ------------------------------
# tp_acc: Akkumulation über alle Dateien eines Modellruns
# ------------------------------
def process_tp_acc_files(data_dir):
    grib_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".grib2")])

    if not grib_files:
        print("Keine GRIB2-Dateien gefunden!")
        return

    file_info = []
    for filename in grib_files:
        path = os.path.join(data_dir, filename)
        try:
            ds = cfgrib.open_dataset(path)
            if "tp" not in ds:
                ds.close()
                continue

            run_time_utc, valid_time_utc = get_valid_time(ds)
            forecast_hour = int((valid_time_utc - run_time_utc).total_seconds() / 3600)

            file_info.append({
                'filename': filename,
                'path': path,
                'run_time': run_time_utc,
                'valid_time': valid_time_utc,
                'forecast_hour': forecast_hour
            })
            ds.close()
        except Exception as e:
            print(f"Fehler beim Lesen von {filename}: {e}")
            continue

    if not file_info:
        print("Keine gültigen tp-Dateien gefunden!")
        return

    file_info_df = pd.DataFrame(file_info)
    file_info_df = file_info_df.sort_values(['run_time', 'forecast_hour'])

    for run_time, group in file_info_df.groupby('run_time'):
        print(f"\n=== Verarbeite Modellrun: {run_time} ===")

        accumulated_tp = None
        previous_tp = None

        for idx, row in group.iterrows():
            ds = cfgrib.open_dataset(row['path'])
            current_tp = np.asarray(ds["tp"].values, dtype=np.float64)
            if current_tp.ndim == 3:
                current_tp = current_tp[0]
            current_tp[current_tp < 0] = 0

            lon = np.asarray(ds["longitude"].values, dtype=np.float64)
            lat = np.asarray(ds["latitude"].values, dtype=np.float64)
            ds.close()

            if current_tp.shape != (lat.size, lon.size):
                print(f"{row['filename']}: Datenform {current_tp.shape} passt nicht zum lat/lon-Gitter, überspringe")
                continue

            if accumulated_tp is None:
                accumulated_tp = current_tp.copy()
            else:
                delta_tp = current_tp - previous_tp
                delta_tp[delta_tp < 0] = 0
                accumulated_tp = accumulated_tp + delta_tp

            previous_tp = current_tp.copy()

            outname = save_field(accumulated_tp, lon, lat, row['valid_time'])

            print(f"  Forecast hour {row['forecast_hour']:03d}: "
                  f"Max akkumuliert = {np.nanmax(accumulated_tp):.1f} mm -> {outname}")

            del current_tp
            gc.collect()


# ------------------------------
# Dateien durchgehen
# ------------------------------
if var_type == "tp_acc":
    process_tp_acc_files(data_dir)
elif var_type in ("t2m", "wind", "tp"):
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".grib2"):
            continue
        path = os.path.join(data_dir, filename)

        if var_type == "tp":
            ds = cfgrib.open_dataset(path, filter_by_keys={'stepType': 'instant'})
        else:
            ds = cfgrib.open_dataset(path)

        if var_type == "t2m":
            if "t2m" not in ds:
                print(f"Keine t2m in {filename}")
                ds.close()
                continue
            data = ds["t2m"].values - 273.15
        elif var_type == "wind":
            if "gust" not in ds:
                print(f"Keine 'gust' in {filename} - vorhandene Variablen: {list(ds.data_vars)}")
                ds.close()
                continue
            data = ds["gust"].values * 3.6  # m/s -> km/h
            data[data < 0] = 0
        elif var_type == "tp":
            if "prate" not in ds:
                print(f"Keine prate in {filename} - vorhandene Variablen: {list(ds.data_vars)}")
                ds.close()
                continue
            data = ds["prate"].values * 3600  # kg/m²/s -> mm/h

        data = np.asarray(data)
        if data.ndim == 3:
            data = data[0]

        lon = np.asarray(ds["longitude"].values, dtype=np.float64)
        lat = np.asarray(ds["latitude"].values, dtype=np.float64)
        _, valid_time_utc = get_valid_time(ds)
        ds.close()

        if data.shape != (lat.size, lon.size):
            print(f"{filename}: Datenform {data.shape} passt nicht zum lat/lon-Gitter, überspringe")
            continue

        outname = save_field(data, lon, lat, valid_time_utc)
        print(f"{filename} -> {outname} ({GRID_WIDTH}x{GRID_HEIGHT})")

        del data
        gc.collect()
else:
    print(f"Unbekannter var_type {var_type}")

print("META: " + json.dumps(meta, ensure_ascii=False))
print("Fertig!")