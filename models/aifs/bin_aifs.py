import sys
import os
import gc
import json
import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from omfiles import OmFileReader

from encoding_config import NODATA_U8, LEVELS, ENCODING

# ------------------------------
# Eingabe-/Ausgabe
# ------------------------------
data_dir = sys.argv[1]        # z.B. "data/t2m"
output_dir = sys.argv[2]      # z.B. "aifs/t2m"
var_type = sys.argv[3]        # 't2m', 'wind', 'tp'
os.makedirs(output_dir, exist_ok=True)

if var_type not in ENCODING:
    print(f"Unbekannter var_type {var_type}")
    sys.exit(1)
enc = ENCODING[var_type]

# ------------------------------
# var_type -> Substring, der im Dateinamen der zugehoerigen .om Datei steht
# ------------------------------
OM_FILENAME_PATTERNS = {
    "t2m": "temperature_2m",
    "wind": "wind_gusts_10m",
    "tp6h": "precipitation",
}

pattern = OM_FILENAME_PATTERNS.get(var_type)
if pattern is None:
    print(f"var_type '{var_type}' hat noch kein Dateinamen-Muster in OM_FILENAME_PATTERNS")
    sys.exit(1)

# Umrechnung Rohwert -> Anzeige-Einheit, je Variable
UNIT_CONVERT = {
    "t2m": lambda v: v - 273.15 if np.nanmax(v) > 200 else v,  # K -> °C, nur falls noetig
    "wind": lambda v: v * 3.6,  # m/s -> km/h
}
convert = UNIT_CONVERT.get(var_type, lambda v: v)

# ------------------------------
# Globale Bbox der Quelldateien (bestaetigt via crs_wkt-Kind der .om Datei)
# ------------------------------
GLOBAL_LAT_MIN, GLOBAL_LON_MIN = -90.0, -180.0
GLOBAL_LAT_MAX, GLOBAL_LON_MAX = 90.0, 179.75

# Speicherreihenfolge der Lat-Achse in der Datei (Zeile 0 = Norden oder Sueden?).
# Verifiziert: Zeile 0 = Suedpol -> aufsteigend gespeichert.
LAT_STORED_DESCENDING = False

# Dimensionsreihenfolge im om-Array: shape=(721,1440,ntime), Zeit ist die LETZTE Dimension.

# ------------------------------
# Ausgaberaster (identisch zum ICON-D2-Raster)
# ------------------------------
# regular-lat-lon, EPSG:4326, row0 = Norden, Werte = Pixelzzentren.
# MUSS zu GRID in generate_metadata.py passen (dort wird die Dateigröße geprüft).
GRID_LON_MIN, GRID_LAT_MIN, GRID_LON_MAX, GRID_LAT_MAX = -3.94, 43.18, 20.34, 58.08
GRID_WIDTH, GRID_HEIGHT = 1215, 746

# Ausschnitt, der aus dem globalen Gitter gelesen wird (= Ausgaberaster)
extent = [GRID_LON_MIN, GRID_LON_MAX, GRID_LAT_MIN, GRID_LAT_MAX]  # lon_min, lon_max, lat_min, lat_max

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


def to_target_grid(data, lon, lat):
    """Reguläres Quellgitter (lat aufsteigend, lon aufsteigend) -> Ausgaberaster
    (HEIGHT, WIDTH), row0 = Norden, bilinear. Außerhalb des Quellgitters = NaN."""
    interp = RegularGridInterpolator(
        (lat, lon), data, method="linear", bounds_error=False, fill_value=np.nan
    )
    return interp(target_points).reshape(GRID_HEIGHT, GRID_WIDTH)


def compute_crop_indices(nlat, nlon):
    lat_res = (GLOBAL_LAT_MAX - GLOBAL_LAT_MIN) / (nlat - 1)
    lon_res = (GLOBAL_LON_MAX - GLOBAL_LON_MIN) / (nlon - 1)

    lon_min, lon_max, lat_min, lat_max = extent

    if LAT_STORED_DESCENDING:
        row_of = lambda v: (GLOBAL_LAT_MAX - v) / lat_res  # Zeile 0 = Norden
    else:
        row_of = lambda v: (v - GLOBAL_LAT_MIN) / lat_res  # Zeile 0 = Sueden

    r1, r2 = sorted([row_of(lat_min), row_of(lat_max)])
    row_start = max(0, int(np.floor(r1)) - 1)
    row_end = min(nlat - 1, int(np.ceil(r2)) + 1)

    col_of = lambda v: (v - GLOBAL_LON_MIN) / lon_res
    col_start = max(0, int(np.floor(col_of(lon_min))) - 1)
    col_end = min(nlon - 1, int(np.ceil(col_of(lon_max))) + 1)

    return row_start, row_end, col_start, col_end, lat_res, lon_res


def crop_lat_lon_arrays(row_start, row_end, col_start, col_end, lat_res, lon_res, nlat):
    if LAT_STORED_DESCENDING:
        lat_full = GLOBAL_LAT_MAX - np.arange(nlat) * lat_res
    else:
        lat_full = GLOBAL_LAT_MIN + np.arange(nlat) * lat_res
    lat_crop = lat_full[row_start:row_end + 1]
    lon_crop = GLOBAL_LON_MIN + np.arange(col_start, col_end + 1) * lon_res
    return lat_crop, lon_crop


def load_valid_times(root, ntime, om_path):
    """Liest die echten (nicht-gleichmaessigen) Zeitschritte direkt aus dem
    'time'-Kind der om-Datei (Unix-Timestamps in Sekunden, UTC)."""
    try:
        time_child = root.get_child_by_name("time")
    except Exception as e:
        raise ValueError(
            f"{om_path}: kein 'time'-Kind gefunden - kann Zeitstempel nicht bestimmen ({e})"
        )

    if not time_child.is_array or time_child.shape[0] != ntime:
        raise ValueError(
            f"{om_path}: 'time'-Kind passt nicht (shape={time_child.shape}, erwartet ntime={ntime})"
        )

    raw = time_child.read_array((slice(0, ntime),))
    return [dt.datetime.fromtimestamp(int(t), tz=dt.timezone.utc) for t in raw]


# ------------------------------
# Dateien durchgehen
# ------------------------------
all_files_global = sorted(f for f in os.listdir(data_dir) if f.endswith(".om"))
matching_files = [f for f in all_files_global if pattern.lower() in f.lower()]

if not matching_files:
    print(f"Keine .om Datei in {data_dir} gefunden, die zu '{pattern}' passt "
          f"(gefunden: {all_files_global})")

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

for filename in matching_files:
    om_path = os.path.join(data_dir, filename)

    with OmFileReader(om_path) as root:
        # root ist gleichzeitig das Datenarray UND hat Metadaten-Kinder
        # (time, crs_wkt, unit, forecast_reference_time, coordinates, created_at).
        if not root.is_array:
            print(f"{om_path}: root ist kein Array (is_group={root.is_group}) - überspringe")
            continue

        nlat, nlon, ntime = root.shape  # bestaetigt: (lat, lon, time)

        row_start, row_end, col_start, col_end, lat_res, lon_res = compute_crop_indices(nlat, nlon)
        lat_crop, lon_crop = crop_lat_lon_arrays(row_start, row_end, col_start, col_end, lat_res, lon_res, nlat)

        # Ganzen Ausschnitt fuer ALLE Zeitschritte auf einmal lesen -
        # passt zum Chunk-Layout (Zeitachse wird ohnehin komplett pro Chunk
        # gespeichert), spart also viele einzelne read_array-Aufrufe.
        data_all = root.read_array((
            slice(row_start, row_end + 1),
            slice(col_start, col_end + 1),
            slice(0, ntime),
        ))  # shape: (nrows, ncols, ntime)

        valid_times_utc = load_valid_times(root, ntime, om_path)

        for t_idx in range(ntime):
            data = convert(np.asarray(data_all[:, :, t_idx], dtype=np.float64))

            if LAT_STORED_DESCENDING:
                data = data[::-1, :]  # Zeile 0 -> Sueden, wie RegularGridInterpolator erwartet
                lat_asc = lat_crop[::-1]
            else:
                lat_asc = lat_crop

            # ------------------------------
            # Quellgitter -> Ausgaberaster -> uint8
            # ------------------------------
            grid_data = to_target_grid(data, lon_crop, lat_asc)   # row0 = Norden
            q = encode_uint8(grid_data, enc)

            valid_time_local = valid_times_utc[t_idx].astimezone(ZoneInfo("Europe/Berlin"))
            outname = f"{var_type}_{valid_time_local:%Y%m%d_%H%M}"
            out_path = os.path.join(output_dir, outname)
            np.ascontiguousarray(q, dtype=np.uint8).tofile(out_path)

            meta["files"].append({
                "file": outname,
                "valid_time_utc": valid_times_utc[t_idx].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "valid_time_local": valid_time_local.strftime("%Y-%m-%dT%H:%M:%S%z"),
            })

            print(f"{filename} t_idx={t_idx} -> {outname} ({GRID_WIDTH}x{GRID_HEIGHT}, {q.nbytes} Bytes)")

            del data, grid_data, q

        del data_all
        gc.collect()

print("META: " + json.dumps(meta, ensure_ascii=False))
print("Fertig!")