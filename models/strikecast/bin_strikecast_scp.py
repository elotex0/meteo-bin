#!/usr/bin/env python3
"""
StrikeCAST - Supercell Composite Parameter (SCP) als uint8-Binärraster (wie ICON-D2).

Ausgabe: bin/scp/scp_<YYYYMMDD>_<HHMM>   (ohne Endung, 1215x746 Bytes,
         row0 = Norden, 0..254 = Wert, 255 = NoData, Zeit = Ende des Intervalls
         in deutscher Ortszeit)
"""

import os
import glob
import re
import gc
from zoneinfo import ZoneInfo

import numpy as np
import xarray as xr
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from encoding_config import ENCODING, NODATA_U8, LEVELS

VAR_TYPE = "scp"
PRED_DIR = "data/output"
OUT_DIR  = "pngs/scp"
os.makedirs(OUT_DIR, exist_ok=True)

TZ_DE = ZoneInfo("Europe/Berlin")
enc = ENCODING[VAR_TYPE]

# Festes Raster = IDENTISCH zum ICON-D2-Raster (EPSG:4326, 0.02°, row0 = Norden).
GRID = {
    "extent": [-3.94, 43.18, 20.34, 58.08],   # lon_min, lat_min, lon_max, lat_max
    "width": 1215,
    "height": 746,
}


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

# ------------------------------
# Zielraster (= ICON-D2-Raster), Süd -> Nord aufsteigend
# ------------------------------
LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = GRID["extent"]
WIDTH, HEIGHT = GRID["width"], GRID["height"]
TARGET_LON = np.linspace(LON_MIN, LON_MAX, WIDTH)
TARGET_LAT = np.linspace(LAT_MIN, LAT_MAX, HEIGHT)


# -------------------------------------------------------
# Zeit-Hilfsfunktionen
# -------------------------------------------------------

def _to_de_local(ts):
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert(TZ_DE)


# -------------------------------------------------------
# Run-Label aus NC holen (nur fürs Logging)
# -------------------------------------------------------

def extract_run_label(ds):
    def _hour_to_label(raw):
        if raw is None:
            return None
        s = str(raw).strip()
        m = re.search(r"(?<!\d)(\d{2})\s*(?:Z|z|UTC|utc)\s*$", s)
        if m:
            h = int(m.group(1))
            if 0 <= h <= 23:
                return f"{h:02d}z"
        m = re.search(r"(?<!\d)(\d{2})\s*$", s)
        if m:
            h = int(m.group(1))
            if 0 <= h <= 23:
                return f"{h:02d}z"
        return None

    run_label = _hour_to_label(ds.attrs.get("run", None))
    if run_label is not None:
        return run_label

    for key in ("run_time_utc", "run_time", "forecast_reference_time", "analysis_time"):
        run_label = _hour_to_label(ds.attrs.get(key, None))
        if run_label is not None:
            return run_label

    return "??z"


# -------------------------------------------------------
# SCP aus Prädiktoren lesen
# -------------------------------------------------------

def extract_scp(ds2d):
    """Liest SCP direkt aus dem Prädiktor-Dataset."""
    if "SCP" not in ds2d:
        raise KeyError("Variable 'SCP' nicht im Dataset gefunden. "
                       "Bitte make_predict.py zuerst ausführen.")
    scp = ds2d["SCP"].values
    return np.clip(scp, 0.0, None)


# -------------------------------------------------------
# ECMWF-Gitter (0.25°) -> ICON-D2-Raster (0.02°)
# -------------------------------------------------------

def regrid_to_icon_grid(data, lat_1d, lon_1d):
    """data: 2D (lat, lon), lat/lon aufsteigend.
    Bilinear auf das feste Zielraster. Zielpixel außerhalb des
    ECMWF-Ausschnitts (schmaler Randstreifen) -> NaN (= NoData)."""
    interp = RegularGridInterpolator(
        (lat_1d, lon_1d), data, method="linear",
        bounds_error=False, fill_value=np.nan,
    )
    pts = np.stack(np.meshgrid(TARGET_LAT, TARGET_LON, indexing="ij"), axis=-1)
    return interp(pts)


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    files = sorted(glob.glob(os.path.join(PRED_DIR, "predictors_*.nc")))

    for f in files:
        ds = xr.open_dataset(f)

        run_label = extract_run_label(ds)

        # --- Koordinaten: 1D erwartet, aufsteigend sortiert ---
        lats = ds["latitude"].values
        lons = ds["longitude"].values

        lat_idx = np.where((lats >= LAT_MIN) & (lats <= LAT_MAX))[0]
        lon_idx = np.where((lons >= LON_MIN) & (lons <= LON_MAX))[0]
        lat_1d = lats[lat_idx]
        lon_1d = lons[lon_idx]

        lat_order = np.argsort(lat_1d)
        lon_order = np.argsort(lon_1d)
        lat_1d = lat_1d[lat_order]
        lon_1d = lon_1d[lon_order]
        lat_idx = lat_idx[lat_order]
        lon_idx = lon_idx[lon_order]

        # --- Zeitscheiben auswerten ---
        if "time" in ds.dims and ds.sizes["time"] == 2:
            prev_time = ds["time"].values[0]
            step_time = ds["time"].values[1]
            print(f"Processing step {prev_time} → {step_time}")

            interval_hours = int(ds.attrs.get("interval_hours", 3))

            ds_start = ds.isel(time=0, drop=True)
            ds_start = ds_start.isel(latitude=lat_idx, longitude=lon_idx)
            valid_to = pd.Timestamp(step_time)

        else:
            interval_hours = int(ds.attrs.get("interval_hours", 3))
            ds_start = ds.isel(time=0, drop=True) if "time" in ds.dims else ds
            ds_start = ds_start.isel(latitude=lat_idx, longitude=lon_idx)

            if "time" in ds.dims:
                valid_to = pd.Timestamp(ds["time"].values[0]) + pd.Timedelta(hours=interval_hours)
            else:
                valid_to = None

        print(f"Processing {f}")

        if valid_to is None:
            print(f"  ⚠️  {f}: keine Zeitinformation -> übersprungen")
            ds.close()
            continue

        scp = extract_scp(ds_start)
        print(f"  SCP: min={scp.min():.3f}  max={scp.max():.3f}  mean={scp.mean():.3f}")

        # --- 0.25° -> 0.02° (ICON-D2-Raster), Rand = NoData ---
        scp_fine = regrid_to_icon_grid(scp, lat_1d, lon_1d)

        # --- row0 = Norden, nach uint8 quantisieren ---
        q = encode_uint8(scp_fine[::-1], enc)
        assert q.shape == (HEIGHT, WIDTH), q.shape

        # --- Dateiname: scp_<Ende des Prognosezeitraums, dt. Ortszeit> ---
        vt_de = _to_de_local(valid_to)
        outname = f"{VAR_TYPE}_{vt_de.strftime('%Y%m%d_%H%M')}"
        outfile = os.path.join(OUT_DIR, outname)
        np.ascontiguousarray(q, dtype=np.uint8).tofile(outfile)

        print(f"  → {outfile}  ({WIDTH}x{HEIGHT}, {q.nbytes} Bytes, run={run_label}, interval={interval_hours}h)")

        ds.close()
        del scp, scp_fine, q
        gc.collect()

    print("✅ Fertig!")


if __name__ == "__main__":
    main()