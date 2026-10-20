#!/usr/bin/env python3

import os
import glob
import re
import gc
from zoneinfo import ZoneInfo

import numpy as np
import xarray as xr
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

from encoding_config import ENCODING, NODATA_U8, LEVELS

VAR_TYPE = "gewitter"
PRED_DIR = "data/output"
LUT_PATH = "data/lut/lightning_lut.nc"
OUT_DIR  = "pngs/gewitter"
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
GRID_STEP_DEG = (LON_MAX - LON_MIN) / (WIDTH - 1)   # 0.02°

SMOOTH_KM = 40.0   # Gauß-Glättung (wie bisher)


# -------------------------------------------------------
# Zeit-Hilfsfunktionen
# -------------------------------------------------------

def _to_de_local(ts):
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert(TZ_DE)


# -------------------------------------------------------
# LUT laden
# -------------------------------------------------------

lut = xr.open_dataset(LUT_PATH)


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
# Wahrscheinlichkeit (unverändert)
# -------------------------------------------------------

def compute_probability(ds2d, lut, interval_hours=1):

    mu_mixr = np.clip(ds2d["MU_MIXR"].values, float(lut["MU_MIXR"].min()), float(lut["MU_MIXR"].max()))
    lsm     = np.clip(ds2d["lsm"].values,     float(lut["lsm"].min()),     float(lut["lsm"].max()))
    mcpr    = np.clip(ds2d["mcpr"].values,     float(lut["mcpr"].min()),    float(lut["mcpr"].max()))
    rhmean  = np.clip(ds2d["RHmean"].values,   float(lut["meanRH_500-850"].min()), float(lut["meanRH_500-850"].max()))
    mu_li   = np.clip(ds2d["MU_LI"].values,    float(lut["MU_LI"].min()),   float(lut["MU_LI"].max()))

    prob = lut["prob_lightning_lt1h"].interp(
        MU_MIXR=("points", mu_mixr.flatten()),
        lsm=("points", lsm.flatten()),
        mcpr=("points", mcpr.flatten()),
        **{"meanRH_500-850": ("points", rhmean.flatten())},
        MU_LI=("points", mu_li.flatten()),
        method="linear",
    )
    prob = prob.values.reshape(ds2d["MU_LI"].shape)
    prob = np.nan_to_num(prob, nan=0.0)
    prob = np.clip(prob, 0.0, 1.0)

    # --- Debug: Ausreißer diagnostizieren ---
    high_mask = prob > 0.5
    if high_mask.any():
        print(f"\n⚠️  {high_mask.sum()} Gitterpunkte mit prob > 50%")
        print(f"   prob:     min={prob[high_mask].min():.3f}  max={prob[high_mask].max():.3f}")
        print(f"   MU_LI:    min={ds2d['MU_LI'].values[high_mask].min():.2f}  max={ds2d['MU_LI'].values[high_mask].max():.2f}")
        print(f"   MU_MIXR:  min={ds2d['MU_MIXR'].values[high_mask].min():.2f}  max={ds2d['MU_MIXR'].values[high_mask].max():.2f}")
        print(f"   mcpr:     min={ds2d['mcpr'].values[high_mask].min():.6f}  max={ds2d['mcpr'].values[high_mask].max():.6f}")
        print(f"   RHmean:   min={ds2d['RHmean'].values[high_mask].min():.1f}  max={ds2d['RHmean'].values[high_mask].max():.1f}")
        print(f"   z_sfc:    min={ds2d['z_sfc'].values[high_mask].min():.0f}  max={ds2d['z_sfc'].values[high_mask].max():.0f}  ← Höhe in m")
        print(f"   CIN:      min={ds2d['CIN'].values[high_mask].min():.1f}  max={ds2d['CIN'].values[high_mask].max():.1f}")

        idx = np.unravel_index(np.argmax(prob), prob.shape)
        print(f"\n   Höchster Wert bei Index {idx}:")
        print(f"   prob={prob[idx]:.3f}  MU_LI={ds2d['MU_LI'].values[idx]:.2f}  MU_MIXR={ds2d['MU_MIXR'].values[idx]:.2f}  mcpr={ds2d['mcpr'].values[idx]:.6f}  z_sfc={ds2d['z_sfc'].values[idx]:.0f}m")

    ih = max(1, int(interval_hours))
    prob_interval = 1.0 - np.power(1.0 - prob, ih)
    return np.clip(prob_interval * 100.0, 0.0, 100.0)


# -------------------------------------------------------
# ECMWF-Gitter (0.25°) -> ICON-D2-Raster (0.02°)
# -------------------------------------------------------

def regrid_to_icon_grid(data, lat_1d, lon_1d):
    """data: 2D (lat, lon), lat/lon aufsteigend.
    Bilinear auf das feste Zielraster. Zielpixel außerhalb des
    ECMWF-Ausschnitts (schmaler Randstreifen) werden per Randwert gefüllt
    (nötig für die Glättung) und über die zurückgegebene Maske
    'outside' später auf NaN (= NoData) gesetzt."""
    lat_c = np.clip(TARGET_LAT, lat_1d[0], lat_1d[-1])
    lon_c = np.clip(TARGET_LON, lon_1d[0], lon_1d[-1])

    interp = RegularGridInterpolator((lat_1d, lon_1d), data, method="linear")
    pts = np.stack(np.meshgrid(lat_c, lon_c, indexing="ij"), axis=-1)
    out = interp(pts)

    outside = (((TARGET_LAT < lat_1d[0]) | (TARGET_LAT > lat_1d[-1]))[:, None]
               | ((TARGET_LON < lon_1d[0]) | (TARGET_LON > lon_1d[-1]))[None, :])
    return out, outside


def smooth_probability(prob_fine, smooth_km=SMOOTH_KM):
    sigma_px = smooth_km / (GRID_STEP_DEG * 111.0)
    prob_fine = gaussian_filter(prob_fine, sigma=sigma_px, mode="nearest")
    return np.clip(prob_fine, 0.0, 100.0)


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

        # --- Zeitscheiben auswerten (unverändert) ---
        if "time" in ds.dims and ds.sizes["time"] == 2:
            prev_time = ds["time"].values[0]
            step_time = ds["time"].values[1]
            print(f"Processing step {prev_time} → {step_time}")

            interval_hours = int(ds.attrs.get("interval_hours", 3))
            if interval_hours < 1:
                dt_hours = (step_time - prev_time) / np.timedelta64(1, "h")
                interval_hours = max(1, int(round(float(dt_hours))))

            ds_start = ds.isel(time=0, drop=True)
            ds_start = ds_start.isel(latitude=lat_idx, longitude=lon_idx)
            prob = compute_probability(ds_start, lut, interval_hours=interval_hours)
            valid_to = pd.Timestamp(step_time)

        else:
            interval_hours = int(ds.attrs.get("interval_hours", 3))
            ds_start = ds.isel(time=0, drop=True) if "time" in ds.dims else ds
            ds_start = ds_start.isel(latitude=lat_idx, longitude=lon_idx)
            prob = compute_probability(ds_start, lut, interval_hours=interval_hours)

            if "time" in ds.dims:
                valid_to = pd.Timestamp(ds["time"].values[0]) + pd.Timedelta(hours=interval_hours)
            else:
                valid_to = None

        print(f"  prob: min={prob.min():.1f}%  max={prob.max():.1f}%  mean={prob.mean():.1f}%")

        if valid_to is None:
            print(f"  ⚠️  {f}: keine Zeitinformation -> übersprungen")
            ds.close()
            continue

        # --- 0.25° -> 0.02° (ICON-D2-Raster), glätten, Rand = NoData ---
        prob_fine, outside = regrid_to_icon_grid(prob, lat_1d, lon_1d)
        prob_fine = np.clip(prob_fine, 0.0, 100.0)
        prob_fine = smooth_probability(prob_fine)
        prob_fine[outside] = np.nan

        # --- row0 = Norden, nach uint8 quantisieren ---
        q = encode_uint8(prob_fine[::-1], enc)
        assert q.shape == (HEIGHT, WIDTH), q.shape

        # --- Dateiname: gewitter_<Ende des Prognosezeitraums, dt. Ortszeit> ---
        vt_de = _to_de_local(valid_to)
        outname = f"{VAR_TYPE}_{vt_de.strftime('%Y%m%d_%H%M')}"
        outfile = os.path.join(OUT_DIR, outname)
        np.ascontiguousarray(q, dtype=np.uint8).tofile(outfile)

        print(f"  → {outfile}  ({WIDTH}x{HEIGHT}, {q.nbytes} Bytes, run={run_label}, interval={interval_hours}h)")

        ds.close()
        del prob, prob_fine, q
        gc.collect()

    print("✅ Fertig!")


if __name__ == "__main__":
    main()