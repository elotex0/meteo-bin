#!/usr/bin/env python3
"""
process_gewitter.py  –  v7 (per-step files, steps 6-144)
"""

import os
import numpy as np
import xarray as xr
from datetime import datetime, timedelta

# -------------------------------------------------------
# Konfiguration
# -------------------------------------------------------
date = os.getenv("DATE", "20260708")
run  = int(os.getenv("RUN", 0))

BASE_DIR   = os.path.join("data", "gewitter")
OUTPUT_DIR = os.path.join("data", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

steps_all = list(range(6, 82, 3))   # 6, 9, 12, … 144

g   = 9.80665
rd  = 287.04
rv  = 461.50
eps = rd / rv
cpd = 1005.7

LAT_MIN, LAT_MAX = 43.18, 58.08
LON_MIN, LON_MAX = -3.94, 20.34

PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# -------------------------------------------------------
# Dateipfade  – {:03d} + z immer step000
# -------------------------------------------------------
def sfc_path(param, step):
    if param == "z":
        return os.path.join(BASE_DIR, "z", "z_step000.grib2")
    return os.path.join(BASE_DIR, param, f"{param}_step{step:03d}.grib2")

def pl_path(param, level, step):
    return os.path.join(BASE_DIR, f"{param}_pl", f"{param}_pl{level}_step{step:03d}.grib2")

def file_ok(*paths):
    return all(os.path.exists(p) for p in paths)

# -------------------------------------------------------
# GRIB lesen
# -------------------------------------------------------
_ds_cache = {}

def _open_grib(path, filter_keys=None):
    if path not in _ds_cache:
        kwargs = {"engine": "cfgrib", "backend_kwargs": {"indexpath": ""}}
        if filter_keys:
            kwargs["filter_by_keys"] = filter_keys
        ds = xr.open_dataset(path, **kwargs)
        ds = ds.sel(
            latitude=slice(LAT_MAX, LAT_MIN),
            longitude=slice(LON_MIN, LON_MAX),
        )
        _ds_cache[path] = ds
    return _ds_cache[path]


def read_sfc(param, step):
    path = sfc_path(param, step)   # z gibt automatisch step000 zurück
    ds   = _open_grib(path)
    var  = list(ds.data_vars)[0]
    da   = ds[var]
    for dim in ["step", "valid_time", "time"]:
        if dim in da.dims:
            da = da.isel({dim: 0})
    return da.values, ds["latitude"].values, ds["longitude"].values


def read_pl(param, step):
    arrays = []
    levels_out = []
    lats = lons = None

    for level in sorted(PRESSURE_LEVELS):
        path = pl_path(param, level, step)
        if not os.path.exists(path):
            continue
        ds  = _open_grib(path, filter_keys={"typeOfLevel": "isobaricInhPa"})
        var = list(ds.data_vars)[0]
        da  = ds[var]
        for dim in ["step", "valid_time", "time", "isobaricInhPa"]:
            if dim in da.dims:
                da = da.isel({dim: 0})
        arrays.append(da.values)
        levels_out.append(level)
        if lats is None:
            lats = ds["latitude"].values
            lons = ds["longitude"].values

    data   = np.stack(arrays, axis=0)
    levels = np.array(levels_out, dtype=float)
    sort_idx = np.argsort(levels)
    return data[sort_idx], levels[sort_idx], lats, lons


def clear_cache():
    _ds_cache.clear()


# -------------------------------------------------------
# Prüfung ob alle Dateien eines Steps vorhanden sind
# -------------------------------------------------------
def check_files(prev_step, step):
    missing = []

    # Oberflächenparameter (beide Steps nötig wegen tp-Differenz)
    for param in ["2t", "2d", "sp", "tp", "lsm", "mucape"]:
        for s in [prev_step, step]:
            p = sfc_path(param, s)
            if not os.path.exists(p):
                missing.append(p)

    # Orographie – immer step000
    z = sfc_path("z", 0)   # gibt z_step000.grib2
    if not os.path.exists(z):
        missing.append(z)

    # Druckniveaus – nur prev_step
    for param in ["t", "q", "r", "u", "v", "gh"]:
        for level in PRESSURE_LEVELS:
            p = pl_path(param, level, prev_step)
            if not os.path.exists(p):
                missing.append(p)

    return missing


# -------------------------------------------------------
# Meteorologische Hilfsfunktionen  (unverändert)
# -------------------------------------------------------
def calc_cin(t2m, td2m, sp, t_pl, q_pl, z_pl, levels):
    """
    Surface-based CIN [J/kg] via pseudoadiabatischem Parcel-Ascent (Bolton).
    Vereinfachung: unterhalb LCL wird Buoyancy = 0 gesetzt (wie im ECMWF-Original).
    CIN wird nur bis zum LFC (erste Schicht mit positiver Buoyancy) aufsummiert.
    """
    sp_hPa = sp / 100.0
    q_sfc  = np.clip(0.622 * es_buck(td2m) / (sp_hPa - es_buck(td2m)), 0.0, 0.06)
    theta_e_parcel = theta_ep_bolton(t2m, sp_hPa, q_sfc)

    # LCL (Bolton) – Druck
    e_sfc = es_buck(td2m)
    T_LCL = 2840.0 / (3.5 * np.log(t2m) - np.log(np.maximum(e_sfc, 1e-9)) - 4.805) + 55.0
    p_LCL = sp_hPa * (T_LCL / t2m) ** (cpd / rd)   # cpd = 1005.7, ggf. oben ergänzen

    nlev = t_pl.shape[0]
    Lv = 2.5e6
    Tv_parcel = np.zeros_like(t_pl)

    # Parcel-Temperatur auf jeder Druckfläche (moist-adiabatisch, theta_e erhalten)
    for k in range(nlev):
        p_k = levels[k]
        T_guess = t_pl[k].copy()
        for _ in range(6):
            es_k      = es_buck(T_guess)
            q_sat_k   = np.clip(0.622 * es_k / (p_k - es_k), 0.0, 0.06)
            theta_e_g = theta_ep_bolton(T_guess, p_k, q_sat_k)
            dtheta_dT = theta_e_g / T_guess * (1.0 + Lv * q_sat_k / (rd * T_guess))
            T_guess   = T_guess + (theta_e_parcel - theta_e_g) / np.maximum(dtheta_dT, 1e-6)
            T_guess   = np.clip(T_guess, 150.0, 340.0)
        q_sat_final  = np.clip(0.622 * es_buck(T_guess) / (p_k - es_buck(T_guess)), 0.0, 0.06)
        Tv_parcel[k] = T_guess * (1.0 + 0.61 * q_sat_final)

    Tv_env   = t_pl * (1.0 + 0.61 * q_pl)
    buoyancy = Tv_parcel - Tv_env

    # unterhalb LCL: keine Buoyancy (Näherung wie im Original)
    p_grid   = levels[:, None, None]
    buoyancy = np.where(p_grid > p_LCL[None, :, :], 0.0, buoyancy)

    # levels sind aufsteigend sortiert (50…1000 hPa), d.h. z_pl[0]=oben, z_pl[-1]=unten
    # für die Integration von unten nach oben umkehren:
    buoyancy_b = buoyancy[::-1]
    z_b        = z_pl[::-1]
    t_b        = t_pl[::-1]

    dz    = z_b[1:] - z_b[:-1]
    b_bot = buoyancy_b[:-1]
    b_top = buoyancy_b[1:]
    t_bot = t_b[:-1]
    t_top = t_b[1:]

    d_energy = ((b_bot + b_top) / (t_bot + t_top)) * g * dz
    dCAPE = np.where(d_energy > 0, d_energy, 0.0)
    dCIN  = np.where(d_energy < 0, d_energy, 0.0)

    has_lfc   = np.any(dCAPE > 0, axis=0)
    lfc_level = np.argmax(dCAPE > 0, axis=0)

    level_idx   = np.arange(dCIN.shape[0])[:, None, None]
    dCIN_masked = np.where(level_idx <= lfc_level[None, :, :], dCIN, 0.0)

    cin = np.where(has_lfc, np.sum(dCIN_masked, axis=0), 0.0)
    return cin


def es_buck(T_K):
    Tc = T_K - 273.16
    return 6.1121 * np.exp((18.678 - Tc / 234.5) * (Tc / (257.14 + Tc)))

def calc_rh_2m(t2m, td2m):
    return np.clip(100.0 * es_buck(td2m) / es_buck(t2m), 0.0, 100.0)

def calc_mixr_2m(td2m, sp_Pa):
    e = es_buck(td2m)
    sp_hPa = sp_Pa / 100.0
    q = 0.622 * e / (sp_hPa - e)
    return np.clip(q * 1000.0, 0.0, 60.0)

def calc_mean_rh(r_pl, levels, p_min=500, p_max=850):
    mask = (levels >= p_min) & (levels <= p_max)
    if not mask.any():
        return np.full(r_pl.shape[1:], np.nan)
    return np.mean(r_pl[mask], axis=0)

def theta_ep_bolton(T_K, p_hPa, q_kgkg):
    r = q_kgkg / np.maximum(1.0 - q_kgkg, 1e-9)
    e = np.maximum(r * p_hPa / (r + eps), 1e-6)
    T_lcl = 2840.0 / (3.5 * np.log(np.maximum(T_K, 1.0)) - np.log(np.maximum(e, 1e-9)) - 4.805) + 55.0
    theta_e = (T_K * (1000.0 / np.maximum(p_hPa, 1.0)) ** (0.2854 * (1.0 - 0.28 * r))
               * np.exp(r * (1.0 + 0.81 * r) * (3376.0 / np.maximum(T_lcl, 1.0) - 2.54)))
    return theta_e

def calc_deg0l(t_pl, z_pl, levels):
    t_c = t_pl - 273.15
    nlev, nlat, nlon = t_pl.shape
    deg0l = np.full((nlat, nlon), np.nan)
    for i in range(nlev - 1, 0, -1):
        j = i - 1
        t_bot = t_c[i]; t_top = t_c[j]
        z_bot = z_pl[i]; z_top = z_pl[j]
        crossing = (t_bot >= 0.0) & (t_top < 0.0) & np.isnan(deg0l)
        dT = t_bot - t_top
        safe_dT = np.where(np.abs(dT) < 1e-6, 1e-6, dT)
        frac = t_bot / safe_dT
        height = z_bot + frac * (z_top - z_bot)
        deg0l = np.where(crossing, height, deg0l)
    return np.where(np.isnan(deg0l), 0.0, deg0l)

def calc_mu_li(t_pl, z_pl, q_pl, levels, t2m, td2m, sp_Pa):
    sp_hPa = sp_Pa / 100.0
    idx_500 = np.argmin(np.abs(levels - 500.0))
    t_env_500 = t_pl[idx_500]
    q_sfc = np.clip(0.622 * es_buck(td2m) / (sp_hPa - es_buck(td2m)), 0.0, 0.06)
    theta_e_parcel = theta_ep_bolton(t2m, sp_hPa, q_sfc)
    Lv = 2.5e6
    T_guess = t_env_500.copy()
    for _ in range(6):
        es_500    = es_buck(T_guess)
        q_sat_500 = np.clip(0.622 * es_500 / (500.0 - es_500), 0.0, 0.06)
        theta_e_guess = theta_ep_bolton(T_guess, 500.0, q_sat_500)
        dtheta_dT = theta_e_guess / T_guess * (1.0 + Lv * q_sat_500 / (rd * T_guess))
        T_guess = T_guess + (theta_e_parcel - theta_e_guess) / np.maximum(dtheta_dT, 1e-6)
        T_guess = np.clip(T_guess, 200.0, 320.0)
    return np.clip(t_env_500 - T_guess, -20.0, 20.0)

def calc_lcl_height(t2m, td2m):
    return np.maximum((t2m - td2m) * 125.0, 0.0)

def interpolate_to_height(z_3d, param_3d, target_z):
    nlev, nlat, nlon = z_3d.shape
    ngrid = nlat * nlon
    z2 = z_3d.reshape(nlev, ngrid).T
    p2 = param_3d.reshape(nlev, ngrid).T
    tgt = target_z.ravel()
    below = z2 < tgt[:, None]
    idx_bot = np.argmax(below, axis=1)
    no_below = ~below.any(axis=1)
    idx_bot = np.where(no_below, nlev - 1, idx_bot)
    idx_top = np.clip(idx_bot - 1, 0, nlev - 1)
    z_bot = z2[np.arange(ngrid), idx_bot]
    z_top = z2[np.arange(ngrid), idx_top]
    p_bot = p2[np.arange(ngrid), idx_bot]
    p_top = p2[np.arange(ngrid), idx_top]
    dz = z_top - z_bot
    frac = np.clip(np.where(np.abs(dz) > 1.0, (tgt - z_bot) / dz, 0.5), 0.0, 1.0)
    return (p_bot + frac * (p_top - p_bot)).reshape(nlat, nlon)

def calc_mean_wind_1_3km(z_3d, u_3d, v_3d, z_sfc):
    u_sum = np.zeros_like(z_sfc)
    v_sum = np.zeros_like(z_sfc)
    for dz in [1000, 2000, 3000]:
        u_sum += interpolate_to_height(z_3d, u_3d, z_sfc + dz)
        v_sum += interpolate_to_height(z_3d, v_3d, z_sfc + dz)
    return np.sqrt((u_sum / 3.0) ** 2 + (v_sum / 3.0) ** 2)

def calc_eff_bulk_shear(z_3d, u_3d, v_3d, z_sfc, cape):
    u_sfc = interpolate_to_height(z_3d, u_3d, z_sfc)
    v_sfc = interpolate_to_height(z_3d, v_3d, z_sfc)
    u_top = interpolate_to_height(z_3d, u_3d, z_sfc + 3000.0)
    v_top = interpolate_to_height(z_3d, v_3d, z_sfc + 3000.0)
    bs = np.sqrt((u_top - u_sfc) ** 2 + (v_top - v_sfc) ** 2)
    return np.where(cape < 10.0, 0.0, bs)

def calc_bunkers_storm_motion(u_pl, v_pl, z_pl, z_sfc, levels):
    """
    Vereinfachter Bunkers-Sturmvektor.
    Mittlerer Wind 0–6 km + Rechtsperforation (Deviation ~7.5 m/s senkrecht).
    """
    u_mean = interpolate_to_height(z_pl, u_pl, z_sfc + 500)
    v_mean = interpolate_to_height(z_pl, v_pl, z_sfc + 500)
    for dz in [1000, 2000, 3000, 4000, 5000, 6000]:
        u_mean += interpolate_to_height(z_pl, u_pl, z_sfc + dz)
        v_mean += interpolate_to_height(z_pl, v_pl, z_sfc + dz)
    u_mean /= 7.0
    v_mean /= 7.0

    u_shear = interpolate_to_height(z_pl, u_pl, z_sfc + 6000) - \
              interpolate_to_height(z_pl, u_pl, z_sfc + 500)
    v_shear = interpolate_to_height(z_pl, v_pl, z_sfc + 6000) - \
              interpolate_to_height(z_pl, v_pl, z_sfc + 500)

    shear_mag = np.maximum(np.sqrt(u_shear**2 + v_shear**2), 1e-6)
    # Rechtsdeviation: Normalvektor 90° rechts vom Schervektor
    u_storm = u_mean + 7.5 * ( v_shear / shear_mag)
    v_storm = v_mean + 7.5 * (-u_shear / shear_mag)
    return u_storm, v_storm

def calc_srh(u_pl, v_pl, z_pl, z_sfc, levels, layer_m=3000):
    """
    Storm-Relative Helicity 0–3 km AGL mit Bunkers-Sturmvektor.
    """
    u_storm, v_storm = calc_bunkers_storm_motion(u_pl, v_pl, z_pl, z_sfc, levels)

    srh = np.zeros_like(z_sfc)
    nlev = len(levels)

    for k in range(nlev - 1):
        z_bot = z_pl[k]
        z_top = z_pl[k + 1]

        in_layer = (
            (z_bot - z_sfc < layer_m) &
            (z_top - z_sfc < layer_m) &
            (z_bot - z_sfc >= 0.0) &
            (z_top - z_sfc >= 0.0)
        )

        du1 = u_pl[k]     - u_storm
        dv1 = v_pl[k]     - v_storm
        du2 = u_pl[k + 1] - u_storm
        dv2 = v_pl[k + 1] - v_storm

        srh += np.where(in_layer, du1 * dv2 - du2 * dv1, 0.0)

    return srh

def calc_stp(mucape, ml_lcl, srh, mu_eff_bs):
    """
    Significant Tornado Parameter – Thompson et al. 2004/2012.
    Alle Terme auf [0, 1] geclippt vor Multiplikation.
    Referenz: SPC Mesoscale Analysis documentation.
    """
    cape_term = np.clip(mucape / 1500.0, 0.0, None)          # kein Deckel (CAPE kann > 1500 sein)
    lcl_term  = np.clip((2000.0 - ml_lcl) / 1000.0, 0.0, 1.0)  # Deckel bei LCL=1000m → Term=1.0, LCL>2000m → 0
    srh_term  = np.clip(srh / 150.0,       0.0, None)
    bs_term   = np.clip(mu_eff_bs / 20.0,  0.0, 1.5)         # Deckel bei 30 m/s

    stp = cape_term * lcl_term * srh_term * bs_term
    return np.clip(stp, 0.0, None)

def calc_scp(mucape, eff_bs, srh):
    """
    Supercell Composite Parameter – Thompson et al. (2004).
    SCP = (MUCAPE/1000) * (EffShear/20) * (EffSRH/50)

    mucape  : J/kg
    eff_bs  : Effective Bulk Shear 0–3 km [m/s]
    srh     : Storm-Relative Helicity 0–3 km [m²/s²]

    SCP > 1  → Superzellpotenzial vorhanden
    SCP > 4  → deutliches Superzellpotenzial
    SCP > 8  → signifikante Superzelle wahrscheinlich
    """
    cape_term = np.clip(mucape / 1000.0, 0.0, None)
    shear_term = np.clip(eff_bs / 20.0,  0.0, None)
    srh_term  = np.clip(srh    / 50.0,   0.0, None)   # nur positive SRH

    scp = cape_term * shear_term * srh_term
    return np.clip(scp, 0.0, None)


# -------------------------------------------------------
# Hauptverarbeitung
# -------------------------------------------------------
def process_step_pair(prev_step, step):
    print(f"\n=== Verarbeite Step {prev_step}h → {step}h ===")

    missing = check_files(prev_step, step)
    if missing:
        print(f"  ⚠️  {len(missing)} Dateien fehlen, überspringe:")
        for p in missing[:5]:
            print(f"     {p}")
        if len(missing) > 5:
            print(f"     ... und {len(missing)-5} weitere")
        return

    t2m,       lats, lons = read_sfc("2t",     prev_step)
    td2m,      _,    _    = read_sfc("2d",     prev_step)
    sp,        _,    _    = read_sfc("sp",     prev_step)
    tp0,       _,    _    = read_sfc("tp",     prev_step)
    tp1,       _,    _    = read_sfc("tp",     step)
    lsm,       _,    _    = read_sfc("lsm",    prev_step)
    mucape,    _,    _    = read_sfc("mucape", prev_step)
    z_sfc_geo, _,    _    = read_sfc("z",      0)   # → immer z_step000.grib2
    z_sfc = z_sfc_geo / g

    t_pl,  levels, _, _ = read_pl("t",  prev_step)
    q_pl,  _,      _, _ = read_pl("q",  prev_step)
    r_pl,  _,      _, _ = read_pl("r",  prev_step)
    r_pl = np.clip(r_pl, 0.0, 100.0)
    u_pl,  _,      _, _ = read_pl("u",  prev_step)
    v_pl,  _,      _, _ = read_pl("v",  prev_step)
    z_pl,  _,      _, _ = read_pl("gh", prev_step)

    rh_mean     = calc_mean_rh(r_pl, levels)
    
    mcpr_raw = np.maximum((tp1 - tp0), 0.0)

    idx_500     = np.argmin(np.abs(levels - 500.0))
    theta_e_sfc = theta_ep_bolton(t2m, sp / 100.0,
                    np.clip(0.622 * es_buck(td2m) / (sp / 100.0 - es_buck(td2m)), 0.0, 0.06))
    theta_e_500 = theta_ep_bolton(t_pl[idx_500], 500.0, q_pl[idx_500])
    dtheta_e    = theta_e_sfc - theta_e_500

    conv_weight = (
        np.clip(dtheta_e / 20.0,          0.0, 1.0) * 0.4 +
        np.clip(mucape   / 500.0,         0.0, 1.0) * 0.4 +
        np.clip((rh_mean - 40.0) / 40.0,  0.0, 1.0) * 0.2
    )
    mcpr = np.clip(mcpr_raw * conv_weight, 0.0, 0.00296)

    mu_mixr     = calc_mixr_2m(td2m, sp)
    ml_lcl      = calc_lcl_height(t2m, td2m)
    deg0l       = calc_deg0l(t_pl, z_pl, levels)
    mu_li       = calc_mu_li(t_pl, z_pl, q_pl, levels, t2m, td2m, sp)
    mu_eff_bs   = calc_eff_bulk_shear(z_pl, u_pl, v_pl, z_sfc, mucape)
    mw_13       = calc_mean_wind_1_3km(z_pl, u_pl, v_pl, z_sfc)
    sb_wmax     = np.sqrt(np.maximum(2.0 * mucape, 0.0))
    mu_cape_m10 = np.maximum(mucape * 0.30, 0.0)
    cin         = calc_cin(t2m, td2m, sp, t_pl, q_pl, z_pl, levels)
    print(f"  CIN check → min: {np.nanmin(cin):.2f}  mean: {np.nanmean(cin):.2f}  max: {np.nanmax(cin):.2f}  J/kg")
    srh         = calc_srh(u_pl, v_pl, z_pl, z_sfc, levels, layer_m=3000)
    stp         = calc_stp(mucape, ml_lcl, srh, mu_eff_bs)
    scp         = calc_scp(mucape, mu_eff_bs, srh)

    predictors = {
        "MU_LI":       mu_li,
        "MU_CAPE":     mucape,
        "MU_CAPE_M10": mu_cape_m10,
        "SB_WMAX":     sb_wmax,
        "CIN":         cin,
        "RHmean":      rh_mean,
        "MU_MIXR":     mu_mixr,
        "ML_MIXR":     mu_mixr,
        "ML_LCL":      ml_lcl,
        "ZeroHeight":  deg0l,
        "mcpr":        mcpr,
        "MU_EFF_BS":   mu_eff_bs,
        "MW_13":       mw_13,
        "lsm":         lsm,
        "z_sfc":       z_sfc,
        "STP":         stp,   # NEU: Significant Tornado Parameter  [-]
        "SCP":         scp,
    }

    save_predictors(predictors, lats, lons, prev_step, step)
    clear_cache()

# -------------------------------------------------------
# Speichern
# -------------------------------------------------------
def save_predictors(predictors, lats, lons, prev_step, step):
    outfile = os.path.join(
        OUTPUT_DIR,
        f"predictors_{date}_{run:02d}Z_step{prev_step:03d}-{step:03d}.nc"
    )
    run_time = datetime.strptime(f"{date}{run:02d}", "%Y%m%d%H")
    times = np.array([
        run_time + timedelta(hours=prev_step),
        run_time + timedelta(hours=step),
    ], dtype="datetime64[ns]")

    ds = xr.Dataset(
        {name: (["time", "latitude", "longitude"],
                np.stack([arr, arr], axis=0).astype(np.float32))
         for name, arr in predictors.items()},
        coords={
            "time":      times,
            "latitude":  ("latitude",  lats),
            "longitude": ("longitude", lons),
        },
        attrs={
            "date":           date,
            "run":            f"{run:02d}Z",
            "step_start":     prev_step,
            "step_end":       step,
            "interval_hours": step - prev_step,
            "source":         "ECMWF IFS Open Data",
            "created":        datetime.utcnow().isoformat(),
            "notes":          "AR-CHaMo Prädiktoren v7 (per-step GRIB, steps 6-144)",
        },
    )
    ds.to_netcdf(outfile)
    print(f"  💾 Gespeichert: {outfile}")

# -------------------------------------------------------
# Hauptprogramm
# -------------------------------------------------------
def main():
    print("=== AR-CHaMo Prädiktor-Berechnung (v7 – per-step GRIB) ===")
    print(f"Datum: {date}  Lauf: {run:02d} UTC")
    print(f"Eingabe: {BASE_DIR}/")
    print(f"Ausgabe: {OUTPUT_DIR}/")
    print(f"Steps: {steps_all[0]}–{steps_all[-1]}h alle 3h")

    for i in range(1, len(steps_all)):
        process_step_pair(steps_all[i - 1], steps_all[i])

    print("\n✅ Alle Steps verarbeitet!")

if __name__ == "__main__":
    main()