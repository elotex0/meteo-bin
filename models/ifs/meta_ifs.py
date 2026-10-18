"""
Baut metadata.json aus den erzeugten Rasterdateien (uint8, ohne Endung).
Dateiname: {var_type}_{YYYYMMDD}_{HHMM}   (var_type darf Unterstriche enthalten)

Aufruf: python generate_metadata.py <ordner> <run> [<datum>]
Nur Standardbibliothek.
"""
import json
import os
import re
import sys
from datetime import datetime, timezone

from encoding_config import ENCODING, LEVELS, NODATA_U8

FILENAME_RE = re.compile(r"^(?P<var_type>.+)_(?P<date>\d{8})_(?P<time>\d{4})$")

# Festes ICON-D2-Raster (regular-lat-lon, EPSG:4326, row0 = Norden).
# Muss zu "extent" in generate_pngs.py passen.
GRID = {
    "extent": [-3.94, 43.18, 20.34, 58.08],   # lon_min, lat_min, lon_max, lat_max
    "width": 1215,
    "height": 746,
    "nodata": NODATA_U8,
    "levels": LEVELS,
}


def scan_output_dir(output_dir):
    """Sammelt die Timesteps pro var_type und prüft die Dateigröße (1 Byte pro Pixel)."""
    var_types = {}
    expected = GRID["width"] * GRID["height"]
    for root, _dirs, files in os.walk(output_dir):
        for fname in files:
            m = FILENAME_RE.match(fname)
            if not m:
                continue
            size = os.path.getsize(os.path.join(root, fname))
            if size != expected:
                sys.exit(f"{fname}: {size} Bytes, erwartet {expected}. "
                         f"width/height in GRID prüfen!")
            ts = f"{m.group('date')}_{m.group('time')}"
            var_types.setdefault(m.group("var_type"), set()).add(ts)
    return var_types


def build_meta(output_dir, run, date):
    found = scan_output_dir(output_dir)
    if not found:
        print(f"Keine passenden Dateien in {output_dir} gefunden.", file=sys.stderr)

    var_types = {}
    for var_type in sorted(found):
        enc = ENCODING[var_type]
        steps = sorted(found[var_type])
        var_types[var_type] = {
            "transform": enc["transform"],
            "vmin": enc["vmin"],
            "vmax": enc["vmax"],
            "num_steps": len(steps),
            "timesteps": steps,
        }

    return {
        "run": run,
        "date": date,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "grid": GRID,
        "var_types": var_types,
    }


def main():
    out_dir = sys.argv[1]
    run = sys.argv[2]
    date = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else datetime.now(timezone.utc).strftime("%Y%m%d")

    meta = build_meta(out_dir, run, date)

    # metadata.json liegt außerhalb des run-Ordners
    meta_path = os.path.join(os.path.dirname(out_dir.rstrip(os.sep)), "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, separators=(",", ":"))

    total = sum(v["num_steps"] for v in meta["var_types"].values())
    print(f"metadata.json geschrieben: {meta_path} ({len(meta['var_types'])} var_types, {total} Dateien)")


if __name__ == "__main__":
    main()