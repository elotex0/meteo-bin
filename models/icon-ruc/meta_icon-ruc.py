"""
Baut metadata.json aus den erzeugten Rasterdateien (uint8, ohne Endung).
Dateiname: {var_type}_{YYYYMMDD}_{HHMM}   (var_type darf Unterstriche enthalten)

Aufruf: python generate_metadata.py <ordner> <run> [<datum>]
Nur Standardbibliothek.
"""
import json
import os
import re
import struct
import sys
from datetime import datetime, timezone

from encoding_config import ENCODING

FILENAME_RE = re.compile(r"^(?P<var_type>.+)_(?P<date>\d{8})_(?P<time>\d{4})$")

FORMAT_VERSION = 1
MAX_HEADER_BYTES = 65536   # Schutz: erkennt alte Rohdateien ohne Header


def read_header(path):
    """Liest den JSON-Header. Gibt (header, payload_bytes) zurück."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        raw = f.read(4)
        if len(raw) < 4:
            sys.exit(f"{path}: Datei zu kurz")
        (n,) = struct.unpack(">I", raw)
        if n == 0 or n > MAX_HEADER_BYTES or 4 + n > size:
            sys.exit(f"{path}: kein gültiger Header (Länge {n}) - "
                     f"alte Rohdatei ohne Header?")
        try:
            header = json.loads(f.read(n).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            sys.exit(f"{path}: Header ist kein gültiges JSON ({e})")
    if header.get("format") != FORMAT_VERSION:
        sys.exit(f"{path}: Format {header.get('format')} nicht unterstützt "
                 f"(erwartet {FORMAT_VERSION})")
    return header, size - 4 - n


def scan_output_dir(output_dir):
    """Sammelt die Timesteps pro var_type und prüft die Dateigröße (1 Byte pro Pixel)."""
    var_types = {}
    grid = None
    grid_source = None
    for root, _dirs, files in os.walk(output_dir):
        for fname in files:
            m = FILENAME_RE.match(fname)
            if not m:
                continue
            path = os.path.join(root, fname)
            header, payload = read_header(path)

            g = header.get("grid")
            if not g:
                sys.exit(f"{fname}: Header enthält kein 'grid'")
            expected = g["width"] * g["height"]
            if payload != expected:
                sys.exit(f"{fname}: Payload {payload} Bytes, erwartet {expected} "
                         f"width/height in GRID prüfen!")
                
            if grid is None:
                grid, grid_source = g, fname
            elif g != grid:
                sys.exit(f"{fname}: Grid {g} stimmt nicht mit {grid_source} überein ({grid})")

            ts = f"{m.group('date')}_{m.group('time')}"
            var_types.setdefault(m.group("var_type"), set()).add(ts)
    return var_types, grid 


def build_meta(output_dir, run, date):
    found, grid = scan_output_dir(output_dir)
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