import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError
import json
import os
import time
import random
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------------------------------------
# Konfiguration via Umgebungsvariablen (GitHub Actions)
# -------------------------------------------------------
DATE = os.getenv("DATE", "20250604")
RUN  = f"{int(os.getenv('RUN', 0)):02d}z"
RUN_HHMM = f"{int(os.getenv('RUN', 0)):02d}0000"
BASE = Path("data/gewitter")
INDEX_CACHE_DIR = Path(".cache/indices")

BUCKET = "ecmwf-forecasts"

PRESSURE_LEVELS  = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
PARAMS_SFC       = ["2t", "2d", "sp", "tp", "lsm", "mucape", "10u", "10v"]
PARAMS_PL        = ["t", "q", "r", "u", "v", "gh"]
STEPS            = list(range(6, 82, 3))

# Threads/Requests
MAX_WORKERS       = 4          # gleichzeitige Downloads
MAX_REQUESTS_PER_SEC = 6       # harte Obergrenze für S3-Requests/Sekunde (glättet Bursts)
DOWNLOAD_RETRIES  = 8
BACKOFF_BASE      = 1.5        # Sekunden
BACKOFF_MAX       = 60.0       # Sekunden

# Range-Merging: Felder, die im File näher als dieser Wert beieinander liegen,
# werden mit EINEM Request geholt statt mit mehreren. 512KB ist ein guter Kompromiss
# zwischen "weniger Requests" und "nicht unnötig viele Daten laden".
MERGE_GAP_BYTES = 512 * 1024

# -------------------------------------------------------
# Einfacher Token-Bucket-Rate-Limiter (threadsicher)
# Begrenzt die *Rate* an S3-Requests, unabhängig von MAX_WORKERS.
# Das ist der Hauptgrund für viele 503/SlowDown: Bursts von parallelen
# Requests gegen dieselbe Datei/denselben Bucket-Prefix.
# -------------------------------------------------------
class RateLimiter:
    def __init__(self, rate_per_sec: float):
        self.rate = rate_per_sec
        self.lock = threading.Lock()
        self.timestamps = []

    def acquire(self):
        with self.lock:
            now = time.monotonic()
            # alte Timestamps außerhalb des 1-Sekunden-Fensters entfernen
            self.timestamps = [t for t in self.timestamps if now - t < 1.0]
            if len(self.timestamps) >= self.rate:
                sleep_time = 1.0 - (now - self.timestamps[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
            self.timestamps.append(time.monotonic())

rate_limiter = RateLimiter(MAX_REQUESTS_PER_SEC)

# -------------------------------------------------------
# Geteilter S3-Client (ein Client für alle Threads)
# WICHTIG: botocore-eigene Retries fast abgeschaltet (max_attempts=1),
# damit nicht zwei verschachtelte Retry-/Backoff-Systeme gleichzeitig laufen.
# Unsere eigene Schleife unten übernimmt das komplett und transparent.
# -------------------------------------------------------
_client_lock   = threading.Lock()
_shared_client = None

def get_client():
    global _shared_client
    with _client_lock:
        if _shared_client is None:
            _shared_client = boto3.client(
                "s3",
                region_name="eu-central-1",
                config=Config(
                    signature_version=UNSIGNED,
                    retries={"max_attempts": 1, "mode": "standard"},
                    max_pool_connections=MAX_WORKERS + 2,
                    connect_timeout=10,
                    read_timeout=30,
                )
            )
        return _shared_client

def _is_throttle(exc: Exception) -> bool:
    """Robuste Erkennung von Throttling-Fehlern über Fehlercode statt String-Suche."""
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in ("SlowDown", "RequestThrottled", "Throttling", "503", "ServiceUnavailable"):
            return True
        if status == 503:
            return True
    if isinstance(exc, EndpointConnectionError):
        return True
    return False

def _backoff_sleep(attempt: int, label: str):
    """AWS-empfohlenes 'full jitter' Backoff."""
    wait = min(BACKOFF_BASE * (2 ** attempt), BACKOFF_MAX)
    wait = random.uniform(0, wait)
    print(f"  ⏳ Throttle ({attempt + 1}) – warte {wait:.1f}s: {label}")
    time.sleep(wait)

def s3_get_with_retry(key: str, range_header: str | None, label: str) -> bytes:
    """Zentrale, einzige Retry-Logik für alle S3-GET-Requests (Index + Daten)."""
    for attempt in range(DOWNLOAD_RETRIES):
        rate_limiter.acquire()
        try:
            s3 = get_client()
            kwargs = {"Bucket": BUCKET, "Key": key}
            if range_header:
                kwargs["Range"] = range_header
            resp = s3.get_object(**kwargs)
            return resp["Body"].read()
        except Exception as e:
            last_attempt = attempt == DOWNLOAD_RETRIES - 1
            if _is_throttle(e) and not last_attempt:
                _backoff_sleep(attempt, label)
                continue
            if not last_attempt:
                # andere transiente Fehler (Verbindungsabbruch etc.) auch retryen,
                # aber mit kürzerer, weniger aggressiver Wartezeit
                time.sleep(random.uniform(1, 3))
                continue
            raise

# -------------------------------------------------------
# Index laden (mit lokalem Cache – Index ändert sich für einen
# bestimmten Lauf nicht mehr, bei Re-Runs spart das komplett den Request)
# -------------------------------------------------------
def get_fields_for_step(step):
    step_str = f"{step}h"
    prefix   = f"{DATE}/{RUN}/ifs/0p25/oper"
    idx_key  = f"{prefix}/{DATE}{RUN_HHMM}-{step_str}-oper-fc.index"

    cache_file = INDEX_CACHE_DIR / f"{DATE}_{RUN}_{step_str}.json"
    if cache_file.exists():
        fields = json.loads(cache_file.read_text())
        return fields, step_str, prefix

    try:
        raw = s3_get_with_retry(idx_key, None, idx_key)
        lines = raw.decode("utf-8").strip().splitlines()
        fields = [json.loads(l) for l in lines]
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(fields))
        return fields, step_str, prefix
    except Exception as e:
        print(f"  ✗ Index nicht gefunden: {idx_key} ({e})")
        return None, step_str, prefix

# -------------------------------------------------------
# Felder pro Datei zu größeren Byte-Ranges zusammenfassen.
# Das ist der wichtigste Hebel gegen 503: aus z.B. 86 Requests pro
# Zeitschritt werden oft nur noch eine Handvoll.
# -------------------------------------------------------
def merge_field_groups(fields_with_targets, gap: int = MERGE_GAP_BYTES):
    """
    fields_with_targets: Liste von (field_dict, out_path)
    Gibt Gruppen zurück: Liste von (group_start, group_end, [(field, out_path), ...])
    """
    items = sorted(fields_with_targets, key=lambda ft: ft[0]["_offset"])
    groups = []
    current_group = []
    current_start = None
    current_end = None

    for field, out_path in items:
        start = field["_offset"]
        end = field["_offset"] + field["_length"] - 1
        if current_group and start - current_end <= gap:
            current_group.append((field, out_path))
            current_end = max(current_end, end)
        else:
            if current_group:
                groups.append((current_start, current_end, current_group))
            current_group = [(field, out_path)]
            current_start = start
            current_end = end
    if current_group:
        groups.append((current_start, current_end, current_group))
    return groups

def download_group(grib_key, group_start, group_end, members):
    """Lädt einen zusammengefassten Byte-Bereich und schneidet die einzelnen Felder lokal heraus."""
    # Falls alle Zieldateien schon existieren, überspringen
    if all(out_path.exists() for _, out_path in members):
        return [(True, str(out_path), field["_length"]) for field, out_path in members]

    label = f"{Path(grib_key).name} [{group_start}-{group_end}] ({len(members)} Felder)"
    try:
        data = s3_get_with_retry(
            grib_key,
            f"bytes={group_start}-{group_end}",
            label,
        )
    except Exception as e:
        return [(False, str(out_path), str(e)) for _, out_path in members]

    results = []
    for field, out_path in members:
        if out_path.exists():
            results.append((True, str(out_path), field["_length"]))
            continue
        local_start = field["_offset"] - group_start
        local_end = local_start + field["_length"]
        chunk = data[local_start:local_end]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(chunk)
        results.append((True, str(out_path), field["_length"]))
    return results

# -------------------------------------------------------
# Tasks aufbauen – normale Steps (6-144)
# -------------------------------------------------------
def build_download_tasks(fields, step, step_str, prefix):
    grib_key = f"{prefix}/{DATE}{RUN_HHMM}-{step_str}-oper-fc.grib2"
    step_tag = f"step{step:03d}"
    tasks    = []

    for param in PARAMS_SFC:
        for field in fields:
            if field["param"] == param and field.get("levtype") == "sfc":
                out = BASE / param / f"{param}_{step_tag}.grib2"
                tasks.append((field, out))

    for param in PARAMS_PL:
        for level in PRESSURE_LEVELS:
            for field in fields:
                if (field["param"] == param
                        and field.get("levtype") == "pl"
                        and field.get("levelist") == str(level)):
                    out = BASE / f"{param}_pl" / f"{param}_pl{level}_{step_tag}.grib2"
                    tasks.append((field, out))

    return grib_key, tasks

# -------------------------------------------------------
# Hauptprogramm
# -------------------------------------------------------
def main():
    print("=== ECMWF Download (verbessert: Range-Merging + Rate-Limit) ===")
    print(f"DATE={DATE}  RUN={RUN}  Steps={STEPS[0]}-{STEPS[-1]}h")
    print(f"Ausgabe: {BASE}/\n")

    # (grib_key, group_start, group_end, members) über alle Steps hinweg
    all_groups = []

    print(f"Indizes laden für Steps {STEPS[0]}-{STEPS[-1]}h ...")
    for step in STEPS:
        fields, step_str, prefix = get_fields_for_step(step)
        if fields is None:
            continue
        grib_key, tasks = build_download_tasks(fields, step, step_str, prefix)
        if not tasks:
            continue
        groups = merge_field_groups(tasks)
        for group_start, group_end, members in groups:
            all_groups.append((grib_key, group_start, group_end, members))
        n_fields = len(tasks)
        print(f"  Step {step:3d}h → {n_fields:3d} Felder in {len(groups):2d} Requests gebündelt "
              f"(statt {n_fields} Einzel-Requests)")

    if not all_groups:
        print("Keine Tasks – Abbruch.")
        return

    total_fields = sum(len(m) for _, _, _, m in all_groups)
    print(f"\nStarte {len(all_groups)} gebündelte Requests für {total_fields} Felder "
          f"mit {MAX_WORKERS} parallelen Threads (max {MAX_REQUESTS_PER_SEC} req/s) ...\n")

    ok = err = total_bytes = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(download_group, grib_key, gs, ge, members): (grib_key, members)
            for grib_key, gs, ge, members in all_groups
        }
        for future in as_completed(futures):
            for success, path, info in future.result():
                if success:
                    ok += 1
                    total_bytes += info
                    short = Path(path).relative_to(BASE)
                    print(f"  ✓ {short}  ({info/1024:.0f} KB)")
                else:
                    err += 1
                    print(f"  ✗ {path}: {info}")

    print(f"\nFertig: {ok} OK, {err} Fehler, {total_bytes/1024/1024:.1f} MB gesamt")
    if err > 0:
        raise SystemExit(f"{err} Download-Fehler – Job schlägt fehl")

if __name__ == "__main__":
    main()