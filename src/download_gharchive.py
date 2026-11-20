"""Download GH Archive hour files for Assignment 2.

Usage (times are UTC, format YYYY-MM-DD-H):

    python src/download_gharchive.py --start 2026-10-01-0 --end 2026-11-18-6 --out raw

Re-running the same command only downloads the hours that are still missing, so on the last
day you can run it again with a later --end to add the newest hours.

Two modes:

  * compact (default): while downloading, keep only the lines whose actor is one of the 9,000
    cohort users (assignment2/scripts/cohort.txt). This is exactly the first filter that
    preprocess.py applies (Step 2, extract_hour), using the same keyed hash, so preprocess.py
    builds the same 8 tables from these files as from the full files, and each hour takes a
    few hundred KB on disk instead of the full GH Archive file.
  * --full: save the original GH Archive files unchanged (needs much more disk space).

The files keep the GH Archive names (YYYY-MM-DD-H.json.gz), so preprocess.py reads them as is:

    python assignment2/scripts/preprocess.py --raw raw --out data
"""
import argparse
import gzip
import io
import sys
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "assignment2" / "scripts"
URL = "https://data.gharchive.org/{name}"
RETRIES = 4
USER_AGENT = "Mozilla/5.0 (compatible; gharchive-downloader for a course assignment)"

_COHORT = None


def hour_name(h):
    """GH Archive file name of an hour, e.g. 2026-10-01-0.json.gz (hour without leading 0)."""
    return f"{h:%Y-%m-%d}-{h.hour}.json.gz"


def parse_hour(text):
    y, m, d, h = map(int, text.split("-"))
    return datetime(y, m, d, h, tzinfo=timezone.utc)


def load_course_helpers():
    """The cohort, the hash and the actor pattern, taken from the course's preprocess.py
    so the filter is exactly the same."""
    sys.path.insert(0, str(SCRIPTS))
    import preprocess  # noqa: E402  (course script)
    cohort = preprocess.load_cohort(SCRIPTS / "cohort.txt")
    return cohort, preprocess.code_int, preprocess.ACTOR_ID_RE


def _init_worker():
    global _COHORT
    _COHORT = load_course_helpers()


def keep_cohort_lines(lines, cohort, code_int, actor_re):
    """The lines whose actor is a cohort user (same test as preprocess.extract_hour)."""
    checked = {}
    for line in lines:
        m = actor_re.search(line)
        if not m:
            continue
        actor = int(m.group(1))
        if actor not in checked:
            checked[actor] = code_int("user", actor) in cohort
        if checked[actor]:
            yield line


def gzip_lines(raw_bytes):
    """Lines of gzip data; a damaged end keeps the lines before it (like preprocess.py)."""
    with gzip.GzipFile(fileobj=io.BytesIO(raw_bytes)) as f:
        try:
            yield from f
        except (EOFError, gzip.BadGzipFile, zlib.error):
            return


def fetch(url):
    """Bytes of url; None if the hour does not exist (yet) on GH Archive."""
    # Some servers refuse Python's default User-Agent ("Python-urllib") with 403 Forbidden.
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 403:  # refused: retrying will not help
                raise RuntimeError(f"{url}: 403 Forbidden (access refused by the server)")
            err = e
        except Exception as e:  # network errors, timeouts
            err = e
        time.sleep(2 ** attempt)
    raise RuntimeError(f"{url}: {err}")


def download_hour(name, out_dir, full):
    """Download one hour into out_dir/name. Returns (name, status)."""
    target = out_dir / name
    if target.exists():
        return name, "exists"
    data = fetch(URL.format(name=name))
    if data is None:
        return name, "missing on GH Archive"
    part = target.with_suffix(".part")
    if full:
        part.write_bytes(data)
    else:
        cohort, code_int, actor_re = _COHORT
        kept = keep_cohort_lines(gzip_lines(data), cohort, code_int, actor_re)
        with gzip.open(part, "wb") as f:
            for line in kept:
                f.write(line if line.endswith(b"\n") else line + b"\n")
    part.replace(target)  # rename only when complete, so a stopped run never leaves half files
    return name, "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True, help="first hour, UTC, YYYY-MM-DD-H")
    ap.add_argument("--end", help="last hour, UTC, YYYY-MM-DD-H (default: 2 hours ago)")
    ap.add_argument("--out", default="raw", help="folder for the hour files (default: raw)")
    ap.add_argument("--full", action="store_true", help="keep the full GH Archive files")
    ap.add_argument("--workers", type=int, default=4, help="hours downloaded at the same time")
    args = ap.parse_args()

    start = parse_hour(args.start)
    end = parse_hour(args.end) if args.end else \
        datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=2)
    hours = []
    h = start
    while h <= end:
        hours.append(h)
        h += timedelta(hours=1)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(hours)} hours from {hour_name(start)} to {hour_name(end)} (UTC) -> {out_dir}"
          f" [{'full' if args.full else 'compact'} mode]")

    if not args.full:
        load_course_helpers()  # fail early if the course files are not found
    status = {}
    with ProcessPoolExecutor(args.workers, initializer=_init_worker) as pool:
        jobs = [pool.submit(download_hour, hour_name(h), out_dir, args.full) for h in hours]
        for i, job in enumerate(as_completed(jobs), 1):
            try:
                name, st = job.result()
            except Exception as e:
                name, st = str(e), "error"
            status[name] = st
            if st not in ("ok", "exists"):
                print(f"  {name}: {st}")
            if i % 24 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} hours done")

    counts = {}
    for st in status.values():
        counts[st] = counts.get(st, 0) + 1
    print("Summary:", counts)
    if counts.get("error"):
        print("Some hours failed: run the same command again to retry them.")


if __name__ == "__main__":
    main()
