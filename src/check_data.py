"""Check the downloaded hour files and the 8 tables: missing or empty hours, users covered,
and the number of activities per day (to see holes in the data).

    python src/check_data.py --raw raw --data data

Writes figures/daily_activity.csv (and figures/daily_activity.png if matplotlib is installed).
"""
import argparse
import gzip
import re
import sys
import zlib
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from activities import ACTIVITIES, activity_events, load_db  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HOUR_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-(\d{1,2})\.json\.gz$")


def count_lines(path):
    n = 0
    with gzip.open(path, "rb") as f:
        try:
            for _ in f:
                n += 1
        except (EOFError, gzip.BadGzipFile, zlib.error, OSError):
            return n, "damaged"
    return n, "ok"


def check_raw(raw_dir):
    rows = []
    for p in Path(raw_dir).glob("*.json.gz"):
        m = HOUR_RE.match(p.name)
        if m:
            y, mo, d, h = map(int, m.groups())
            n, st = count_lines(p)
            rows.append((pd.Timestamp(y, mo, d, h), p.name, n, st))
    df = pd.DataFrame(rows, columns=["hour", "file", "lines", "status"]).sort_values("hour")
    full = pd.date_range(df["hour"].min(), df["hour"].max(), freq="h")
    missing = full.difference(df["hour"])
    print(f"RAW FILES: {len(df)} files, {df['hour'].min()} -> {df['hour'].max()} UTC")
    print(f"  hours missing from the folder: {len(missing)}")
    if len(missing):
        print("   ", ", ".join(f"{h:%m-%d %H}h" for h in missing[:40]), "..." if len(missing) > 40 else "")
    empty = df[df["lines"] == 0]
    print(f"  files with 0 lines (no cohort event, or a failed download): {len(empty)}")
    if len(empty):
        print("   ", ", ".join(empty["file"].head(40)), "..." if len(empty) > 40 else "")
    if (df["status"] != "ok").any():
        print(f"  damaged files: {(df['status'] != 'ok').sum()}")
    print(f"  lines per hour: median {df['lines'].median():.0f}, min {df['lines'].min()}, max {df['lines'].max()}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--data", default="data")
    args = ap.parse_args()

    check_raw(args.raw)

    db = load_db(args.data)
    users = db["Users"]
    test = pd.read_csv(ROOT / "assignment2" / "submission" / "test_input.csv")["UserID"]
    cohort = users[users["is_cohort"].fillna(False).astype(bool)]
    print(f"\nUSERS: {len(users)} rows, {len(cohort)} cohort users seen (of 9,000), "
          f"{test.isin(users['github_user_id']).sum()} of {len(test)} test users seen")

    ev = activity_events(db)
    ev = ev[ev["time"] >= ev["row_time"].min()]  # drop activities dated before the data starts
    ev["day"] = ev["time"].dt.floor("D")
    daily = ev.pivot_table(index="day", columns="activity", values="user_id", aggfunc="size").fillna(0)
    daily = daily.reindex(columns=ACTIVITIES, fill_value=0).astype(int)
    daily["active_users"] = ev.groupby("day")["user_id"].nunique()

    # weeks starting on Sunday, like the test week (Sunday 11/22 Taiwan time)
    ev["week"] = ev["day"] - pd.to_timedelta((ev["day"].dt.dayofweek + 1) % 7, unit="D")
    weekly = ev.pivot_table(index="week", columns="activity", values="user_id", aggfunc="size").fillna(0)
    weekly = weekly.reindex(columns=ACTIVITIES, fill_value=0).astype(int)
    weekly["active_users"] = ev.groupby("week")["user_id"].nunique()
    print("\nACTIVITIES PER WEEK (week starting on the Sunday shown, UTC):")
    print(weekly.to_string())

    med = daily[ACTIVITIES].median()
    low = daily[(daily[ACTIVITIES] < 0.3 * med).any(axis=1)]
    print(f"\nDAYS WITH AN ACTIVITY BELOW 30% OF ITS MEDIAN ({len(low)} days, possible holes):")
    if len(low):
        print(low.to_string())

    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    daily.to_csv(out / "daily_activity.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ax = daily[ACTIVITIES].plot(figsize=(11, 4.5), linewidth=1.4)
        ax.set_title("Activities of the cohort users per day (UTC)")
        ax.set_ylabel("number of activities")
        ax.set_xlabel("")
        plt.tight_layout()
        plt.savefig(out / "daily_activity.png", dpi=150)
        print(f"\nWrote {out / 'daily_activity.csv'} and {out / 'daily_activity.png'}")
    except ImportError:
        print(f"\nWrote {out / 'daily_activity.csv'} (pip install matplotlib for the figure)")


if __name__ == "__main__":
    main()
