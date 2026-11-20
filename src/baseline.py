"""Baseline "Copy last week": predict 1 for an activity if the user did it in the 7 days
before the prediction time t, otherwise 0.

Validation week (the answers are in the data, so we can score it):

    python src/baseline.py --data data --t "2026-11-08 00:00" --pred val_pred.csv --truth val_truth.csv
    python assignment2/scripts/evaluate.py --pred val_pred.csv --truth val_truth.csv

Test week (t = 2026/11/22 00:00 Taiwan time):

    python src/baseline.py --data data --t "2026-11-22 00:00" --pred output_{studentID}.csv

--t is in Taiwan time. Only rows that appear in the data before t are used for the prediction.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from activities import ACTIVITIES, activity_events, activity_matrix, data_end, load_db, taiwan_to_utc  # noqa: E402

WEEK = pd.Timedelta(days=7)
TEST_INPUT = Path(__file__).resolve().parent.parent / "assignment2" / "submission" / "test_input.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="output folder of preprocess.py")
    ap.add_argument("--t", required=True, help='prediction time, Taiwan time, e.g. "2026-11-08 00:00"')
    ap.add_argument("--pred", required=True, help="prediction CSV to write")
    ap.add_argument("--truth", help="also write the answers of the week [t, t+7d) to this CSV")
    args = ap.parse_args()

    t = taiwan_to_utc(args.t)
    db = load_db(args.data)
    events = activity_events(db)
    users = pd.read_csv(TEST_INPUT)["UserID"]
    print(f"t = {args.t} Taiwan = {t} UTC; data ends at {data_end(events)} UTC; {len(users)} test users")

    pred = activity_matrix(events, db["Users"], users, t - WEEK, t, known_before=t)
    pred.to_csv(args.pred, index=False)
    print(f"Wrote {args.pred}. Share of users predicted 1:")
    print(pred[ACTIVITIES].mean().round(3).to_string())

    if args.truth:
        if data_end(events) < t + WEEK - pd.Timedelta(hours=2):
            print(f"WARNING: the data ends before the end of the week ({t + WEEK} UTC): "
                  "the answers are incomplete.")
        truth = activity_matrix(events, db["Users"], users, t, t + WEEK)
        truth.to_csv(args.truth, index=False)
        print(f"Wrote {args.truth}. Share of users who did each activity that week:")
        print(truth[ACTIVITIES].mean().round(3).to_string())


if __name__ == "__main__":
    main()
