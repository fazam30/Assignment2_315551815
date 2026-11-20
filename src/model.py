"""Model: one gradient-boosted tree classifier per activity, on the features of features.py.

Training examples: every day at 00:00 Taiwan time (16:00 UTC) is a prediction time t; for each
cohort user, the features at t and the answers of the week [t, t+7d) make one example. The first
14 days of data are only used as history.

The 0/1 threshold of each activity is chosen to maximize its F1 on the last week that has
answers, with a model that has not seen that week; then the model is trained again on all
weeks that have answers and used to predict.

Validation (week 2026/11/08 - 11/15 Taiwan time), with the ablation study:

    python src/model.py validate --data data

Final prediction (test week 2026/11/22 - 11/29 Taiwan time):

    python src/model.py predict --data data --student-id 314551051
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "assignment2" / "scripts"))
from activities import ACTIVITIES, activity_matrix, load_db, taiwan_to_utc  # noqa: E402
from features import FeatureBuilder, feature_groups  # noqa: E402
from evaluate import scores  # noqa: E402  (the course's scoring function)

WEEK = pd.Timedelta(days=7)
HISTORY = pd.Timedelta(days=14)
TEST_INPUT = ROOT / "assignment2" / "submission" / "test_input.csv"
CONFIGS = {
    "A. own (last 7 days)": ["own"],
    "B. own + temporal": ["own", "temporal"],
    "C. own + relational": ["own", "relational"],
    "D. own + temporal + relational": ["own", "temporal", "relational"],
}
FINAL = "D. own + temporal + relational"


class Data:
    def __init__(self, data_dir):
        self.db = load_db(data_dir)
        self.fb = FeatureBuilder(self.db)
        self.events = self.fb.ev
        self.start = self.events["row_time"].min()
        self.end = self.events["time"].max()
        self.test_ids = pd.read_csv(TEST_INPUT)["UserID"].to_numpy()
        self.train_ids = self.fb.cohort
        self._x = {}

    def times(self, cut):
        """Daily prediction times (16:00 UTC) whose week of answers ends before cut."""
        first = (self.start + HISTORY).normalize() + pd.Timedelta(hours=16)
        if first < self.start + HISTORY:
            first += pd.Timedelta(days=1)
        return [t for t in pd.date_range(first, cut, freq="D") if t + WEEK <= cut]

    def X(self, t, ids):
        key = (t, ids is self.test_ids)
        if key not in self._x:
            self._x[key] = self.fb.features(t, ids)
        return self._x[key]

    def Y(self, t, ids):
        y = activity_matrix(self.events, self.db["Users"], ids, t, t + WEEK)
        return y[ACTIVITIES].to_numpy()

    def dataset(self, ts, ids):
        return (pd.concat([self.X(t, ids) for t in ts], ignore_index=True),
                np.vstack([self.Y(t, ids) for t in ts]))


def fit(X, Y):
    models = []
    for i in range(len(ACTIVITIES)):
        if Y[:, i].sum() == 0:  # never happens in training: always predict 0
            models.append(None)
            continue
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
                                           min_samples_leaf=50, l2_regularization=1.0,
                                           early_stopping=False, random_state=0)
        models.append(m.fit(X, Y[:, i]))
    return models


def proba(models, X):
    return np.column_stack([np.zeros(len(X)) if m is None else m.predict_proba(X)[:, 1] for m in models])


def best_thresholds(P, Y):
    grid = np.arange(0.02, 0.95, 0.01)
    thr = []
    for i in range(Y.shape[1]):
        if Y[:, i].sum() == 0:
            thr.append(0.5)
            continue
        f1 = [2 * ((P[:, i] >= g) & (Y[:, i] == 1)).sum() /
              max((P[:, i] >= g).sum() + Y[:, i].sum(), 1) for g in grid]
        thr.append(float(grid[int(np.argmax(f1))]))
    return np.array(thr)


def train_and_predict(data, cut, eval_t, cols):
    """Choose the thresholds on the last week with answers, retrain on all weeks with answers
    (answers known before cut), and predict at eval_t for the test users."""
    ts = data.times(cut)
    tune_t = ts[-1]
    ts_a = [t for t in ts if t + WEEK <= tune_t]
    if not ts_a:
        sys.exit("Not enough data: need at least 14 days of history + 2 weeks of answers.")
    Xa, Ya = data.dataset(ts_a, data.train_ids)
    ma = fit(Xa[cols], Ya)
    Xt = data.X(tune_t, data.train_ids)
    thr = best_thresholds(proba(ma, Xt[cols]), data.Y(tune_t, data.train_ids))

    Xb, Yb = data.dataset(ts, data.train_ids)
    mb = fit(Xb[cols], Yb)
    P = proba(mb, data.X(eval_t, data.test_ids)[cols])
    pred = pd.DataFrame((P >= thr).astype(int), columns=ACTIVITIES)
    pred.insert(0, "UserID", data.test_ids)
    info = {"train_days": f"{ts[0]:%m-%d} to {ts[-1]:%m-%d}", "examples": len(Xb), "tune_day": f"{tune_t:%m-%d}"}
    return pred, thr, info


def to_frame(arr, ids):
    df = pd.DataFrame(arr, columns=ACTIVITIES)
    df.insert(0, "UserID", ids)
    return df


def validate(args):
    data = Data(args.data)
    t = taiwan_to_utc(args.t)
    print(f"Validation week: {args.t} Taiwan time + 7 days (t = {t} UTC). Data: {data.start} -> {data.end} UTC")
    if data.end < t + WEEK - pd.Timedelta(hours=6):
        print("WARNING: the data ends before the end of the validation week.")
    truth = to_frame(data.Y(t, data.test_ids), data.test_ids)

    rows = []

    def row(name, pred):
        s = scores(pred, truth)
        s5 = scores(pred, truth, [a for a in ACTIVITIES if a != "push"])
        return {"model": name, **s["per_event"], "macro_f1": s["macro_f1"], "micro_f1": s["micro_f1"],
                "score": s["score"], "score_without_push": s5["score"]}

    copy = activity_matrix(data.events, data.db["Users"], data.test_ids, t - WEEK, t, known_before=t)
    rows.append(row("Copy last week", copy))
    groups = feature_groups(list(data.X(t, data.test_ids).columns))
    for name, gs in CONFIGS.items():
        cols = [c for g in gs for c in groups[g]]
        t0 = time.time()
        pred, thr, info = train_and_predict(data, t, t, cols)
        rows.append(row(name, pred))
        print(f"  {name}: {len(cols)} features, score {100 * rows[-1]['score']:.2f}% "
              f"({info['examples']} training examples, days {info['train_days']}, {time.time() - t0:.0f}s)")
        if name == FINAL:
            pred.to_csv("val_pred.csv", index=False)
            truth.to_csv("val_truth.csv", index=False)
            print("  thresholds: " + ", ".join(f"{a} {v:.2f}" for a, v in zip(ACTIVITIES, thr)))

    res = pd.DataFrame(rows).set_index("model")
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    res.to_csv(out / "validation_scores.csv")
    print("\nF1 per activity and Score (%), validation week:")
    print((100 * res).round(2).to_string())
    print(f"\nWrote {out / 'validation_scores.csv'}, val_pred.csv and val_truth.csv (model {FINAL}).")
    print("Note: an activity that nobody does that week (e.g. push when GH Archive misses the pushes)\n"
          "gets F1 = 100% from the course's scoring; score_without_push leaves push out (like --skip push).")
    print("Share of test users doing each activity that week:")
    print(truth[ACTIVITIES].mean().round(3).to_string())


def predict(args):
    data = Data(args.data)
    t = taiwan_to_utc(args.t)
    print(f"Test week: {args.t} Taiwan time + 7 days (t = {t} UTC). Data: {data.start} -> {data.end} UTC")
    if data.end < t - pd.Timedelta(hours=12) and not args.features_at_data_end:
        print("WARNING: the data stops more than 12 hours before t: download the latest hours first,\n"
              "         or use --features-at-data-end.")
    t_feat = t
    if args.features_at_data_end and data.end < t:
        # the data stops before t: describe the users at the last hour with data, so that the
        # hours with no data are not read as "nobody did anything" (an unseen situation in training)
        t_feat = data.end.ceil("h")
        print(f"Features computed at the end of the data, {t_feat} UTC ({(t - t_feat) / pd.Timedelta(hours=1):.0f} h before t).")
    groups = feature_groups(list(data.X(t_feat, data.test_ids).columns))
    cols = [c for g in CONFIGS[FINAL] for c in groups[g]]
    pred, thr, info = train_and_predict(data, data.end, t_feat, cols)
    out = ROOT / f"output_{args.student_id}.csv"
    pred.to_csv(out, index=False)
    print(f"Model {FINAL}: {len(cols)} features, {info['examples']} training examples "
          f"(days {info['train_days']}), thresholds chosen on the week of {info['tune_day']}")
    print("  thresholds: " + ", ".join(f"{a} {v:.2f}" for a, v in zip(ACTIVITIES, thr)))
    print(f"Wrote {out}. Share of users predicted 1:")
    print(pred[ACTIVITIES].mean().round(3).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="score the models on a past week (ablation study)")
    v.add_argument("--data", required=True)
    v.add_argument("--t", default="2026-11-08 00:00", help="start of the validation week, Taiwan time")
    p = sub.add_parser("predict", help="write output_{studentID}.csv for the test week")
    p.add_argument("--data", required=True)
    p.add_argument("--student-id", required=True)
    p.add_argument("--t", default="2026-11-22 00:00", help="start of the test week, Taiwan time")
    p.add_argument("--features-at-data-end", action="store_true",
                   help="if the data stops before t, compute the features at the last hour with data")
    args = ap.parse_args()
    validate(args) if args.cmd == "validate" else predict(args)


if __name__ == "__main__":
    main()
