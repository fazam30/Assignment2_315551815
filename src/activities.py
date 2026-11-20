"""Shared helpers: load the 8 tables and find when each user did each of the 6 activities.

Times in the tables are UTC (no time zone). Prediction times are given in Taiwan time
(UTC+8) on the command line and converted to UTC here.
"""
from pathlib import Path

import pandas as pd

ACTIVITIES = ["open_pr", "open_issue", "issue_comment", "pr_review", "pr_review_comment", "push"]
TABLES = ["Users", "Repositories", "PullRequests", "Issues", "IssueComments",
          "PRReviews", "PRReviewComments", "Pushes"]
TAIWAN_OFFSET = pd.Timedelta(hours=8)


def taiwan_to_utc(text):
    """'2026-11-22 00:00' (Taiwan time) -> UTC Timestamp without time zone."""
    return pd.Timestamp(text) - TAIWAN_OFFSET


def load_db(data_dir):
    """The 8 tables written by preprocess.py (data_dir/db/*.parquet)."""
    db_dir = Path(data_dir) / "db"
    return {name: pd.read_parquet(db_dir / f"{name}.parquet") for name in TABLES}


def activity_events(db):
    """One row per activity done by a user:
        user_id     Users.user_id of the user who did it
        activity    one of ACTIVITIES
        time        when the user did it
        row_time    when the row appears in the data (use only rows with row_time < t)
        repo_id     repository where it happened
    """
    pr, iss = db["PullRequests"], db["Issues"]
    ic, rv, rc, pu = db["IssueComments"], db["PRReviews"], db["PRReviewComments"], db["Pushes"]
    is_pr_conv = iss["is_pull_request"].fillna(False).astype(bool)
    oi = iss[~is_pr_conv]
    parts = [
        # opening a PR: the PR's author, at the PR's opening time
        ("open_pr", pr["author_id"], pr["created_at"], pr["first_seen_at"], pr["repo_id"]),
        # opening an issue (PR conversations are excluded)
        ("open_issue", oi["author_id"], oi["created_at"], oi["first_seen_at"], oi["repo_id"]),
        ("issue_comment", ic["author_id"], ic["created_at"], ic["created_at"], ic["repo_id"]),
        # a review: its submission time (or when it was first seen if that is unknown)
        ("pr_review", rv["reviewer_id"], rv["submitted_at"].fillna(rv["first_seen_at"]),
         rv["first_seen_at"], rv["repo_id"]),
        ("pr_review_comment", rc["author_id"], rc["created_at"], rc["created_at"], rc["repo_id"]),
        ("push", pu["user_id"], pu["created_at"], pu["created_at"], pu["repo_id"]),
    ]
    frames = [pd.DataFrame({"user_id": u.to_numpy(), "activity": a,
                            "time": pd.to_datetime(t).to_numpy(),
                            "row_time": pd.to_datetime(r).to_numpy(),
                            "repo_id": pd.array(rp, dtype="Int64")})
              for a, u, t, r, rp in parts]
    ev = pd.concat(frames, ignore_index=True).dropna(subset=["user_id", "time"])
    ev["user_id"] = ev["user_id"].astype("int64")
    return ev


def activity_matrix(events, users, github_ids, start, end, known_before=None):
    """0/1 table (one row per github_user_id in github_ids, one column per activity):
    1 if the user did the activity in [start, end).
    known_before: if given, only rows whose row_time is before it are used."""
    ev = events[(events["time"] >= start) & (events["time"] < end)]
    if known_before is not None:
        ev = ev[ev["row_time"] < known_before]
    gh = users.set_index("user_id")["github_user_id"]
    ev = ev.assign(UserID=ev["user_id"].map(gh))
    done = (ev.drop_duplicates(["UserID", "activity"])
              .assign(v=1)
              .pivot_table(index="UserID", columns="activity", values="v", aggfunc="max"))
    out = pd.DataFrame({"UserID": list(github_ids)})
    out = out.join(done, on="UserID")
    for a in ACTIVITIES:
        out[a] = out[a].fillna(0).astype(int) if a in out else 0
    return out[["UserID"] + ACTIVITIES]


def data_end(events):
    """Latest activity time in the data (to check that a week is fully covered)."""
    return events["time"].max()
