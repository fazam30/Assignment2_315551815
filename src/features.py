"""Features of each user at a prediction time t, computed from the 8 tables.

Only rows that appear in the data before t (row_time < t) are used, and dates such as
closed_at / merged_at are used only if they are before t, so nothing from the future leaks in.

Three groups of features (used for the ablation study in model.py):

  own        what the user did in the last 7 days, per activity (a learned "copy last week")
  temporal   how the user's activity is spread in time: counts over 1/3/14/28 days, days since
             the last time of each activity, time-decayed counts (half-life 3 and 14 days),
             active days, how long the user has been in the data, and the volume of each activity
             over all users in the last 1/7 days (shows holes in the data, e.g. missing pushes)
  relational information that reaches the user through the foreign keys (message passing over
             the schema graph):
               User -> Repositories they worked on -> other users' activity in those repos
               User -> their PullRequests -> PRReviews / PRReviewComments written by others
               User -> their Issues -> IssueComments written by others
               User -> PullRequests they merged (merged_by_id), their open PullRequests
"""
import numpy as np
import pandas as pd

from activities import ACTIVITIES, activity_events

DAY = pd.Timedelta(days=1)
WINDOWS = [1, 3, 7, 14, 28]
HALF_LIVES = [3, 14]
CAP = 60.0  # "days since" value for "never seen before t"


class FeatureBuilder:
    def __init__(self, db):
        users = db["Users"]
        gh = users.set_index("user_id")["github_user_id"]
        self.first_seen = users.set_index("github_user_id")["first_seen_at"]
        self.cohort = users.loc[users["is_cohort"].fillna(False).astype(bool), "github_user_id"].to_numpy()

        ev = activity_events(db)
        ev["gh"] = ev["user_id"].map(gh)
        self.ev = ev

        pr = db["PullRequests"]
        self.pr = pd.DataFrame({
            "author": pr["author_id"].map(gh), "merger": pr["merged_by_id"].map(gh),
            "created_at": pr["created_at"], "first_seen_at": pr["first_seen_at"],
            "closed_at": pr["closed_at"], "merged_at": pr["merged_at"]})
        pr_author = pd.Series(self.pr["author"].to_numpy(), index=pr["pr_id"])
        iss = db["Issues"]
        issue_author = pd.Series(iss["author_id"].map(gh).to_numpy(), index=iss["issue_id"])

        # what other users did on this user's PRs and issues ("received")
        rv, rc, ic = db["PRReviews"], db["PRReviewComments"], db["IssueComments"]
        rec = pd.concat([
            pd.DataFrame({"kind": "review", "owner": rv["pr_id"].map(pr_author),
                          "actor": rv["reviewer_id"].map(gh),
                          "time": rv["submitted_at"].fillna(rv["first_seen_at"]),
                          "row_time": rv["first_seen_at"]}),
            pd.DataFrame({"kind": "review_comment", "owner": rc["pr_id"].map(pr_author),
                          "actor": rc["author_id"].map(gh),
                          "time": rc["created_at"], "row_time": rc["created_at"]}),
            pd.DataFrame({"kind": "issue_comment", "owner": ic["issue_id"].map(issue_author),
                          "actor": ic["author_id"].map(gh),
                          "time": ic["created_at"], "row_time": ic["created_at"]}),
        ], ignore_index=True)
        self.received = rec[rec["owner"].notna() & (rec["owner"] != rec["actor"])]

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _per_user(df, value=None, agg="size"):
        """Series indexed by (gh, activity) -> wide table indexed by gh."""
        g = df.groupby(["gh", "activity"])
        s = g.size() if value is None else g[value].agg(agg)
        return s.unstack("activity")

    def features(self, t, gh_ids):
        F = pd.DataFrame(index=pd.Index(gh_ids, name="gh"))
        ev = self.ev[(self.ev["row_time"] < t) & (self.ev["time"] < t)].copy()
        ev["age"] = (t - ev["time"]) / DAY

        # ---------------- own + temporal: the user's own history
        for w in WINDOWS:
            c = self._per_user(ev[ev["age"] < w])
            for a in ACTIVITIES:
                F[f"own_{a}_{w}d"] = c[a] if a in c else np.nan
        last = self._per_user(ev, "age", "min")
        for a in ACTIVITIES:
            F[f"days_since_{a}"] = last[a] if a in last else np.nan
        for h in HALF_LIVES:
            ev["w"] = np.power(0.5, ev["age"] / h)
            d = self._per_user(ev, "w", "sum")
            for a in ACTIVITIES:
                F[f"decay{h}_{a}"] = d[a] if a in d else np.nan
        recent = ev[ev["age"] < 28]
        F["active_days_28d"] = recent.assign(d=recent["time"].dt.floor("D")).groupby("gh")["d"].nunique()
        fs = self.first_seen.reindex(F.index)
        F["days_in_data"] = ((t - fs) / DAY).where(fs < t)
        for a in ACTIVITIES:
            ea = ev[ev["activity"] == a]
            F[f"global_{a}_1d"] = float((ea["age"] < 1).sum())
            F[f"global_{a}_7d"] = float((ea["age"] < 7).sum())

        # ---------------- relational: user -> repositories -> other users
        r = ev[ev["repo_id"].notna()]
        r28 = r[r["age"] < 28]
        r7 = r[r["age"] < 7]
        user_repo = r28[["gh", "repo_id"]].drop_duplicates()
        F["n_repos_28d"] = user_repo.groupby("gh").size()
        F["n_repos_7d"] = r7[["gh", "repo_id"]].drop_duplicates().groupby("gh").size()
        repo_act = r7.groupby(["repo_id", "activity"]).size().unstack("activity", fill_value=0)
        repo_act = repo_act.reindex(columns=ACTIVITIES, fill_value=0)
        in_my_repos = user_repo.join(repo_act, on="repo_id").groupby("gh")[ACTIVITIES].sum()
        own7_in_repos = self._per_user(r7).reindex(columns=ACTIVITIES).fillna(0)
        others = in_my_repos.sub(own7_in_repos, fill_value=0)
        for a in ACTIVITIES:
            F[f"others_in_my_repos_{a}_7d"] = others[a]
        pairs = user_repo.merge(r7[["repo_id", "gh"]].drop_duplicates(), on="repo_id", suffixes=("", "_other"))
        pairs = pairs[pairs["gh"] != pairs["gh_other"]]
        F["coworkers_active_7d"] = pairs.groupby("gh")["gh_other"].nunique()

        # ---------------- relational: user -> their PRs / issues <- other users
        rec = self.received[(self.received["row_time"] < t) & (self.received["time"] < t)]
        rec = rec[(t - rec["time"]) / DAY < 7]
        rc = rec.groupby(["owner", "kind"]).size().unstack("kind")
        for k in ["review", "review_comment", "issue_comment"]:
            F[f"received_{k}_7d"] = rc[k] if k in rc else np.nan

        pr = self.pr[self.pr["first_seen_at"] < t]
        still_open = ~(pr["closed_at"] < t) & ~(pr["merged_at"] < t)
        recent_pr = (t - pr["created_at"]) / DAY < 60
        F["open_prs"] = pr[still_open & recent_pr].groupby("author").size()
        merged = pr[(pr["merged_at"] < t) & ((t - pr["merged_at"]) / DAY < 28)]
        F["prs_merged_by_user_28d"] = merged.groupby("merger").size()

        # missing = nothing found before t
        days = [c for c in F if c.startswith("days_since_")]
        F[days] = F[days].fillna(CAP).clip(upper=CAP)
        F["days_in_data"] = F["days_in_data"].fillna(-1)
        return F.fillna(0).astype(float)


def feature_groups(columns):
    """Column names of each group, for the ablation study."""
    own = [c for c in columns if c.startswith("own_") and c.endswith("_7d")]
    relational = [c for c in columns if c.startswith(("n_repos", "others_in_my_repos", "coworkers",
                                                      "received_", "open_prs", "prs_merged"))]
    temporal = [c for c in columns if c not in own and c not in relational]
    return {"own": own, "temporal": temporal, "relational": relational}
