# Big Data Assignment 2 – Predicting GitHub User Activity

Predict, for each test user, which of 6 activities (open_pr, open_issue, issue_comment,
pr_review, pr_review_comment, push) they perform during 2026/11/22 00:00 – 11/29 00:00
(Taiwan time).

## Data Source

The data used in this project comes from [GH Archive](https://www.gharchive.org/), a public
archive of GitHub events. It is used for educational purposes within the course
*Big Data Analytics Techniques and Applications* (Fall 2026).

## Repository structure

```
.
├── assignment2/              # course files (unchanged): preprocess.py, evaluate.py, cohort.txt, test_input.csv
├── src/
│   ├── download_gharchive.py # step 1: download the GH Archive hour files
│   ├── activities.py         # load the tables; when each user did each activity
│   ├── baseline.py           # "Copy last week" baseline + validation answers
│   ├── check_data.py         # data quality checks and activity-per-day figure
│   ├── features.py           # own / temporal / relational features of a user at time t
│   └── model.py              # training, validation (ablation study), final prediction
├── requirements.txt
└── README.md
```

## How to run

### 0. Install

```bash
pip install -r requirements.txt
```

### 1. Download GH Archive (period used: 2026-10-01 00:00 UTC → 2026-11-20 08:59 UTC)

```bash
python src/download_gharchive.py --start 2026-10-01-0 --end 2026-11-20-8 --out raw
```

By default the script keeps only the lines of the 9,000 cohort users while downloading
(the same filter as Step 2 of `preprocess.py`, with the same cohort file and hash), which
saves disk space; `preprocess.py` produces the same tables from these files as from the
full GH Archive files. Use `--full` to keep the original files unchanged.

### 2. Build the 8 tables (course script)

```bash
python assignment2/scripts/preprocess.py --raw raw --out data
```

### 3. Baseline "Copy last week" and validation

`--t` is the prediction time in Taiwan time. Validation week: 2026/11/08 00:00 – 11/15 00:00.

```bash
python src/baseline.py --data data --t "2026-11-08 00:00" --pred val_pred.csv --truth val_truth.csv
python assignment2/scripts/evaluate.py --pred val_pred.csv --truth val_truth.csv
```

### 4. Model: validation (ablation study) and final prediction

Features (`src/features.py`), computed at each prediction time t from rows seen before t:
the user's own activity in the last 7 days, temporal features (1/3/14/28-day counts, days since
the last activity, time-decayed counts, global activity volume) and relational features obtained
by following the foreign keys (user → repositories → other users' activity; user → their PRs /
issues → reviews and comments by others; PRs merged by the user). Model (`src/model.py`): one
gradient-boosted tree classifier per activity, trained on daily prediction times; the 0/1
threshold of each activity maximizes its F1 on the last week with answers.
The data stops on 2026-11-20 (about 32 hours before the test week), so the final prediction
computes the features at the last hour with data (`--features-at-data-end`): otherwise the
hours without data would look like every user had stopped.

```bash
python src/check_data.py --raw raw --data data           # data quality: missing hours, activity per day
python src/model.py validate --data data                 # week 2026/11/08-11/15, ablation study -> results/
python src/model.py predict --data data --student-id {studentID} --features-at-data-end   # -> output_{studentID}.csv
```

### 5. Check the output format

```bash
python assignment2/scripts/evaluate.py --pred output_{studentID}.csv --input assignment2/submission/test_input.csv
```
