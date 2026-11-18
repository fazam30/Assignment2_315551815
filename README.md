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
│   └── download_gharchive.py # step 1: download the GH Archive hour files
├── requirements.txt
└── README.md
```

## How to run

### 0. Install

```bash
pip install -r requirements.txt
```

### 1. Download GH Archive (period used: 2026-10-01 00:00 UTC → 2026-11-21 UTC)

```bash
python src/download_gharchive.py --start 2026-10-01-0 --end 2026-11-21-14 --out raw
```

By default the script keeps only the lines of the 9,000 cohort users while downloading
(the same filter as Step 2 of `preprocess.py`, with the same cohort file and hash), which
saves disk space; `preprocess.py` produces the same tables from these files as from the
full GH Archive files. Use `--full` to keep the original files unchanged.

### 2. Build the 8 tables (course script)

```bash
python assignment2/scripts/preprocess.py --raw raw --out data
```

### 3. Model and prediction

*(to be completed)*

### 4. Check the output format

```bash
python assignment2/scripts/evaluate.py --pred output_{studentID}.csv --input assignment2/submission/test_input.csv
```
