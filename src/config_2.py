# src/config.py

from pathlib import Path

BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed(job matcher)"

POSTINGS_PATH = RAW_DIR / "postings.csv"

JOB_SKILLS_PATH = RAW_DIR / "jobs" / "job_skills.csv"
JOB_INDUSTRIES_PATH = RAW_DIR / "jobs" / "job_industries.csv"

SKILLS_PATH = RAW_DIR / "mappings" / "skills.csv"
INDUSTRIES_PATH = RAW_DIR / "mappings" / "industries.csv"

PROCESSED_JOBS_PATH = PROCESSED_DIR / "processed_jobs.csv"