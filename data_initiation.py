"""
Dataset Builder
Author: Talha Zaheer
Roll No: 23i-2609

Purpose:
This file creates the raw combined review dataset used by the RAG pipeline.
It reads compressed Amazon review JSON files, extracts only review text and
rating, shuffles the collected records, and saves a fixed-size CSV.

No model is trained in this file.
"""

import gzip
import json
import random
from pathlib import Path

import pandas as pd


RANDOM_SEED = 42
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "TZ"
OUTPUT_FILE = RAW_DIR / "TZ.csv"
SOURCE_FILES = [
    "Industrial_and_Scientific.json.gz",
    "Digital_Music.json.gz",
    "Musical_Instruments.json.gz",
    "Prime_Pantry.json.gz",
]
TARGET_SIZE = 42000


def READ_JSON_GZ(FILE_PATH):
    """Read one .json.gz file and return valid review rows."""
    REVIEW_ROWS = []
    with gzip.open(FILE_PATH, "rt", encoding="utf-8") as FILE_HANDLE:
        for LINE in FILE_HANDLE:
            try:
                REVIEW = json.loads(LINE)
                if "reviewText" in REVIEW and "overall" in REVIEW:
                    REVIEW_ROWS.append({"text": REVIEW["reviewText"], "rating": REVIEW["overall"]})
            except json.JSONDecodeError:
                continue
    return REVIEW_ROWS


def MAIN():
    """Build the combined raw CSV from the configured compressed files."""
    random.seed(RANDOM_SEED)
    ALL_ROWS = []

    print("Preparing dataset for Talha Zaheer - 23i-2609")
    for FILE_NAME in SOURCE_FILES:
        FILE_PATH = RAW_DIR / FILE_NAME
        print(f"Reading {FILE_PATH}...")
        ALL_ROWS.extend(READ_JSON_GZ(FILE_PATH))

    print(f"Collected rows before sampling: {len(ALL_ROWS)}")
    random.shuffle(ALL_ROWS)
    FINAL_ROWS = ALL_ROWS[:TARGET_SIZE]

    OUTPUT_DF = pd.DataFrame(FINAL_ROWS)
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    OUTPUT_DF.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(OUTPUT_DF)} rows -> {OUTPUT_FILE}")


if __name__ == "__main__":
    MAIN()

