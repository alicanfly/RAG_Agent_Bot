"""
Data Cleaning and Split Builder
Author: Talha Zaheer
Roll No: 23i-2609

Purpose:
This file cleans the raw review CSV, creates sentiment and length labels,
splits the data into train/validation/test CSV files, and builds the vocabulary
used by the encoder and decoder.

No model is trained in this file.
"""

import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
BASE_DIR = Path(__file__).resolve().parent
RAW_CSV_PATH = BASE_DIR / "TZ" / "TZ.csv"
DATA_DIR = BASE_DIR / "data"


def CLEAN_TEXT(TEXT_VALUE):
    """Remove noisy markup, URLs, non-ASCII symbols, and repeated spaces."""
    TEXT_VALUE = str(TEXT_VALUE)
    TEXT_VALUE = re.sub(r"<[^>]+>", " ", TEXT_VALUE)
    TEXT_VALUE = re.sub(r"http\S+|www\.\S+", " ", TEXT_VALUE)
    TEXT_VALUE = re.sub(r"[^\x00-\x7F]+", " ", TEXT_VALUE)
    TEXT_VALUE = re.sub(r"[^a-zA-Z0-9\s\.,!?'\"-]", " ", TEXT_VALUE)
    TEXT_VALUE = re.sub(r"\s+", " ", TEXT_VALUE).strip()
    return TEXT_VALUE


def MAP_SENTIMENT(RATING):
    """Convert numeric rating into Negative, Neutral, or Positive."""
    if RATING <= 2:
        return "Negative"
    if RATING == 3:
        return "Neutral"
    return "Positive"


def REVIEW_LENGTH_BUCKET(TEXT_VALUE):
    """Convert review word count into short, medium, or long class IDs."""
    WORD_COUNT = len(str(TEXT_VALUE).split())
    if WORD_COUNT < 20:
        return 0
    if WORD_COUNT < 60:
        return 1
    return 2


def BUILD_VOCAB(TRAIN_TEXTS):
    """Build a word vocabulary using words that appear at least twice."""
    ALL_WORDS = []
    for TEXT_VALUE in TRAIN_TEXTS:
        ALL_WORDS.extend(str(TEXT_VALUE).lower().split())

    WORD_COUNTS = Counter(ALL_WORDS)
    VOCAB = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
    for WORD, COUNT in WORD_COUNTS.most_common():
        if COUNT >= 2:
            VOCAB[WORD] = len(VOCAB)
    return VOCAB


def MAIN():
    """Clean, label, split, and save all preprocessing artifacts."""
    np.random.seed(RANDOM_SEED)
    os.makedirs(DATA_DIR, exist_ok=True)

    DATAFRAME = pd.read_csv(RAW_CSV_PATH)
    DATAFRAME.columns = [COLUMN.strip().lower() for COLUMN in DATAFRAME.columns]

    DATAFRAME = DATAFRAME.dropna(subset=["text", "rating"])
    DATAFRAME["rating"] = pd.to_numeric(DATAFRAME["rating"], errors="coerce")
    DATAFRAME = DATAFRAME.dropna(subset=["rating"])
    DATAFRAME = DATAFRAME[DATAFRAME["rating"].between(1, 5)]
    DATAFRAME["rating"] = DATAFRAME["rating"].astype(int)

    DATAFRAME["text"] = DATAFRAME["text"].apply(CLEAN_TEXT)
    DATAFRAME = DATAFRAME[DATAFRAME["text"].str.len() >= 10]
    DATAFRAME = DATAFRAME[DATAFRAME["text"].str.len() <= 5000]
    DATAFRAME = DATAFRAME.drop_duplicates(subset=["text"]).reset_index(drop=True)

    DATAFRAME["sentiment"] = DATAFRAME["rating"].apply(MAP_SENTIMENT)
    DATAFRAME["length_label"] = DATAFRAME["text"].apply(REVIEW_LENGTH_BUCKET)
    DATAFRAME = DATAFRAME.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    TOTAL_ROWS = len(DATAFRAME)
    TRAIN_END = int(0.70 * TOTAL_ROWS)
    VAL_END = int(0.85 * TOTAL_ROWS)

    TRAIN_DF = DATAFRAME.iloc[:TRAIN_END].reset_index(drop=True)
    VAL_DF = DATAFRAME.iloc[TRAIN_END:VAL_END].reset_index(drop=True)
    TEST_DF = DATAFRAME.iloc[VAL_END:].reset_index(drop=True)

    TRAIN_DF.to_csv(DATA_DIR / "train.csv", index=False)
    VAL_DF.to_csv(DATA_DIR / "val.csv", index=False)
    TEST_DF.to_csv(DATA_DIR / "test.csv", index=False)

    VOCAB = BUILD_VOCAB(TRAIN_DF["text"])
    with open(DATA_DIR / "vocab.json", "w", encoding="utf-8") as FILE_HANDLE:
        json.dump(VOCAB, FILE_HANDLE, indent=2)

    print(f"Total clean samples : {TOTAL_ROWS}")
    print(f"Train               : {len(TRAIN_DF)}")
    print(f"Val                 : {len(VAL_DF)}")
    print(f"Test                : {len(TEST_DF)}")
    print("\nSentiment distribution (train):")
    print(TRAIN_DF["sentiment"].value_counts())
    print("\nLength label distribution (train):")
    print(TRAIN_DF["length_label"].value_counts().sort_index())
    print(f"\nVocabulary size: {len(VOCAB)}")


if __name__ == "__main__":
    MAIN()

