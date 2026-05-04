"""
Task B - Retrieval Context Builder
Author: Talha Zaheer
Roll No: 23i-2609

Purpose:
Load the trained encoder and saved training embeddings, retrieve the most
similar training reviews for each test review, and save RAG contexts for the
decoder. This file does not train any model.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity

from A import DEVICE, EMBED_DIM, ENCODER_MODEL, MAX_LEN, NUM_HEADS


BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "models"
RESULTS_DIR = BASE / "results"
TOP_K = 5


def TOKENIZE_AND_ENCODE(TEXT_VALUE, VOCAB, PAD_IDX, UNK_IDX):
    """Convert one review into padded token IDs."""
    TOKEN_IDS = [VOCAB.get(TOKEN, UNK_IDX) for TOKEN in str(TEXT_VALUE).lower().split()[:MAX_LEN]]
    return TOKEN_IDS + [PAD_IDX] * (MAX_LEN - len(TOKEN_IDS))


def ENCODE_TEXT(TEXT_VALUE, ENCODER, VOCAB, PAD_IDX, UNK_IDX):
    """Use the encoder to produce one normalized review embedding."""
    TOKEN_IDS = TOKENIZE_AND_ENCODE(TEXT_VALUE, VOCAB, PAD_IDX, UNK_IDX)
    INPUT_TENSOR = torch.tensor([TOKEN_IDS], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        _, _, EMBEDDING = ENCODER(INPUT_TENSOR)
    VECTOR = EMBEDDING.squeeze(0).cpu().numpy()
    return VECTOR / (np.linalg.norm(VECTOR) + 1e-9)


def RETRIEVE(QUERY_TEXT, ENCODER, TRAIN_EMBEDDINGS_NORM, TRAIN_DF, VOCAB, PAD_IDX, UNK_IDX, K_VALUE=TOP_K):
    """Return top-k retrieved training rows using cosine similarity."""
    QUERY_VECTOR = ENCODE_TEXT(QUERY_TEXT, ENCODER, VOCAB, PAD_IDX, UNK_IDX).reshape(1, -1)
    SIMILARITIES = cosine_similarity(QUERY_VECTOR, TRAIN_EMBEDDINGS_NORM)[0]
    TOP_INDICES = np.argsort(SIMILARITIES)[::-1][:K_VALUE]
    RESULTS = []
    for INDEX in TOP_INDICES:
        RESULTS.append(
            {
                "text": TRAIN_DF.iloc[INDEX]["text"],
                "sentiment": TRAIN_DF.iloc[INDEX]["sentiment"],
                "length_label": int(TRAIN_DF.iloc[INDEX]["length_label"]),
                "score": float(SIMILARITIES[INDEX]),
                "train_idx": int(INDEX),
            }
        )
    return RESULTS


def BUILD_CONTEXT_STRING(RETRIEVED_ROWS):
    """Format retrieved examples into the compact RAG context string."""
    PARTS = []
    for POSITION, ROW in enumerate(RETRIEVED_ROWS):
        PARTS.append(f"[Example {POSITION + 1}] ({ROW['sentiment']}) {ROW['text'][:200]}")
    return " | ".join(PARTS)


def MAIN():
    """Build retrieval reports, precision plot, and test context CSV."""
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(DATA_DIR / "vocab.json", encoding="utf-8") as FILE_HANDLE:
        VOCAB = json.load(FILE_HANDLE)
    PAD_IDX, UNK_IDX = VOCAB["<PAD>"], VOCAB["<UNK>"]

    ENCODER = ENCODER_MODEL(len(VOCAB), PAD_IDX).to(DEVICE)
    ENCODER.load_state_dict(torch.load(MODEL_DIR / "encoder_best.pt", map_location=DEVICE))
    ENCODER.eval()

    TRAIN_EMBEDDINGS = np.load(RESULTS_DIR / "train_embeddings.npy")
    NORMS = np.linalg.norm(TRAIN_EMBEDDINGS, axis=1, keepdims=True)
    NORMS[NORMS == 0] = 1e-9
    TRAIN_EMBEDDINGS_NORM = TRAIN_EMBEDDINGS / NORMS

    TRAIN_DF = pd.read_csv(DATA_DIR / "train.csv")
    TEST_DF = pd.read_csv(DATA_DIR / "test.csv")

    print("Retrieval quality analysis on 10 test samples:")
    print("=" * 80)
    for INDEX in range(min(10, len(TEST_DF))):
        ROW = TEST_DF.iloc[INDEX]
        RETRIEVED = RETRIEVE(ROW["text"], ENCODER, TRAIN_EMBEDDINGS_NORM, TRAIN_DF, VOCAB, PAD_IDX, UNK_IDX)
        print(f"\nQuery [{ROW['sentiment']}]: {ROW['text'][:120]}...")
        for RESULT in RETRIEVED:
            MATCH = "Y" if RESULT["sentiment"] == ROW["sentiment"] else "N"
            print(f"  [{MATCH}] [{RESULT['sentiment']}] sim={RESULT['score']:.4f}: {RESULT['text'][:80]}...")

    K_VALUES = [1, 3, 5, 10]
    PRECISION_VALUES = []
    for K_VALUE in K_VALUES:
        HITS = []
        for INDEX in range(min(200, len(TEST_DF))):
            ROW = TEST_DF.iloc[INDEX]
            RETRIEVED = RETRIEVE(ROW["text"], ENCODER, TRAIN_EMBEDDINGS_NORM, TRAIN_DF, VOCAB, PAD_IDX, UNK_IDX, K_VALUE)
            HITS.append(sum(1 for RESULT in RETRIEVED if RESULT["sentiment"] == ROW["sentiment"]) / K_VALUE)
        PRECISION_VALUES.append(np.mean(HITS))
        print(f"Precision@{K_VALUE}: {np.mean(HITS):.4f}")

    plt.figure(figsize=(6, 4))
    plt.plot(K_VALUES, PRECISION_VALUES, marker="o")
    plt.xlabel("k")
    plt.ylabel("Precision@k (same sentiment)")
    plt.title("Retrieval Precision vs k")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "retrieval_precision.png", dpi=150)
    plt.close()

    TEST_CONTEXTS = []
    for INDEX in range(len(TEST_DF)):
        ROW = TEST_DF.iloc[INDEX]
        RETRIEVED = RETRIEVE(ROW["text"], ENCODER, TRAIN_EMBEDDINGS_NORM, TRAIN_DF, VOCAB, PAD_IDX, UNK_IDX)
        TEST_CONTEXTS.append(BUILD_CONTEXT_STRING(RETRIEVED))
        if (INDEX + 1) % 500 == 0:
            print(f"Retrieved context for {INDEX + 1}/{len(TEST_DF)} test samples...")

    pd.Series(TEST_CONTEXTS).to_csv(RESULTS_DIR / "test_contexts.csv", index=False, header=["context"])
    print(f"Saved {len(TEST_CONTEXTS)} test contexts -> {RESULTS_DIR / 'test_contexts.csv'}")


if __name__ == "__main__":
    MAIN()

