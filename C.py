"""
Task C - Explanation Decoder Training and RAG Evaluation
Author: Talha Zaheer
Roll No: 23i-2609

Purpose:
Train a decoder-only language model for short explanations, then compare
perplexity with and without retrieved RAG context.
"""

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "models"
RESULTS_DIR = BASE / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_SRC_LEN = 256
MAX_TGT_LEN = 40
BATCH_SIZE = 8
EMBED_DIM = 128
NUM_HEADS = 4
FF_DIM = 256
NUM_LAYERS = 3
DROPOUT = 0.1
EPOCHS = 12
LR = 5e-4
SENTIMENT_MAP = {"Negative": 0, "Neutral": 1, "Positive": 2}
SENT_TOKENS = ["[NEG]", "[NEU]", "[POS]"]
LEN_TOKENS = ["[SHORT]", "[MED]", "[LONG]"]


def TOKENIZE(TEXT_VALUE, VOCAB, UNK_IDX, MAX_LEN):
    """Whitespace tokenizer matching the original project behavior."""
    return [VOCAB.get(TOKEN, UNK_IDX) for TOKEN in str(TEXT_VALUE).lower().split()[:MAX_LEN]]


def BUILD_INPUT_SEQUENCE(REVIEW_TEXT, SENTIMENT_ID, LENGTH_ID, CONTEXT_TEXT, VOCAB, PAD_IDX, UNK_IDX, BOS_IDX):
    """Build decoder source sequence from labels, review, and optional context."""
    SENT_TOKEN_ID = VOCAB.get(SENT_TOKENS[SENTIMENT_ID], UNK_IDX)
    LEN_TOKEN_ID = VOCAB.get(LEN_TOKENS[LENGTH_ID], UNK_IDX)
    REVIEW_IDS = TOKENIZE(REVIEW_TEXT, VOCAB, UNK_IDX, 100)
    CONTEXT_IDS = TOKENIZE(CONTEXT_TEXT, VOCAB, UNK_IDX, 120)
    SEQUENCE = [BOS_IDX, SENT_TOKEN_ID, LEN_TOKEN_ID] + REVIEW_IDS + [VOCAB.get("|", UNK_IDX)] + CONTEXT_IDS
    SEQUENCE = SEQUENCE[:MAX_SRC_LEN]
    return SEQUENCE + [PAD_IDX] * (MAX_SRC_LEN - len(SEQUENCE))


def BUILD_REFERENCE(SENTIMENT_LABEL, REVIEW_TEXT, VOCAB, PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX):
    """Build target explanation sentence for supervised decoder training."""
    LABEL_WORD = {"Negative": "negative", "Neutral": "neutral", "Positive": "positive"}[SENTIMENT_LABEL]
    EXCERPT = " ".join(str(REVIEW_TEXT).split()[:12])
    REFERENCE = f"this review is {LABEL_WORD} because {EXCERPT}"
    TOKEN_IDS = [BOS_IDX] + TOKENIZE(REFERENCE, VOCAB, UNK_IDX, MAX_TGT_LEN - 2) + [EOS_IDX]
    TOKEN_IDS = TOKEN_IDS + [PAD_IDX] * (MAX_TGT_LEN - len(TOKEN_IDS))
    return TOKEN_IDS[:MAX_TGT_LEN]


class EXPLANATION_DATASET(Dataset):
    """Dataset returning decoder source and target token sequences."""

    def __init__(self, DATA_PATH, CONTEXT_PATH, VOCAB, PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX):
        self.DATAFRAME = pd.read_csv(DATA_PATH)
        CONTEXT_DF = pd.read_csv(CONTEXT_PATH)
        self.CONTEXTS = CONTEXT_DF["context"].fillna("").tolist()
        self.VOCAB = VOCAB
        self.PAD_IDX = PAD_IDX
        self.UNK_IDX = UNK_IDX
        self.BOS_IDX = BOS_IDX
        self.EOS_IDX = EOS_IDX

    def __len__(self):
        return len(self.DATAFRAME)

    def __getitem__(self, INDEX):
        ROW = self.DATAFRAME.iloc[INDEX]
        CONTEXT_TEXT = self.CONTEXTS[INDEX] if INDEX < len(self.CONTEXTS) else ""
        SOURCE = BUILD_INPUT_SEQUENCE(
            ROW["text"], SENTIMENT_MAP[ROW["sentiment"]], int(ROW["length_label"]),
            CONTEXT_TEXT, self.VOCAB, self.PAD_IDX, self.UNK_IDX, self.BOS_IDX
        )
        TARGET = BUILD_REFERENCE(ROW["sentiment"], ROW["text"], self.VOCAB, self.PAD_IDX, self.UNK_IDX, self.BOS_IDX, self.EOS_IDX)
        return torch.tensor(SOURCE, dtype=torch.long), torch.tensor(TARGET, dtype=torch.long)


class ATTENTION(nn.Module):
    """Scaled dot-product attention for decoder self-attention."""

    def __init__(self):
        super().__init__()
        self.DROPOUT_LAYER = nn.Dropout(DROPOUT)

    def forward(self, QUERY, KEY, VALUE, ATTN_MASK=None, PAD_MASK=None):
        SCORES = torch.matmul(QUERY, KEY.transpose(-2, -1)) / math.sqrt(QUERY.size(-1))
        if ATTN_MASK is not None:
            SCORES = SCORES + ATTN_MASK
        if PAD_MASK is not None:
            SCORES = SCORES.masked_fill(PAD_MASK.unsqueeze(1).unsqueeze(2), float("-inf"))
        WEIGHTS = self.DROPOUT_LAYER(F.softmax(SCORES, dim=-1))
        return torch.matmul(WEIGHTS, VALUE)


class MULTI_HEAD_SELF_ATTENTION(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self):
        super().__init__()
        self.HEADS = NUM_HEADS
        self.HEAD_DIM = EMBED_DIM // NUM_HEADS
        self.W_Q = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.W_K = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.W_V = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.W_O = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.ATTENTION = ATTENTION()

    def forward(self, INPUT_TENSOR, CAUSAL_MASK=None, PAD_MASK=None):
        BATCH, TIME, _ = INPUT_TENSOR.shape
        QUERY = self.W_Q(INPUT_TENSOR).view(BATCH, TIME, self.HEADS, self.HEAD_DIM).transpose(1, 2)
        KEY = self.W_K(INPUT_TENSOR).view(BATCH, TIME, self.HEADS, self.HEAD_DIM).transpose(1, 2)
        VALUE = self.W_V(INPUT_TENSOR).view(BATCH, TIME, self.HEADS, self.HEAD_DIM).transpose(1, 2)
        OUTPUT = self.ATTENTION(QUERY, KEY, VALUE, CAUSAL_MASK, PAD_MASK)
        OUTPUT = OUTPUT.transpose(1, 2).contiguous().view(BATCH, TIME, -1)
        return self.W_O(OUTPUT)


class DECODER_BLOCK(nn.Module):
    """Causal transformer decoder block."""

    def __init__(self):
        super().__init__()
        self.ATTENTION = MULTI_HEAD_SELF_ATTENTION()
        self.FF = nn.Sequential(nn.Linear(EMBED_DIM, FF_DIM), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(FF_DIM, EMBED_DIM))
        self.NORM_1 = nn.LayerNorm(EMBED_DIM)
        self.NORM_2 = nn.LayerNorm(EMBED_DIM)
        self.DROP = nn.Dropout(DROPOUT)

    def forward(self, INPUT_TENSOR, CAUSAL_MASK=None, PAD_MASK=None):
        INPUT_TENSOR = self.NORM_1(INPUT_TENSOR + self.DROP(self.ATTENTION(INPUT_TENSOR, CAUSAL_MASK, PAD_MASK)))
        return self.NORM_2(INPUT_TENSOR + self.DROP(self.FF(INPUT_TENSOR)))


class POSITIONAL_ENCODING(nn.Module):
    """Sinusoidal position encoding for decoder tokens."""

    def __init__(self, MAX_LEN):
        super().__init__()
        self.DROP = nn.Dropout(DROPOUT)
        POSITION_TABLE = torch.zeros(MAX_LEN, EMBED_DIM)
        POSITION = torch.arange(0, MAX_LEN).unsqueeze(1).float()
        DIVISOR = torch.exp(torch.arange(0, EMBED_DIM, 2).float() * (-math.log(10000.0) / EMBED_DIM))
        POSITION_TABLE[:, 0::2] = torch.sin(POSITION * DIVISOR)
        POSITION_TABLE[:, 1::2] = torch.cos(POSITION * DIVISOR)
        self.register_buffer("POSITION_TABLE", POSITION_TABLE.unsqueeze(0))

    def forward(self, INPUT_TENSOR):
        return self.DROP(INPUT_TENSOR + self.POSITION_TABLE[:, : INPUT_TENSOR.size(1)])


class DECODER_LM(nn.Module):
    """Decoder-only language model."""

    def __init__(self, VOCAB_SIZE, PAD_IDX):
        super().__init__()
        self.PAD_IDX = PAD_IDX
        self.EMBED = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=PAD_IDX)
        self.POSITION = POSITIONAL_ENCODING(MAX_SRC_LEN + MAX_TGT_LEN)
        self.LAYERS = nn.ModuleList([DECODER_BLOCK() for _ in range(NUM_LAYERS)])
        self.NORM = nn.LayerNorm(EMBED_DIM)
        self.LM_HEAD = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def MAKE_CAUSAL_MASK(self, TIME_STEPS):
        MASK = torch.triu(torch.ones(TIME_STEPS, TIME_STEPS), diagonal=1).bool()
        return MASK.float().masked_fill(MASK, float("-inf")).to(DEVICE)

    def forward(self, TOKEN_IDS, PAD_MASK=None):
        CAUSAL_MASK = self.MAKE_CAUSAL_MASK(TOKEN_IDS.size(1))
        OUTPUT = self.POSITION(self.EMBED(TOKEN_IDS))
        for LAYER in self.LAYERS:
            OUTPUT = LAYER(OUTPUT, CAUSAL_MASK, PAD_MASK)
        return self.LM_HEAD(self.NORM(OUTPUT))


def RUN_EPOCH(MODEL, LOADER, OPTIMIZER, CRITERION, PAD_IDX, VOCAB_SIZE, IS_TRAINING):
    """Train or validate for one epoch."""
    MODEL.train() if IS_TRAINING else MODEL.eval()
    TOTAL_LOSS = 0
    CONTEXT = torch.enable_grad() if IS_TRAINING else torch.no_grad()
    with CONTEXT:
        for BATCH_INDEX, (SOURCE, TARGET) in enumerate(LOADER):
            SOURCE, TARGET = SOURCE.to(DEVICE), TARGET.to(DEVICE)
            FULL_SEQUENCE = torch.cat([SOURCE, TARGET[:, :-1]], dim=1)
            PAD_MASK = FULL_SEQUENCE == PAD_IDX
            LOGITS = MODEL(FULL_SEQUENCE, PAD_MASK)
            SOURCE_LEN, TARGET_LEN = SOURCE.size(1), TARGET.size(1)
            TARGET_LOGITS = LOGITS[:, SOURCE_LEN : SOURCE_LEN + TARGET_LEN - 1, :]
            LOSS = CRITERION(TARGET_LOGITS.reshape(-1, VOCAB_SIZE), TARGET[:, 1:].reshape(-1))
            if IS_TRAINING:
                OPTIMIZER.zero_grad()
                LOSS.backward()
                nn.utils.clip_grad_norm_(MODEL.parameters(), 1.0)
                OPTIMIZER.step()
            TOTAL_LOSS += LOSS.item()
            if (BATCH_INDEX + 1) % 100 == 0:
                MODE = "Train" if IS_TRAINING else "Val"
                print(f"  [{MODE}] batch {BATCH_INDEX + 1}/{len(LOADER)} loss={TOTAL_LOSS / (BATCH_INDEX + 1):.4f}", flush=True)
    return TOTAL_LOSS / len(LOADER)


def COMPUTE_PERPLEXITY(MODEL, LOADER, PAD_IDX, VOCAB_SIZE):
    """Compute perplexity over non-padding target tokens."""
    MODEL.eval()
    TOTAL_LOSS, TOTAL_TOKENS = 0, 0
    CRITERION = nn.CrossEntropyLoss(ignore_index=PAD_IDX, reduction="sum")
    with torch.no_grad():
        for SOURCE, TARGET in LOADER:
            SOURCE, TARGET = SOURCE.to(DEVICE), TARGET.to(DEVICE)
            FULL_SEQUENCE = torch.cat([SOURCE, TARGET[:, :-1]], dim=1)
            LOGITS = MODEL(FULL_SEQUENCE, FULL_SEQUENCE == PAD_IDX)
            SOURCE_LEN, TARGET_LEN = SOURCE.size(1), TARGET.size(1)
            TARGET_LOGITS = LOGITS[:, SOURCE_LEN : SOURCE_LEN + TARGET_LEN - 1, :]
            TOTAL_LOSS += CRITERION(TARGET_LOGITS.reshape(-1, VOCAB_SIZE), TARGET[:, 1:].reshape(-1)).item()
            TOTAL_TOKENS += (TARGET[:, 1:] != PAD_IDX).sum().item()
    return math.exp(TOTAL_LOSS / max(TOTAL_TOKENS, 1))


def MAKE_DUMMY_CONTEXT_FILE(NUM_ROWS, OUTPUT_PATH):
    """Create an empty context file for no-RAG baseline evaluation."""
    pd.Series([""] * NUM_ROWS).to_csv(OUTPUT_PATH, index=False, header=["context"])


def GENERATE(MODEL, SOURCE_IDS, BOS_IDX, EOS_IDX, PAD_IDX, INV_VOCAB):
    """Greedy explanation generation for qualitative examples."""
    MODEL.eval()
    GENERATED = [BOS_IDX]
    with torch.no_grad():
        for _ in range(MAX_TGT_LEN):
            INPUT_IDS = torch.tensor([SOURCE_IDS + GENERATED], dtype=torch.long).to(DEVICE)
            LOGITS = MODEL(INPUT_IDS, INPUT_IDS == PAD_IDX)
            NEXT_TOKEN = LOGITS[0, -1, :].argmax(-1).item()
            if NEXT_TOKEN == EOS_IDX:
                break
            GENERATED.append(NEXT_TOKEN)
    return " ".join(INV_VOCAB.get(TOKEN_ID, "<UNK>") for TOKEN_ID in GENERATED[1:])


def MAIN():
    """Train decoder, evaluate RAG ablation, and save outputs."""
    sys.stdout.reconfigure(line_buffering=True)
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    with open(DATA_DIR / "vocab.json", encoding="utf-8") as FILE_HANDLE:
        VOCAB = json.load(FILE_HANDLE)
    for TOKEN in SENT_TOKENS + LEN_TOKENS:
        if TOKEN not in VOCAB:
            VOCAB[TOKEN] = len(VOCAB)

    PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX = VOCAB["<PAD>"], VOCAB["<UNK>"], VOCAB["<BOS>"], VOCAB["<EOS>"]
    VOCAB_SIZE = len(VOCAB)
    INV_VOCAB = {INDEX: TOKEN for TOKEN, INDEX in VOCAB.items()}

    TRAIN_DF = pd.read_csv(DATA_DIR / "train.csv")
    VAL_DF = pd.read_csv(DATA_DIR / "val.csv")
    TEST_DF = pd.read_csv(DATA_DIR / "test.csv")
    TRAIN_CTX = RESULTS_DIR / "dummy_train_ctx.csv"
    VAL_CTX = RESULTS_DIR / "dummy_val_ctx.csv"
    TEST_NORAG_CTX = RESULTS_DIR / "dummy_test_norag_ctx.csv"
    TEST_RAG_CTX = RESULTS_DIR / "test_contexts.csv"
    MAKE_DUMMY_CONTEXT_FILE(len(TRAIN_DF), TRAIN_CTX)
    MAKE_DUMMY_CONTEXT_FILE(len(VAL_DF), VAL_CTX)
    MAKE_DUMMY_CONTEXT_FILE(len(TEST_DF), TEST_NORAG_CTX)

    TRAIN_DS = EXPLANATION_DATASET(DATA_DIR / "train.csv", TRAIN_CTX, VOCAB, PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX)
    VAL_DS = EXPLANATION_DATASET(DATA_DIR / "val.csv", VAL_CTX, VOCAB, PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX)
    TEST_RAG_DS = EXPLANATION_DATASET(DATA_DIR / "test.csv", TEST_RAG_CTX, VOCAB, PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX)
    TEST_NORAG_DS = EXPLANATION_DATASET(DATA_DIR / "test.csv", TEST_NORAG_CTX, VOCAB, PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX)
    TRAIN_LOADER = DataLoader(TRAIN_DS, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    VAL_LOADER = DataLoader(VAL_DS, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    TEST_RAG_LOADER = DataLoader(TEST_RAG_DS, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    TEST_NORAG_LOADER = DataLoader(TEST_NORAG_DS, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    MODEL = DECODER_LM(VOCAB_SIZE, PAD_IDX).to(DEVICE)
    CRITERION = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
    SCHEDULER = torch.optim.lr_scheduler.CosineAnnealingLR(OPTIMIZER, T_max=EPOCHS)

    TRAIN_LOSSES, VAL_LOSSES = [], []
    BEST_VAL_LOSS = float("inf")
    for EPOCH in range(1, EPOCHS + 1):
        TRAIN_LOSS = RUN_EPOCH(MODEL, TRAIN_LOADER, OPTIMIZER, CRITERION, PAD_IDX, VOCAB_SIZE, True)
        VAL_LOSS = RUN_EPOCH(MODEL, VAL_LOADER, OPTIMIZER, CRITERION, PAD_IDX, VOCAB_SIZE, False)
        SCHEDULER.step()
        TRAIN_LOSSES.append(TRAIN_LOSS); VAL_LOSSES.append(VAL_LOSS)
        print(f"Epoch {EPOCH:02d} | Train Loss {TRAIN_LOSS:.4f} | Val Loss {VAL_LOSS:.4f}", flush=True)
        if VAL_LOSS < BEST_VAL_LOSS:
            BEST_VAL_LOSS = VAL_LOSS
            torch.save(MODEL.state_dict(), MODEL_DIR / "decoder_best.pt")

    MODEL.load_state_dict(torch.load(MODEL_DIR / "decoder_best.pt", map_location=DEVICE))
    plt.figure(figsize=(7, 4))
    plt.plot(TRAIN_LOSSES, label="Train"); plt.plot(VAL_LOSSES, label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Decoder LM Training Curves"); plt.legend()
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "decoder_curves.png", dpi=150); plt.close()

    PPL_RAG = COMPUTE_PERPLEXITY(MODEL, TEST_RAG_LOADER, PAD_IDX, VOCAB_SIZE)
    PPL_NORAG = COMPUTE_PERPLEXITY(MODEL, TEST_NORAG_LOADER, PAD_IDX, VOCAB_SIZE)
    print(f"\nPerplexity (with RAG context)    : {PPL_RAG:.2f}")
    print(f"Perplexity (without RAG context) : {PPL_NORAG:.2f}")

    TEST_CONTEXTS = pd.read_csv(TEST_RAG_CTX)["context"].fillna("").tolist()
    print("\nGenerated Explanations - WITH RAG")
    for INDEX in range(min(5, len(TEST_DF))):
        ROW = TEST_DF.iloc[INDEX]
        SOURCE_IDS = BUILD_INPUT_SEQUENCE(
            ROW["text"], SENTIMENT_MAP[ROW["sentiment"]], int(ROW["length_label"]),
            TEST_CONTEXTS[INDEX] if INDEX < len(TEST_CONTEXTS) else "",
            VOCAB, PAD_IDX, UNK_IDX, BOS_IDX
        )
        print(f"\n[{INDEX + 1}] Review      : {ROW['text'][:120]}...")
        print(f"    True Sent   : {ROW['sentiment']}")
        print(f"    Explanation : {GENERATE(MODEL, SOURCE_IDS, BOS_IDX, EOS_IDX, PAD_IDX, INV_VOCAB)}")

    with open(RESULTS_DIR / "ablation.json", "w", encoding="utf-8") as FILE_HANDLE:
        json.dump({"perplexity_rag": PPL_RAG, "perplexity_norag": PPL_NORAG}, FILE_HANDLE, indent=2)


if __name__ == "__main__":
    MAIN()

