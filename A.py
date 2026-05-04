"""
Task A - Encoder Training
Author: Talha Zaheer
Roll No: 23i-2609

Purpose:
Train a transformer encoder for sentiment classification, review length
classification, and review embedding extraction. Training is inside MAIN() so
the file can be reviewed or imported without starting epochs.
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset


BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "models"
RESULTS_DIR = BASE / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 128
BATCH_SIZE = 64
EMBED_DIM = 128
NUM_HEADS = 4
FF_DIM = 256
NUM_LAYERS = 3
DROPOUT = 0.1
EPOCHS = 15
LR = 1e-3
WARMUP_STEPS = 500
SENTIMENT_MAP = {"Negative": 0, "Neutral": 1, "Positive": 2}


def TOKENIZE_AND_ENCODE(TEXT_VALUE, VOCAB, PAD_IDX, UNK_IDX):
    """Convert review text into fixed-length token IDs."""
    TOKEN_IDS = [VOCAB.get(TOKEN, UNK_IDX) for TOKEN in str(TEXT_VALUE).lower().split()[:MAX_LEN]]
    return TOKEN_IDS + [PAD_IDX] * (MAX_LEN - len(TOKEN_IDS))


class REVIEW_DATASET(Dataset):
    """Dataset returning token IDs, sentiment class, and length class."""

    def __init__(self, CSV_PATH, VOCAB, PAD_IDX, UNK_IDX):
        self.DATAFRAME = pd.read_csv(CSV_PATH)
        self.VOCAB = VOCAB
        self.PAD_IDX = PAD_IDX
        self.UNK_IDX = UNK_IDX

    def __len__(self):
        return len(self.DATAFRAME)

    def __getitem__(self, INDEX):
        ROW = self.DATAFRAME.iloc[INDEX]
        TOKEN_IDS = TOKENIZE_AND_ENCODE(ROW["text"], self.VOCAB, self.PAD_IDX, self.UNK_IDX)
        return (
            torch.tensor(TOKEN_IDS, dtype=torch.long),
            torch.tensor(SENTIMENT_MAP[ROW["sentiment"]], dtype=torch.long),
            torch.tensor(int(ROW["length_label"]), dtype=torch.long),
        )


class SCALED_DOT_PRODUCT_ATTENTION(nn.Module):
    """Scaled dot-product attention with optional padding mask."""

    def __init__(self):
        super().__init__()
        self.DROPOUT_LAYER = nn.Dropout(DROPOUT)

    def forward(self, QUERY, KEY, VALUE, PAD_MASK=None):
        SCORES = torch.matmul(QUERY, KEY.transpose(-2, -1)) / math.sqrt(QUERY.size(-1))
        if PAD_MASK is not None:
            SCORES = SCORES.masked_fill(PAD_MASK.unsqueeze(1).unsqueeze(2), float("-inf"))
        ATTENTION = self.DROPOUT_LAYER(F.softmax(SCORES, dim=-1))
        return torch.matmul(ATTENTION, VALUE)


class MULTI_HEAD_ATTENTION(nn.Module):
    """Multi-head self-attention used by the encoder block."""

    def __init__(self):
        super().__init__()
        self.HEADS = NUM_HEADS
        self.HEAD_DIM = EMBED_DIM // NUM_HEADS
        self.W_Q = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.W_K = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.W_V = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.W_O = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.ATTENTION = SCALED_DOT_PRODUCT_ATTENTION()

    def forward(self, INPUT_TENSOR, PAD_MASK=None):
        BATCH, TIME, _ = INPUT_TENSOR.shape
        QUERY = self.W_Q(INPUT_TENSOR).view(BATCH, TIME, self.HEADS, self.HEAD_DIM).transpose(1, 2)
        KEY = self.W_K(INPUT_TENSOR).view(BATCH, TIME, self.HEADS, self.HEAD_DIM).transpose(1, 2)
        VALUE = self.W_V(INPUT_TENSOR).view(BATCH, TIME, self.HEADS, self.HEAD_DIM).transpose(1, 2)
        OUTPUT = self.ATTENTION(QUERY, KEY, VALUE, PAD_MASK)
        OUTPUT = OUTPUT.transpose(1, 2).contiguous().view(BATCH, TIME, -1)
        return self.W_O(OUTPUT)


class FEED_FORWARD(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self):
        super().__init__()
        self.NETWORK = nn.Sequential(nn.Linear(EMBED_DIM, FF_DIM), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(FF_DIM, EMBED_DIM))

    def forward(self, INPUT_TENSOR):
        return self.NETWORK(INPUT_TENSOR)


class POSITIONAL_ENCODING(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self):
        super().__init__()
        self.DROPOUT_LAYER = nn.Dropout(DROPOUT)
        POSITION_TABLE = torch.zeros(MAX_LEN, EMBED_DIM)
        POSITION = torch.arange(0, MAX_LEN).unsqueeze(1).float()
        DIVISOR = torch.exp(torch.arange(0, EMBED_DIM, 2).float() * (-math.log(10000.0) / EMBED_DIM))
        POSITION_TABLE[:, 0::2] = torch.sin(POSITION * DIVISOR)
        POSITION_TABLE[:, 1::2] = torch.cos(POSITION * DIVISOR)
        self.register_buffer("POSITION_TABLE", POSITION_TABLE.unsqueeze(0))

    def forward(self, INPUT_TENSOR):
        return self.DROPOUT_LAYER(INPUT_TENSOR + self.POSITION_TABLE[:, : INPUT_TENSOR.size(1)])


class ENCODER_BLOCK(nn.Module):
    """Transformer encoder block with residual connections."""

    def __init__(self):
        super().__init__()
        self.ATTENTION = MULTI_HEAD_ATTENTION()
        self.FEED_FORWARD = FEED_FORWARD()
        self.NORM_1 = nn.LayerNorm(EMBED_DIM)
        self.NORM_2 = nn.LayerNorm(EMBED_DIM)
        self.DROPOUT_LAYER = nn.Dropout(DROPOUT)

    def forward(self, INPUT_TENSOR, PAD_MASK):
        INPUT_TENSOR = self.NORM_1(INPUT_TENSOR + self.DROPOUT_LAYER(self.ATTENTION(INPUT_TENSOR, PAD_MASK)))
        return self.NORM_2(INPUT_TENSOR + self.DROPOUT_LAYER(self.FEED_FORWARD(INPUT_TENSOR)))


class ENCODER_MODEL(nn.Module):
    """Review encoder with sentiment head, length head, and embedding output."""

    def __init__(self, VOCAB_SIZE, PAD_IDX):
        super().__init__()
        self.PAD_IDX = PAD_IDX
        self.EMBED = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=PAD_IDX)
        self.POSITION = POSITIONAL_ENCODING()
        self.LAYERS = nn.ModuleList([ENCODER_BLOCK() for _ in range(NUM_LAYERS)])
        self.SENTIMENT_HEAD = nn.Linear(EMBED_DIM, 3)
        self.LENGTH_HEAD = nn.Linear(EMBED_DIM, 3)

    def forward(self, TOKEN_IDS):
        PAD_MASK = TOKEN_IDS == self.PAD_IDX
        OUTPUT = self.POSITION(self.EMBED(TOKEN_IDS))
        for LAYER in self.LAYERS:
            OUTPUT = LAYER(OUTPUT, PAD_MASK)
        CLS_VECTOR = OUTPUT[:, 0, :]
        return self.SENTIMENT_HEAD(CLS_VECTOR), self.LENGTH_HEAD(CLS_VECTOR), CLS_VECTOR


def GET_LR(STEP):
    """Transformer warmup learning rate schedule."""
    STEP = max(STEP, 1)
    return EMBED_DIM ** (-0.5) * min(STEP ** (-0.5), STEP * WARMUP_STEPS ** (-1.5))


def RUN_EPOCH(MODEL, LOADER, OPTIMIZER, SCHEDULER, SENTIMENT_LOSS, LENGTH_LOSS, IS_TRAINING):
    """Run one train or evaluation epoch."""
    MODEL.train() if IS_TRAINING else MODEL.eval()
    TOTAL_LOSS = 0
    SENT_PRED, SENT_TRUE, LEN_PRED, LEN_TRUE = [], [], [], []
    CONTEXT = torch.enable_grad() if IS_TRAINING else torch.no_grad()
    with CONTEXT:
        for TOKEN_IDS, SENTIMENT_IDS, LENGTH_IDS in LOADER:
            TOKEN_IDS, SENTIMENT_IDS, LENGTH_IDS = TOKEN_IDS.to(DEVICE), SENTIMENT_IDS.to(DEVICE), LENGTH_IDS.to(DEVICE)
            SENT_LOGITS, LEN_LOGITS, _ = MODEL(TOKEN_IDS)
            LOSS = SENTIMENT_LOSS(SENT_LOGITS, SENTIMENT_IDS) + 0.5 * LENGTH_LOSS(LEN_LOGITS, LENGTH_IDS)
            if IS_TRAINING:
                OPTIMIZER.zero_grad()
                LOSS.backward()
                nn.utils.clip_grad_norm_(MODEL.parameters(), 1.0)
                OPTIMIZER.step()
                SCHEDULER.step()
            TOTAL_LOSS += LOSS.item()
            SENT_PRED.extend(SENT_LOGITS.argmax(-1).cpu().tolist())
            SENT_TRUE.extend(SENTIMENT_IDS.cpu().tolist())
            LEN_PRED.extend(LEN_LOGITS.argmax(-1).cpu().tolist())
            LEN_TRUE.extend(LENGTH_IDS.cpu().tolist())
    return (
        TOTAL_LOSS / len(LOADER),
        f1_score(SENT_TRUE, SENT_PRED, average="macro", zero_division=0),
        f1_score(LEN_TRUE, LEN_PRED, average="macro", zero_division=0),
    )


def MAIN():
    """Train encoder, save best model, plots, reports, and train embeddings."""
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    with open(DATA_DIR / "vocab.json", encoding="utf-8") as FILE_HANDLE:
        VOCAB = json.load(FILE_HANDLE)
    PAD_IDX, UNK_IDX = VOCAB["<PAD>"], VOCAB["<UNK>"]

    TRAIN_DS = REVIEW_DATASET(DATA_DIR / "train.csv", VOCAB, PAD_IDX, UNK_IDX)
    VAL_DS = REVIEW_DATASET(DATA_DIR / "val.csv", VOCAB, PAD_IDX, UNK_IDX)
    TEST_DS = REVIEW_DATASET(DATA_DIR / "test.csv", VOCAB, PAD_IDX, UNK_IDX)
    TRAIN_LOADER = DataLoader(TRAIN_DS, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    VAL_LOADER = DataLoader(VAL_DS, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    TEST_LOADER = DataLoader(TEST_DS, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    MODEL = ENCODER_MODEL(len(VOCAB), PAD_IDX).to(DEVICE)
    TRAIN_DF = pd.read_csv(DATA_DIR / "train.csv")
    SENT_COUNTS = TRAIN_DF["sentiment"].map(SENTIMENT_MAP).value_counts().sort_index()
    RAW_WEIGHTS = (1.0 / SENT_COUNTS.values).tolist()
    SENT_WEIGHTS = torch.tensor([W / sum(RAW_WEIGHTS) for W in RAW_WEIGHTS], dtype=torch.float).to(DEVICE)
    SENTIMENT_LOSS = nn.CrossEntropyLoss(weight=SENT_WEIGHTS)
    LENGTH_LOSS = nn.CrossEntropyLoss()
    OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
    SCHEDULER = torch.optim.lr_scheduler.LambdaLR(OPTIMIZER, lr_lambda=lambda STEP: GET_LR(STEP))

    HISTORY = {"train_loss": [], "val_loss": [], "train_f1_sent": [], "val_f1_sent": [], "train_f1_len": [], "val_f1_len": []}
    BEST_VAL_F1 = 0.0
    for EPOCH in range(1, EPOCHS + 1):
        TRAIN_LOSS, TRAIN_F1_SENT, TRAIN_F1_LEN = RUN_EPOCH(MODEL, TRAIN_LOADER, OPTIMIZER, SCHEDULER, SENTIMENT_LOSS, LENGTH_LOSS, True)
        VAL_LOSS, VAL_F1_SENT, VAL_F1_LEN = RUN_EPOCH(MODEL, VAL_LOADER, OPTIMIZER, SCHEDULER, SENTIMENT_LOSS, LENGTH_LOSS, False)
        HISTORY["train_loss"].append(TRAIN_LOSS); HISTORY["val_loss"].append(VAL_LOSS)
        HISTORY["train_f1_sent"].append(TRAIN_F1_SENT); HISTORY["val_f1_sent"].append(VAL_F1_SENT)
        HISTORY["train_f1_len"].append(TRAIN_F1_LEN); HISTORY["val_f1_len"].append(VAL_F1_LEN)
        print(f"Epoch {EPOCH:02d} | Train Loss {TRAIN_LOSS:.4f} | Val Loss {VAL_LOSS:.4f} | Val F1-Sent {VAL_F1_SENT:.4f} | Val F1-Len {VAL_F1_LEN:.4f}")
        if VAL_F1_SENT > BEST_VAL_F1:
            BEST_VAL_F1 = VAL_F1_SENT
            torch.save(MODEL.state_dict(), MODEL_DIR / "encoder_best.pt")

    MODEL.load_state_dict(torch.load(MODEL_DIR / "encoder_best.pt", map_location=DEVICE))
    FIG, AXES = plt.subplots(1, 3, figsize=(15, 4))
    AXES[0].plot(HISTORY["train_loss"], label="Train"); AXES[0].plot(HISTORY["val_loss"], label="Val"); AXES[0].set_title("Loss"); AXES[0].legend()
    AXES[1].plot(HISTORY["train_f1_sent"], label="Train"); AXES[1].plot(HISTORY["val_f1_sent"], label="Val"); AXES[1].set_title("F1 - Sentiment"); AXES[1].legend()
    AXES[2].plot(HISTORY["train_f1_len"], label="Train"); AXES[2].plot(HISTORY["val_f1_len"], label="Val"); AXES[2].set_title("F1 - Length Label"); AXES[2].legend()
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "encoder_curves.png", dpi=150); plt.close(FIG)

    ALL_SENT_PRED, ALL_SENT_TRUE, ALL_LEN_PRED, ALL_LEN_TRUE = [], [], [], []
    MODEL.eval()
    with torch.no_grad():
        for TOKEN_IDS, SENTIMENT_IDS, LENGTH_IDS in TEST_LOADER:
            SENT_LOGITS, LEN_LOGITS, _ = MODEL(TOKEN_IDS.to(DEVICE))
            ALL_SENT_PRED.extend(SENT_LOGITS.argmax(-1).cpu().tolist())
            ALL_SENT_TRUE.extend(SENTIMENT_IDS.tolist())
            ALL_LEN_PRED.extend(LEN_LOGITS.argmax(-1).cpu().tolist())
            ALL_LEN_TRUE.extend(LENGTH_IDS.tolist())
    print(classification_report(ALL_SENT_TRUE, ALL_SENT_PRED, target_names=["Negative", "Neutral", "Positive"]))
    print(classification_report(ALL_LEN_TRUE, ALL_LEN_PRED, target_names=["Short", "Medium", "Long"]))

    FULL_LOADER = DataLoader(TRAIN_DS, batch_size=256, shuffle=False, num_workers=0, pin_memory=False)
    EMBEDDING_BATCHES = []
    with torch.no_grad():
        for TOKEN_IDS, _, _ in FULL_LOADER:
            _, _, EMB = MODEL(TOKEN_IDS.to(DEVICE))
            EMBEDDING_BATCHES.append(EMB.cpu().numpy())
    np.save(RESULTS_DIR / "train_embeddings.npy", np.concatenate(EMBEDDING_BATCHES, axis=0))


if __name__ == "__main__":
    MAIN()

