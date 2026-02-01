from __future__ import annotations
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.spm_tokenizer import SPMTokenizer
from src.model import GPT
from src.data_pretrain import PretrainDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CORPUS = "data/corpus.txt"
SP_MODEL = "tokenizer/boran.model"
OUT = "checkpoints/boran0_base.pt"

# безопасно для RTX 3050
BLOCK_SIZE = 256
BATCH_SIZE = 8          # если VRAM 6GB можно 16
GRAD_ACCUM = 4          # если VRAM мало -> 8
LR = 3e-4
EPOCHS = 2
DROPOUT = 0.1

def main():
    os.makedirs("checkpoints", exist_ok=True)

    tok = SPMTokenizer(SP_MODEL)
    with open(CORPUS, "r", encoding="utf-8") as f:
        text = f.read()

    ids = tok.encode(text)
    ds = PretrainDataset(ids, BLOCK_SIZE)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = GPT(
        vocab_size=tok.vocab_size,
        block_size=BLOCK_SIZE,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=DROPOUT,
    ).to(DEVICE)

    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    model.train()
    step = 0

    for ep in range(EPOCHS):
        pbar = tqdm(dl, desc=f"base epoch {ep+1}/{EPOCHS}")
        optim.zero_grad(set_to_none=True)

        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                _, loss = model(x, y)
                loss = loss / GRAD_ACCUM

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            pbar.set_postfix(loss=float(loss.item()) * GRAD_ACCUM)
            step += 1

        torch.save(
            {
                "model_state": model.state_dict(),
                "sp_model": SP_MODEL,
                "cfg": {
                    "block_size": BLOCK_SIZE,
                    "n_layer": 6,
                    "n_head": 6,
                    "n_embd": 384,
                    "dropout": DROPOUT,
                    "vocab_size": tok.vocab_size,
                },
            },
            OUT,
        )
        print("Saved:", OUT)

if __name__ == "__main__":
    main()
