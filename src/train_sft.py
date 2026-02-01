from __future__ import annotations
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.spm_tokenizer import SPMTokenizer
from src.model import GPT
from src.data_sft import SFTDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_CKPT = "checkpoints/boran0_base.pt"
SP_MODEL = "tokenizer/boran.model"
SFT_JSONL = "data/sft.jsonl"
OUT = "checkpoints/boran0_instruct.pt"

BLOCK_SIZE = 256
BATCH_SIZE = 8
GRAD_ACCUM = 4
LR = 1e-4
EPOCHS = 2

def main():
    os.makedirs("checkpoints", exist_ok=True)
    if not os.path.exists(BASE_CKPT):
        raise FileNotFoundError("Сначала обучи base: python -m src.train_base")
    if not os.path.exists(SFT_JSONL):
        raise FileNotFoundError("Создай data/sft.jsonl или сделай export: python -m src.export_sft_from_sqlite")

    tok = SPMTokenizer(SP_MODEL)

    ck = torch.load(BASE_CKPT, map_location=DEVICE)
    cfg = ck["cfg"]

    model = GPT(
        vocab_size=cfg["vocab_size"],
        block_size=cfg["block_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        dropout=cfg["dropout"],
    ).to(DEVICE)
    model.load_state_dict(ck["model_state"], strict=True)

    ds = SFTDataset(SFT_JSONL, tok, BLOCK_SIZE)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    model.train()
    step = 0

    for ep in range(EPOCHS):
        pbar = tqdm(dl, desc=f"sft epoch {ep+1}/{EPOCHS}")
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
                "cfg": cfg,
            },
            OUT,
        )
        print("Saved:", OUT)

if __name__ == "__main__":
    main()
