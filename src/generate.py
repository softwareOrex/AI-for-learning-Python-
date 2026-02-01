from __future__ import annotations
import torch
from src.spm_tokenizer import SPMTokenizer
from src.model import GPT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPT = "checkpoints/boran0_instruct.pt"  # если нет — поставь boran0_base.pt
SP_MODEL = "tokenizer/boran.model"

def main():
    tok = SPMTokenizer(SP_MODEL)

    ck = torch.load(CKPT, map_location=DEVICE)
    cfg = ck["cfg"]

    model = GPT(
        vocab_size=cfg["vocab_size"],
        block_size=cfg["block_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        dropout=0.0,
    ).to(DEVICE)
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()

    prompt = (
        "Ты BORAN.\n"
        "Отвечай на русском языке.\n"
        "Кратко. По шагам. Без Markdown.\n\n"
        "Вопрос:\n"
        "Объясни что такое процент.\n\n"
        "Ответ:\n"
    )

    ids = [tok.bos_id()] + tok.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    out = model.generate(x, max_new_tokens=140, temperature=0.8, top_k=50)
    text = tok.decode(out[0].tolist())
    print(text)

if __name__ == "__main__":
    main()
