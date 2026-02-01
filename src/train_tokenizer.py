from __future__ import annotations
from pathlib import Path
import sentencepiece as spm

CORPUS = Path("data/corpus.txt")
OUT_DIR = Path("tokenizer")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PREFIX = str(OUT_DIR / "boran")
VOCAB_SIZE = 16000

def main():
    if not CORPUS.exists():
        raise FileNotFoundError("Создай data/corpus.txt (открытый текст)")

    spm.SentencePieceTrainer.train(
        input=str(CORPUS),
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        character_coverage=0.9995,
        model_type="bpe",
        bos_id=1,
        eos_id=2,
        pad_id=0,
        unk_id=3,
        normalization_rule_name="identity",  # FIX для ошибки nmt_nfkc на Windows
    )

    print("Saved:", MODEL_PREFIX + ".model")

if __name__ == "__main__":
    main()
