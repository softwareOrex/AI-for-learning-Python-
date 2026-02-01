from __future__ import annotations
import sentencepiece as spm

class SPMTokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)

    def bos_id(self) -> int:
        return int(self.sp.bos_id())

    def eos_id(self) -> int:
        return int(self.sp.eos_id())

    def pad_id(self) -> int:
        return int(self.sp.pad_id())

    def unk_id(self) -> int:
        return int(self.sp.unk_id())
