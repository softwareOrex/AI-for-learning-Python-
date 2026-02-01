from __future__ import annotations
import json
import torch
from torch.utils.data import Dataset

# Формат строки:
# {"instruction":"...","output":"..."}
# Мы учим модель предсказывать только часть "output" (instruction маскируем)

PROMPT_TEMPLATE = (
    "Ты BORAN.\n"
    "Отвечай на языке пользователя.\n"
    "Кратко. По шагам. Без Markdown.\n\n"
    "Вопрос:\n{instruction}\n\n"
    "Ответ:\n"
)

class SFTDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, block_size: int):
        self.rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                ins = (obj.get("instruction") or "").strip()
                out = (obj.get("output") or "").strip()
                if len(ins) < 5 or len(out) < 5:
                    continue
                self.rows.append((ins, out))

        self.tok = tokenizer
        self.block = block_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        instruction, output = self.rows[i]

        prompt = PROMPT_TEMPLATE.format(instruction=instruction)
        prompt_ids = self.tok.encode(prompt)
        out_ids = self.tok.encode(output)

        bos = self.tok.bos_id()
        eos = self.tok.eos_id()

        ids = [bos] + prompt_ids + out_ids + [eos]
        ids = ids[: self.block]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)

        # Маска: все токены prompt не считаем в loss
        # Определяем границу: bos + prompt_ids
        prompt_len = min(1 + len(prompt_ids), len(ids) - 1)
        y_masked = y.clone()
        if prompt_len > 0:
            y_masked[:prompt_len] = -100  # ignore_index

        # Если слишком коротко после обрезки — всё равно вернём
        return x, y_masked
