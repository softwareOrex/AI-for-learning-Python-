from __future__ import annotations
import json
import sqlite3
from pathlib import Path
import re

DB_PATH = Path("../boran.db")  # если твоя БД в другом месте — поправь
OUT = Path("data/sft.jsonl")

def sanitize(s: str) -> str:
    s = s or ""
    s = re.sub(r"\b\d{16}\b", "", s)
    s = re.sub(r"\b\+?\d[\d\s\-]{8,}\b", "", s)
    s = re.sub(r"[\u0000-\u001f]+", " ", s)
    return s.strip()

def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # ожидаем knowledge(question, answer)
    cur.execute("SELECT question, answer FROM knowledge")
    rows = cur.fetchall()
    conn.close()

    OUT.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for q, a in rows:
            q = sanitize(str(q or ""))
            a = sanitize(str(a or ""))
            if len(q) < 8 or len(a) < 20:
                continue
            obj = {"instruction": q, "output": a}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1

    print("SFT saved:", OUT, "rows:", n)

if __name__ == "__main__":
    main()
