"""Portable R1 artifacts and optional atomic audit writes."""
import json
import os
from pathlib import Path
CODE = Path(__file__).resolve().parent

def output_path(path):
    return Path(path).resolve()

def write_json(path, value):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)
