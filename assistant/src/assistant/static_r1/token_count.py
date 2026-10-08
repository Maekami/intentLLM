"""Optional pinned tokenizer for delivered text; never substitute private usage."""
import os
from functools import lru_cache
from pathlib import Path

@lru_cache(maxsize=4)
def _load(path):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(path)
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer

def configured_counter():
    path = os.getenv('R1_VISIBLE_TOKENIZER')
    if not path:
        return None
    tokenizer = _load(str(Path(path).expanduser().resolve()))
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)
