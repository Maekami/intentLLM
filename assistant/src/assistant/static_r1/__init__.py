from .session import R1Session, R1Baseline
from .agent import fingerprint
from . import prompts

__all__ = ['R1Session', 'R1Baseline', 'snapshot']

def snapshot():
    import hashlib
    return {
        'architecture': 'A3_R1',
        'source_sha256': fingerprint(),
        'original_snapshot_sha256': 'fa09d6ccac3de47568bd4b0c69de48c82439abc222a675785c2495c8fbb97744',
        'prompts': {name: {'system': getattr(prompts, name),
                          'sha256': hashlib.sha256(getattr(prompts, name).encode()).hexdigest()}
                    for name in ('TRACKER', 'DIRECT', 'INTRA', 'INTER', 'EDITOR')},
    }
