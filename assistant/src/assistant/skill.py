import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StaticSkill:
    """A frozen Trace2Skill artifact injected as one system message."""

    name: str
    system: str
    path: str

    @classmethod
    def load(cls, path: str | Path) -> "StaticSkill":
        skill_path = Path(path)
        content = skill_path.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(f"{path} contains an empty Trace2Skill artifact")
        return cls(name="trace2skill", system=content, path=str(skill_path))

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.system.encode()).hexdigest()
