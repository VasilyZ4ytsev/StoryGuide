from dataclasses import dataclass, field
from typing import List


@dataclass
class MediaEntity:
    """Сущность для графа знаний (фильм или книга)."""

    name: str
    attributes: List[str] = field(default_factory=list)
    value: float = 0.0

    def __str__(self) -> str:
        return f"{self.name} ({', '.join(self.attributes)})"
