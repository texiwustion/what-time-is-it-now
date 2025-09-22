from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Tuple, Union, Protocol

BBox = List[Tuple[float, float]]
ImageLike = Union[str, "np.ndarray", "Image.Image"]

@dataclass
class OCRLine:
    line_id: int
    text: str
    confidence: float
    bbox: BBox
    page_id: int = 1

@dataclass
class OCRResult:
    texts: List[OCRLine]
    avg_confidence: float
    time_ms: int
    raw: Any

class OCRBackend(Protocol):
    def ocr(self, image: Any, cls: bool = True) -> List[List[Tuple[BBox, Tuple[str, float]]]]:
        ...

class OCREngineProto(Protocol):
    def infer(self, image: ImageLike) -> OCRResult:
        ...