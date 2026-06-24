from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class FaceBox:
    origin_x: int
    origin_y: int
    width: int
    height: int
    score: float | None = None

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "origin_x": self.origin_x,
            "origin_y": self.origin_y,
            "width": self.width,
            "height": self.height,
            "score": self.score,
        }


@dataclass(slots=True)
class FaceAnalysis:
    face_count: int
    landmarks: list[list[list[float]]] = field(default_factory=list)
    detector_boxes: list[FaceBox] = field(default_factory=list)
    facial_transformation_matrixes: list[list[list[float]]] = field(default_factory=list)

    def to_metadata(self) -> dict[str, object]:
        return {
            "face_count": self.face_count,
            "detector_boxes": [box.to_dict() for box in self.detector_boxes],
            "landmark_faces": len(self.landmarks),
            "has_facial_transformation_matrixes": bool(self.facial_transformation_matrixes),
        }
