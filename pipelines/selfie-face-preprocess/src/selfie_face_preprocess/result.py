from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .image_io import save_png


@dataclass(slots=True)
class FacePreprocessResult:
    """Structured output from `preprocess_selfie`."""

    accepted: bool
    reject_reasons: list[str]
    warnings: list[str]
    quality: dict[str, Any]
    metadata: dict[str, Any]
    model_input: np.ndarray | None = field(default=None, repr=False)
    face_aligned: np.ndarray | None = field(default=None, repr=False)
    face_mask: np.ndarray | None = field(default=None, repr=False)

    def to_metadata(self) -> dict[str, Any]:
        payload = dict(self.metadata)
        payload["accepted"] = self.accepted
        payload["reject_reasons"] = list(self.reject_reasons)
        payload["warnings"] = list(self.warnings)
        payload["quality"] = self.quality
        payload["outputs"] = {
            "model_input": self.model_input is not None,
            "face_aligned": self.face_aligned is not None,
            "face_mask": self.face_mask is not None,
        }
        return _json_ready(payload)

    def save_outputs(self, out_dir: str | Path) -> dict[str, Path]:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        written: dict[str, Path] = {}
        if self.model_input is not None:
            path = out_path / "model_input.png"
            save_png(self.model_input, path)
            written["model_input"] = path
        if self.face_aligned is not None:
            path = out_path / "face_aligned.png"
            save_png(self.face_aligned, path)
            written["face_aligned"] = path
        if self.face_mask is not None:
            path = out_path / "face_mask.png"
            save_png(self.face_mask, path)
            written["face_mask"] = path

        metadata_path = out_path / "metadata.json"
        metadata_path.write_text(
            json.dumps(self.to_metadata(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        written["metadata"] = metadata_path
        return written


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value
