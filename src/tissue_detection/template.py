from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple
from pathlib import Path
from enum import IntEnum

import cv2
import numpy as np


class MatchMethods(IntEnum):
    SQDIFF = cv2.TM_SQDIFF
    SQDIFF_NORMED = cv2.TM_SQDIFF_NORMED
    CCORR = cv2.TM_CCORR
    CCORR_NORMED = cv2.TM_CCORR_NORMED
    CCOEFF = cv2.TM_CCOEFF
    CCOEFF_NORMED = cv2.TM_CCOEFF_NORMED


class TemplateMatchResult(NamedTuple):
    result: np.ndarray
    template: np.ndarray
    template_mask: np.ndarray
    match_result: np.ndarray


@dataclass
class Template:
    template: np.ndarray
    padding: int = 0
    scale: float = 1.0

    def __post_init__(self):
        if self.scale != 1.0:
            self.template_original = self.template
            w, h = self.template.shape
            width = int(w * self.scale)
            height = int(h * self.scale)
            self.template = cv2.resize(self.template, (height, width))
            self.padding = int(self.padding * self.scale)

    @property
    def mask(self) -> np.ndarray:
        return self.template

    @classmethod
    def from_file(cls, fname: Path | str) -> "Template":
        return cls(cv2.imread(Path(fname).as_posix(), cv2.IMREAD_GRAYSCALE))

    def create_result(self, img: np.ndarray) -> np.ndarray:
        return img

    def get_cropped_mask(
        self, img: np.ndarray, res: np.ndarray, padding: int
    ) -> np.ndarray:
        h, w = self.mask.shape

        x, y = cv2.minMaxLoc(res)[-1]
        x -= padding
        y -= padding
        mask = np.zeros_like(img[padding:-padding, padding:-padding])

        y0 = max(y, 0)
        sy = abs(min(y, 0))
        y1 = y0 + h - sy
        dy = y1 - y0

        x0 = max(x, 0)
        sx = abs(min(x, 0))
        x1 = x0 + w - sx
        dx = x1 - x0

        mask[y0:y1, x0:x1] = self.mask[sy : sy + dy, sx : sx + dx]
        return mask

    def match(
        self,
        img: np.ndarray,
        method: MatchMethods = MatchMethods.CCOEFF,
        padding: int = 50,
    ) -> TemplateMatchResult:
        # Add some padding to make more room for template to move
        padding = max(padding, 0)
        if padding > 0:
            img = cv2.copyMakeBorder(
                img,
                padding,
                padding,
                padding,
                padding,
                cv2.BORDER_REPLICATE,
            )
        res = cv2.matchTemplate(img, self.template, method)
        mask_cropped = self.get_cropped_mask(img=img, res=res, padding=padding)
        final_mask = self.create_result(img=mask_cropped)

        return TemplateMatchResult(
            result=final_mask,
            template=self.template,
            template_mask=self.mask,
            match_result=res,
        )
