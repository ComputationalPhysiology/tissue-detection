from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple, Tuple
from pathlib import Path
from enum import IntEnum

import cv2
import numpy as np


def to_uint8(img):
    if img.dtype == np.uint8:
        return img
    elif img.dtype == np.uint16:
        return cv2.convertScaleAbs(img, alpha=(255.0 / 65535.0))
    else:
        msg = f"Image has datatype {img.dtype}. Please convert to uint8"
        raise ValueError(msg)


def scale_image(
    img: np.ndarray, scale: float, interpolation=cv2.INTER_CUBIC
) -> np.ndarray:
    if np.isclose(scale, 1.0):
        return img
    w, h = img.shape
    width = int(w * scale)
    height = int(h * scale)
    return cv2.resize(img, (height, width), interpolation=interpolation)


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

    @property
    def mask(self) -> np.ndarray:
        return self.template

    @classmethod
    def from_file(cls, fname: Path | str) -> "Template":
        return cls(cv2.imread(Path(fname).as_posix(), cv2.IMREAD_GRAYSCALE))

    def create_result(self, img: np.ndarray) -> np.ndarray:
        return img

    def get_template_location(self, res: np.ndarray) -> Tuple[int, int]:
        return cv2.minMaxLoc(res)[-1]

    def get_cropped_mask(
        self, img: np.ndarray, res: np.ndarray, padding: int
    ) -> np.ndarray:
        h, w = self.mask.shape

        x, y = self.get_template_location(res)
        x -= padding
        y -= padding
        mask = np.zeros_like(img[padding:-padding, padding:-padding])
        H, W = mask.shape

        y0 = max(y, 0)
        sy = abs(min(y, 0))
        y1 = min(y0 + h - sy, H)
        dy = y1 - y0

        x0 = max(x, 0)
        sx = abs(min(x, 0))
        x1 = min(x0 + w - sx, W)
        dx = x1 - x0

        mask[y0:y1, x0:x1] = self.mask[sy : sy + dy, sx : sx + dx]
        return mask

    def match(
        self,
        img: np.ndarray,
        method: MatchMethods = MatchMethods.CCOEFF,
        padding: int = 50,
        scale: float = 1.0,
        invert: bool = False,
    ) -> TemplateMatchResult:
        img = to_uint8(img)
        if invert:
            img = 255 - img
        if scale != 1.0:
            img = scale_image(img, 1 / scale)
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
        if scale != 1.0:
            final_mask = scale_image(
                final_mask, scale=scale, interpolation=cv2.INTER_NEAREST
            )

        return TemplateMatchResult(
            result=final_mask,
            template=self.template,
            template_mask=self.mask,
            match_result=res,
        )
