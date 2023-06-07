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
    template_padded: np.ndarray
    match_result: np.ndarray


@dataclass
class Template:
    template: np.ndarray
    padding: int = 0

    @property
    def padded_template(self) -> np.ndarray:
        if self.padding == 0:
            return self.template
        else:
            return cv2.copyMakeBorder(
                self.template,
                self.padding,
                self.padding,
                self.padding,
                self.padding,
                cv2.BORDER_REPLICATE,
            )

    @property
    def mask(self) -> np.ndarray:
        return self.padded_template

    @classmethod
    def from_file(cls, fname: Path | str) -> "Template":
        return cls(cv2.imread(Path(fname).as_posix(), cv2.IMREAD_GRAYSCALE))

    def create_result(self, img: np.ndarray) -> np.ndarray:
        return img

    def get_cropped_mask(self, img: np.ndarray, res: np.ndarray) -> np.ndarray:
        h, w = img.shape

        x, y = cv2.minMaxLoc(res)[-1]
        return self.mask[y : y + h, x : x + w]

    def match(
        self,
        img: np.ndarray,
        method: MatchMethods = MatchMethods.CCOEFF,
    ) -> TemplateMatchResult:
        res = cv2.matchTemplate(img, self.padded_template, method)
        mask_cropped = self.get_cropped_mask(img=img, res=res)
        final_mask = self.create_result(img=mask_cropped)

        return TemplateMatchResult(
            result=final_mask,
            template=self.template,
            template_mask=self.mask,
            template_padded=self.padded_template,
            match_result=res,
        )
