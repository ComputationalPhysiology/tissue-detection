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
    padded_img: np.ndarray
    match_result: np.ndarray


@dataclass
class Template:
    template: np.ndarray
    padding: int = 0

    def padded(self, img: np.ndarray) -> np.ndarray:
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

    def create_mask(self, img: np.ndarray) -> np.ndarray:
        return img

    @classmethod
    def from_file(cls, fname: Path | str) -> "Template":
        return cls(cv2.imread(Path(fname).as_posix(), cv2.IMREAD_GRAYSCALE))

    def create_result(self, img: np.ndarray) -> np.ndarray:
        return img

    def get_cropped_mask(
        self, img: np.ndarray, res: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        h, w = img.shape

        x, y = cv2.minMaxLoc(res)[-1]
        return mask[y : y + h, x : x + w]

    def match(
        self,
        img: np.ndarray,
        method: MatchMethods = MatchMethods.CCOEFF,
    ) -> TemplateMatchResult:
        padded = self.padded(img)
        res = cv2.matchTemplate(img, padded, method)

        mask = self.create_mask(padded)
        mask_cropped = self.get_cropped_mask(img=img, res=res, mask=mask)
        final_mask = self.create_result(img=mask_cropped)

        return TemplateMatchResult(
            result=final_mask,
            template=self.template,
            template_mask=mask,
            padded_img=padded,
            match_result=res,
        )
