from typing import Tuple
import cv2
import numpy as np

from .template import Template
from .templates import files as templates


class TPE1(Template):
    def __init__(self, scale: float = 1.0, padding: int = 0) -> None:
        template = cv2.imread(
            templates["tpe1_template"].as_posix(), cv2.IMREAD_GRAYSCALE
        )
        self.template = template[80:520, 50:640]
        self.padding = padding
        self.mask_lb = 0
        self.mask_ub = 60
        self.scale = scale
        self.__post_init__()

    @property
    def mask(self):
        mask = cv2.inRange(self.template, self.mask_lb, self.mask_ub)

        # Distance between tissues
        wx = 150
        # Center for tissue1
        cx1, cy1 = 70, 215
        # Center for tissue2
        cx2, cy2 = cx1 + wx, 215
        # Center for tissue3
        cx3, cy3 = cx1 + 2 * wx, 215
        # Center for tissue4
        cx4, cy4 = cx1 + 3 * wx, 215

        p = self.padding // 2
        px1, py1 = 56 + p, 74 + p
        px2, py2 = 94 + p, 74 + p
        px3, py3 = 56 + p, 370 + p
        px4, py4 = 94 + p, 370 + p

        r = 20
        for i in range(4):
            mask[py1 - r : py1 + r, wx * i + px1 - r : wx * i + px1 + r] = 0
            mask[py2 - r : py2 + r, wx * i + px2 - r : wx * i + px2 + r] = 0
            mask[py3 - r : py3 + r, wx * i + px3 - r : wx * i + px3 + r] = 0
            mask[py4 - r : py4 + r, wx * i + px4 - r : wx * i + px4 + r] = 0

        s1 = (cx1, cy1)
        s2 = (cx2, cy2)
        s3 = (cx3, cy3)
        s4 = (cx4, cy4)

        cv2.floodFill(mask, None, s1, 100)
        cv2.floodFill(mask, None, s2, 125)
        cv2.floodFill(mask, None, s3, 150)
        cv2.floodFill(mask, None, s4, 175)

        return mask

    def create_result(self, img: np.ndarray):
        mask1 = cv2.inRange(img, 90, 110).astype(bool)
        mask2 = cv2.inRange(img, 120, 130).astype(bool)
        mask3 = cv2.inRange(img, 145, 155).astype(bool)
        mask4 = cv2.inRange(img, 170, 180).astype(bool)

        final_mask = np.zeros_like(img)
        final_mask[mask1] = 1
        final_mask[mask2] = 2
        final_mask[mask3] = 3
        final_mask[mask4] = 4
        return final_mask


class TPE2(TPE1):
    def __init__(self):
        self.template = cv2.imread(
            templates["tpe1_outline"].as_posix(), cv2.IMREAD_GRAYSCALE
        )
        self.padding = 50
        self.mask_lb = 100
        self.mask_ub = 255

    def get_template_location(self, res: np.ndarray) -> Tuple[int, int]:
        return cv2.minMaxLoc(res)[-2]
