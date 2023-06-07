import cv2
import numpy as np

from .template import Template
from .templates import files as templates


class TPE1(Template):
    def __init__(self):
        self.template = cv2.imread(
            templates["tpe1_template"].as_posix(), cv2.IMREAD_GRAYSCALE
        )
        self.padding = 50
        self.mask_lb = 0
        self.mask_ub = 60

    @property
    def mask(self):
        mask = cv2.inRange(self.padded_template, self.mask_lb, self.mask_ub)

        # remove pillar features so can be used with 3 pillar design
        mask[170:220, 140:220] = 0
        mask[470:520, 140:220] = 0

        mask[170:220, 285:360] = 0
        mask[470:520, 285:360] = 0

        mask[170:220, 430:510] = 0
        mask[470:520, 430:510] = 0

        mask[180:215, 570:655] = 0
        mask[470:520, 575:655] = 0
        cv2.floodFill(mask, None, (170, 300), 100)
        cv2.floodFill(mask, None, (320, 300), 125)
        cv2.floodFill(mask, None, (470, 300), 150)
        cv2.floodFill(mask, None, (620, 300), 175)
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

    def get_cropped_mask(self, img: np.ndarray, res: np.ndarray) -> np.ndarray:
        h, w = img.shape
        x, y = cv2.minMaxLoc(res)[-2]
        return self.mask[y : y + h, x : x + w]
