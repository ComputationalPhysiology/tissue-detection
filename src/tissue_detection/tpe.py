from typing import Tuple
import cv2
import numpy as np

from .template import Template
from .templates import files as templates


class TPE1(Template):
    def __init__(self, include_pilars: bool = False) -> None:
        template = cv2.imread(
            templates["tpe1_template"].as_posix(), cv2.IMREAD_GRAYSCALE
        )
        self.template = template[80:520, 50:640]
        self.include_pilars = include_pilars

    @property
    def width(self):
        return self.template.shape[1]

    @property
    def height(self):
        return self.template.shape[0]

    @property
    def distance_between_tissues(self) -> int:
        return 147

    @property
    def center_tissue1(self) -> Tuple[int, int]:
        return (57, 223)

    @property
    def center_tissue2(self) -> Tuple[int, int]:
        y, x = self.center_tissue1
        w = self.distance_between_tissues
        return (y + w, x)

    @property
    def center_tissue3(self) -> Tuple[int, int]:
        y, x = self.center_tissue1
        w = self.distance_between_tissues
        return (y + 2 * w, x)

    @property
    def center_tissue4(self) -> Tuple[int, int]:
        y, x = self.center_tissue1
        w = self.distance_between_tissues
        return (y + 3 * w, x)

    @property
    def pilar1(self) -> Tuple[int, int]:
        return (56, 74)

    @property
    def pilar2(self) -> Tuple[int, int]:
        return (94, 74)

    @property
    def pilar3(self) -> Tuple[int, int]:
        return (56, 370)

    @property
    def pilar4(self) -> Tuple[int, int]:
        return (94, 370)

    @property
    def mask(self):
        mask = cv2.inRange(self.template, 0, 60)

        # Position of pilars
        px1, py1 = self.pilar1
        px2, py2 = self.pilar2
        px3, py3 = self.pilar3
        px4, py4 = self.pilar4

        r = 20
        if not self.include_pilars:
            w = self.distance_between_tissues
            for i in range(4):
                mask[py1 - r : py1 + r, w * i + px1 - r : w * i + px1 + r] = 0
                mask[py2 - r : py2 + r, w * i + px2 - r : w * i + px2 + r] = 0
                mask[py3 - r : py3 + r, w * i + px3 - r : w * i + px3 + r] = 0
                mask[py4 - r : py4 + r, w * i + px4 - r : w * i + px4 + r] = 0

        # Remove an artifact from template
        mask[67 - r // 2 : 67 + r // 2, 475 - r // 2 : 475 + r // 2] = 0

        cv2.floodFill(mask, None, self.center_tissue1, 100)
        cv2.floodFill(mask, None, self.center_tissue2, 125)
        cv2.floodFill(mask, None, self.center_tissue3, 150)
        cv2.floodFill(mask, None, self.center_tissue4, 175)

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
