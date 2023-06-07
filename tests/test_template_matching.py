import pytest
import cv2

import tissue_detection


@pytest.fixture(scope="session")
def img():
    return cv2.imread(
        tissue_detection.examples.files["A13_firstframe"].as_posix(),
        cv2.IMREAD_GRAYSCALE,
    )


def test_template_matching_A13_TPE1(img):
    template = tissue_detection.tpe.TPE1()
    result = template.match(img)
    assert result is not None
    # TODO: assert something


def test_template_matching_A13_TPE2(img):
    template = tissue_detection.tpe.TPE2()
    result = template.match(img)
    assert result is not None
    # TODO: assert something
