import pytest
import cv2
import numpy as np

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

    # Check location of tissue 1
    x1, y1 = np.where(result.result == 1)
    cy1, cx1 = template.center_tissue1
    assert np.isclose(x1.mean(), cx1, atol=1.0)
    assert np.isclose(y1.mean(), cy1, atol=1.0)
    assert np.isclose(y1.min(), cy1 - 55, atol=1.0)
    assert np.isclose(y1.max(), cy1 + 55, atol=1.0)

    x2, y2 = np.where(result.result == 2)
    cy2, cx2 = template.center_tissue2
    assert np.isclose(x2.mean(), cx2, atol=1.0)
    assert np.isclose(y2.mean(), cy2, atol=1.0)
    assert np.isclose(y2.min(), cy2 - 55, atol=1.0)
    assert np.isclose(y2.max(), cy2 + 55, atol=1.0)

    x3, y3 = np.where(result.result == 3)
    cy3, cx3 = template.center_tissue3
    assert np.isclose(x3.mean(), cx3, atol=1.0)
    assert np.isclose(y3.mean(), cy3, atol=1.0)
    assert np.isclose(y3.min(), cy3 - 55, atol=1.0)
    assert np.isclose(y3.max(), cy3 + 55, atol=1.0)

    x4, y4 = np.where(result.result == 4)
    cy4, cx4 = template.center_tissue4
    assert np.isclose(x4.mean(), cx4, atol=1.0)
    assert np.isclose(y4.mean(), cy4, atol=1.0)
    assert np.isclose(y4.min(), cy4 - 55, atol=1.0)
    assert np.isclose(y4.max(), cy4 + 55, atol=1.0)


@pytest.mark.parametrize("scale", (1.0, 0.8, 0.5, 0.3))
def test_template_matching_A13_TPE1_scale(img, scale):
    atol = 2.0
    img = tissue_detection.template.scale_image(img, scale=scale)

    template = tissue_detection.tpe.TPE1()
    result = template.match(img, scale=scale)

    # Check location of tissue 1
    x1, y1 = np.where(result.result == 1)
    cy1, cx1 = template.center_tissue1
    assert np.isclose(x1.mean(), scale * cx1, atol=atol)
    assert np.isclose(y1.mean(), scale * cy1, atol=atol)
    assert np.isclose(y1.min(), int(scale * (cy1 - 55)), atol=atol)
    assert np.isclose(y1.max(), int(scale * (cy1 + 55)), atol=atol)

    x2, y2 = np.where(result.result == 2)
    cy2, cx2 = template.center_tissue2
    assert np.isclose(x2.mean(), scale * cx2, atol=atol)
    assert np.isclose(y2.mean(), scale * cy2, atol=atol)
    assert np.isclose(y2.min(), int(scale * (cy2 - 55)), atol=atol)
    assert np.isclose(y2.max(), int(scale * (cy2 + 55)), atol=atol)

    x3, y3 = np.where(result.result == 3)
    cy3, cx3 = template.center_tissue3
    assert np.isclose(x3.mean(), scale * cx3, atol=atol)
    assert np.isclose(y3.mean(), scale * cy3, atol=atol)
    assert np.isclose(y3.min(), int(scale * (cy3 - 55)), atol=atol)
    assert np.isclose(y3.max(), int(scale * (cy3 + 55)), atol=atol)

    x4, y4 = np.where(result.result == 4)
    cy4, cx4 = template.center_tissue4
    assert np.isclose(x4.mean(), scale * cx4, atol=atol)
    assert np.isclose(y4.mean(), scale * cy4, atol=atol)
    assert np.isclose(y4.min(), int(scale * (cy4 - 55)), atol=atol)
    assert np.isclose(y4.max(), int(scale * (cy4 + 55)), atol=atol)
