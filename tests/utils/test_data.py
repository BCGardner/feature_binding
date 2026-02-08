import numpy as np

from hsnn.utils.data import ImageLoader, ImageSet
from hsnn.utils.io import DATA_DIR


_IMAGE_DIR = DATA_DIR / 'n3p2'


def test_imageloader():
    image_loader = ImageLoader()
    for image_path in _IMAGE_DIR.glob('*.png'):
        image = image_loader.load(image_path)
        assert isinstance(image, np.ndarray)


def test_imageset():
    image_set = ImageSet(_IMAGE_DIR)
    for image in image_set:
        assert isinstance(image, np.ndarray)
