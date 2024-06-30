from typing import Tuple
import cv2

from learncv import resize
from .threading import thread_first


def RGB(path, by: Tuple[int, int] | None = None):
    """Read an image from a file and return it in RGB format,
    resizing with kept aspect ratio it if needed

    Args:
        path (str): path to the image file
        by (Tuple[int, int]): desired width and height of the image
    """
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return (
        img if by is None else resize.with_aspect_ratio(img, width=by[0], height=by[1])
    )


def RGBs(path):
    return cv2.split(RGB(path))


def gray(path, by: Tuple[int, int] | None = None):
    """Read an image from a file and return it in grayscale format,
    resizing with kept aspect ratio it if needed

    Args:
        path (str): path to the image file
        by (Tuple[int, int]): desired width and height of the image
    """
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return (
        img if by is None else resize.with_aspect_ratio(img, width=by[0], height=by[1])
    )
