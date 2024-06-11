import cv2
import math
from typing import Iterable


def polar_lines(img, lines, color=255, thickness=1, lineType=cv2.LINE_AA, length=2000):
    """Draws lines in polar coordinates on the image.

    Args:
        img: Image to draw lines on.
        lines: Iterable of lines in polar coordinates.
            Each line is an iterable of two elements: rho and theta.
        color: Color of the lines.
        thickness: Thickness of the lines.
        lineType: Type of the lines.
        length: Length of the lines.
    """
    step = length // 2
    for line in lines:
        rho, theta = line

        # to convert polar coordinates to Cartesian coordinates...
        # given:
        cos = math.cos(theta)  # = x / rho
        sin = math.sin(theta)  # = y / rho

        # then x0, y0 of point on the line pointed by rho vector with theta angle:
        x0 = cos * rho
        y0 = sin * rho

        # if we go step along the line, then this step is hypotenuse
        # then delta for x is opposite leg, i.e. (its opposite angle sin) * step
        # and delta for y is adjacent leg, i.e. (its adjacent angle cos) * step
        pt1 = (int(x0 + step * (-sin)), int(y0 + step * (cos)))
        pt2 = (int(x0 - step * (-sin)), int(y0 - step * (cos)))

        cv2.line(img, pt1, pt2, color, thickness, lineType)

    return img
