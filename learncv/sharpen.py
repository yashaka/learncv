import cv2
import numpy as np


def USM(img, radius=3, amount=1.0):
    """Apply unsharp mask to the image

    Args:
        img (np.uint8): input image (0-255)
        radius (int): Radius affects the size of the edges to be enhanced
            or how wide the edge rims become, so a smaller radius enhances
            smaller-scale detail. Higher radius values can cause halos
            at the edges, a detectable faint light rim around objects.
            Fine detail needs a smaller radius.
        amount (float): controls the magnitude of each overshoot (how much
            darker and how much lighter the edge borders become).
            This can also be thought of as how much contrast is added at
            the edges. It does not affect the width of the edge rims.

    Returns:
        np.uint8: Image with unsharp mask applied.

    Radius and amount interact; reducing one allows more of the other.

    Generally a radius of 0.5 to 2 pixels and an amount of 0.5–1.5
    is recommended.

    Unsharp masking may also be used with a large radius and a small amount
    (such as 30–100 radius and 0.05–0.2 amount), which yields increased
    local contrast, a technique termed local contrast enhancement.

    Documented based on https://en.wikipedia.org/wiki/Unsharp_masking.
    """
    img = img.astype(np.float32)

    unsharp = cv2.GaussianBlur(
        img,
        ksize=(radius, radius),
        sigmaX=0,
        sigmaY=0,
    )
    high_freqs = img - unsharp

    return np.clip(
        img + high_freqs * amount,
        0,
        255,
    ).astype(np.uint8)
