import numpy as np
import cv2
from .threading import thread_last, map_with


def by_white_patch(img, at_row=0, at_col=0):
    return thread_last(
        cv2.split(img),
        map_with(
            lambda ch: np.clip(ch * (255 / ch[at_row, at_col]), 0, 255).astype(np.uint8)
        ),
        cv2.merge,
    )


def by_white_patch_to_float(img, at_row=0, at_col=0):
    return thread_last(
        cv2.split(img),
        map_with(lambda ch: np.clip(ch / ch[at_row, at_col], 0, 1)),
        cv2.merge,
    )


def by_gray_world(img):
    R, G, B = cv2.split(img)
    mean_r = np.mean(R)
    mean_g = np.mean(G)
    mean_b = np.mean(B)

    max_mean = np.max([mean_r, mean_g, mean_b])
    kr = max_mean / mean_r
    kg = max_mean / mean_g
    kb = max_mean / mean_b

    balanced = np.zeros_like(img)
    balanced[..., 0] = np.clip(R * kr, 0, 255)
    balanced[..., 1] = np.clip(G * kg, 0, 255)
    balanced[..., 2] = np.clip(B * kb, 0, 255)

    return balanced


def by_max_scale(img):
    return thread_last(
        cv2.split(img),
        map_with(lambda ch: (ch * (255 / np.max(ch))).astype(np.uint8)),
        cv2.merge,
    )
