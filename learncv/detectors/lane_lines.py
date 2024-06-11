import cv2
import numpy as np
from sklearn.cluster import KMeans

from learncv import draw
from learncv.threading import (
    thread_last,
    filter_with,
    setitem,
    thread_first,
)

# TODO: add docstrings


def edges_from(
    img,
    thresholds=(100, 150),
    horizon=350,  # TODO: define automatically if not set
):
    return thread_first(
        img,
        (cv2.cvtColor, cv2.COLOR_RGB2GRAY),
        (cv2.Canny, thresholds[0], thresholds[1]),
        (setitem, slice(0, horizon), 0),
    )


def detect(
    img,
    edges=None,
    edge_thresholds=(100, 150),
    horizon=350,  # TODO: set to None as default, then define automatically by img res.
    resolution=(2, 2),
    acc_threshold=190,
    without_theta=(70, 110),
    clusters: int | None = 6,
):
    rho = resolution[0]
    theta_rads = resolution[1] * np.pi / 180
    out_theta_rads_min = without_theta[0] * np.pi / 180
    out_theta_rads_max = without_theta[1] * np.pi / 180

    return thread_last(
        (edges if edges is not None else edges_from(img, edge_thresholds, horizon)),
        lambda edges: cv2.HoughLines(edges, rho, theta_rads, threshold=acc_threshold),
        lambda lines: lines[:, 0, :],  # Get rid of the unnecessary dimension
        filter_with(  # Filter out lines that are aprox. horizontal
            lambda line: not (out_theta_rads_min < line[1] < out_theta_rads_max)
        ),
        lambda lines: (
            KMeans(n_clusters=clusters).fit(lines).cluster_centers_
            if clusters is not None
            else lines
        ),
        # KMeans(n_clusters=clusters).fit,
        # lambda kmeans: kmeans.cluster_centers_,
        (draw.polar_lines, np.copy(img)),
    )
