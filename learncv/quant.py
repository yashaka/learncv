import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans


def quantize(
    img: NDArray[np.int8],
    colors: NDArray[np.int8] | int,
    dithering=True,
) -> NDArray[np.int8]:
    colors = (
        colors
        if not isinstance(colors, int)
        else KMeans(n_clusters=16).fit(np.reshape(img, (-1, 1))).cluster_centers_
    )
    img = np.copy(img).astype(np.float32) if dithering else img.astype(np.float32)
    rows, cols, _ = img.shape

    quantized = np.zeros_like(img)

    # TODO: refactor from nested for to more optimized numpy code
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            # Extract the original pixel value
            some_pixel = img[r, c, :]
            # Find the closest colour from the pallette (using absolute value/Euclidean distance)
            new_pixel = colors[
                np.argmin(
                    np.linalg.norm(
                        colors - some_pixel,  # differences
                        axis=1,  # Compute the norm for each row vector ⬇️
                        ord=None,  # == default == Frobenius norm => Euclidean distance for vectors
                    )
                )
            ]

            if dithering:
                # Compute quantization error
                quant_error = some_pixel - new_pixel
                # Diffuse the quantization error accroding to the FS diffusion matrix
                #      [0,  0,  0]
                # 1/16 [0,  ▪️,  7]
                #      [3,  5,  1]
                img[r, c + 1, :] += quant_error * 7 / 16
                img[r + 1, c - 1, :] += quant_error * 3 / 16
                img[r + 1, c, :] += quant_error * 5 / 16
                img[r + 1, c + 1, :] += quant_error * 1 / 16

            # Apply dithering
            quantized[r, c, :] = new_pixel

    return quantized.astype(np.uint8)


ize = quantize
"""An alias for `quantize` to be used as quant.ize on explicit module import."""
