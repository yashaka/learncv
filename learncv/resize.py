import cv2
import numpy as np


# TODO: add padding_color int param
def with_aspect_ratio(image, width, height):
    """Resize image to the desired dimensions while maintaining the aspect ratio
    (without distortions), padding with black, if needed
    """
    (original_height, original_width) = image.shape[:2]
    if original_width == width and original_height == height:
        return image

    # Calculate the aspect ratio
    aspect_ratio_original = original_width / original_height
    aspect_ratio_desired = width / height

    if aspect_ratio_original > aspect_ratio_desired:
        # Resize based on width
        new_width = width
        new_height = int(new_width / aspect_ratio_original)
    else:
        # Resize based on height
        new_height = height
        new_width = int(new_height * aspect_ratio_original)

    resized_image = cv2.resize(image, (new_width, new_height))

    # Create a new image with the desired dimensions and fill it with black
    new_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Calculate the position to paste the resized image
    x_offset = (width - new_width) // 2
    y_offset = (height - new_height) // 2

    # Paste the resized image onto the new image
    new_image[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = (
        resized_image
    )

    return new_image
