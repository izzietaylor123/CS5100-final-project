"""
Handles one image at a time. Reads a staff image from as grayscale, resizes it to a fixed height of 
64 pixels (keeping the proportions so symbols aren't stretched), scales the pixel values to 0-1, and 
inverts them so ink is 1 and background is 0. The output is a NumPy array.
"""

import cv2
import numpy as np

# every image is resized to 64 pixels tall.
# small enough to train quickly and large enough to see lines, note heads, and stems.
IMG_HEIGHT = 64


def load_image(path):
    """Loads an image as grayscale, resize to fixed height, normalize values."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")

    # resize to fixed height
    height, width = img.shape
    new_width = max(1, int(width * IMG_HEIGHT / height))
    img = cv2.resize(img, (new_width, IMG_HEIGHT),
                     interpolation=cv2.INTER_AREA)

    # ink = 1 and background = 0 in images
    img = img.astype(np.float32) / 255.0
    img = 1.0 - img

    return img
