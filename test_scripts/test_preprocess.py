from src.preprocess import load_image, IMG_HEIGHT
import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

IMAGE_PATH = "data/primus/package_aa/000051650-1_1_1/000051650-1_1_1.png"

img = load_image(IMAGE_PATH)

print("Shape (height, width):", img.shape)
print("Dtype:", img.dtype)
print("Min / max:", img.min(), img.max())
print("Height is", IMG_HEIGHT, "->", img.shape[0] == IMG_HEIGHT)

# Convert back to normal black and white and save
out = ((1.0 - img) * 255).astype(np.uint8)
cv2.imwrite("test_image.png", out)
print("Saved test_image.png")
