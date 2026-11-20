
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.model import MaxPoolingLayer


# --------------------------------------------------
# 1. Load the actual PrIMuS sheet-music image
# --------------------------------------------------

image_path = (
    "package_aa/000051650-1_1_1/"
    "000051650-1_1_1.png"
)

image = Image.open(image_path).convert("L")

print("Image format:", image.format)
print("Image dimensions:", image.size)


# --------------------------------------------------
# 2. Convert image into a NumPy array
# --------------------------------------------------

gray_image = np.array(image, dtype=np.float32)


# --------------------------------------------------
# 3. Invert pixel values
# --------------------------------------------------
# Original: black = 0, white = 255
# Inverted: black = 255, white = 0
#
# This allows max pooling to preserve dark notation.

ink_image = 255 - gray_image


# --------------------------------------------------
# 4. Convert image into a PyTorch tensor
# --------------------------------------------------

tensor = torch.tensor(ink_image).unsqueeze(0).unsqueeze(0)

# Tensor shape: (Batch, Channels, Height, Width)

print("\nOriginal tensor shape:", tensor.shape)


# --------------------------------------------------
# 5. Apply two different pooling configurations
# --------------------------------------------------

pool_2 = MaxPoolingLayer(kernel_size=2, stride=2)
pool_4 = MaxPoolingLayer(kernel_size=4, stride=4)

pooled_2 = pool_2(tensor)
pooled_4 = pool_4(tensor)


# --------------------------------------------------
# 6. Print dimensions for comparison
# --------------------------------------------------

print("\n--- Pooling Comparison ---")

print("Original image:", tensor.shape)
print("2x2 pooling:", pooled_2.shape)
print("4x4 pooling:", pooled_4.shape)


# --------------------------------------------------
# 7. Convert pooled tensors back to NumPy
# --------------------------------------------------

original_image = gray_image

pooled_2_image = pooled_2.squeeze().numpy()
pooled_4_image = pooled_4.squeeze().numpy()


# --------------------------------------------------
# 8. Visualize all three images
# --------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Original image
axes[0].imshow(
    original_image,
    cmap="gray",
    vmin=0,
    vmax=255
)
axes[0].set_title("Original Sheet Music")
axes[0].axis("off")

# 2x2 pooling
axes[1].imshow(
    pooled_2_image,
    cmap="gray_r",
    vmin=0,
    vmax=255
)
axes[1].set_title("After 2x2 Max Pooling")
axes[1].axis("off")

# 4x4 pooling
axes[2].imshow(
    pooled_4_image,
    cmap="gray_r",
    vmin=0,
    vmax=255
)
axes[2].set_title("After 4x4 Max Pooling")
axes[2].axis("off")


plt.tight_layout()
plt.show()
