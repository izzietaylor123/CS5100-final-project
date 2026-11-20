
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# --------------------------------------------------
# 1. Load the real PrIMuS sheet-music image
# --------------------------------------------------

image_path = (
    "package_aa/000051650-1_1_1/"
    "000051650-1_1_1.png"
)

image = Image.open(image_path).convert("L")

gray_image = np.array(image, dtype=np.float32)


# --------------------------------------------------
# 2. Invert image
# --------------------------------------------------
# Dark notation becomes high values.
# White background becomes zero.

ink_image = 255 - gray_image

tensor = torch.tensor(ink_image).unsqueeze(0).unsqueeze(0)

print("Input shape:", tensor.shape)


# --------------------------------------------------
# 3. Define two convolution filters
# --------------------------------------------------

# Filter 1: Detect horizontal lines
horizontal_filter = torch.tensor([
    [-1., -1., -1.],
    [ 2.,  2.,  2.],
    [-1., -1., -1.]
])

# Filter 2: Detect vertical lines
vertical_filter = torch.tensor([
    [-1.,  2., -1.],
    [-1.,  2., -1.],
    [-1.,  2., -1.]
])

# Combine filters into convolution weights
filters = torch.stack([
    horizontal_filter,
    vertical_filter
]).unsqueeze(1)

print("Filter shape:", filters.shape)


# --------------------------------------------------
# 4. Apply convolution
# --------------------------------------------------

# padding=1 keeps the original height and width
convolution_output = F.conv2d(
    tensor,
    filters,
    padding=1
)

print("Convolution output shape:", convolution_output.shape)


# --------------------------------------------------
# 5. Separate the two feature maps
# --------------------------------------------------

horizontal_features = convolution_output[0, 0].abs().numpy()
vertical_features = convolution_output[0, 1].abs().numpy()


# --------------------------------------------------
# 6. Normalize feature maps for visualization
# --------------------------------------------------

def normalize(feature_map):
    maximum = feature_map.max()

    if maximum == 0:
        return feature_map

    return feature_map / maximum


horizontal_features = normalize(horizontal_features)
vertical_features = normalize(vertical_features)


# --------------------------------------------------
# 7. Visualize the results
# --------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(gray_image, cmap="gray")
axes[0].set_title("Original Sheet Music")
axes[0].axis("off")

axes[1].imshow(horizontal_features, cmap="inferno")
axes[1].set_title("Horizontal Line Features")
axes[1].axis("off")

axes[2].imshow(vertical_features, cmap="inferno")
axes[2].set_title("Vertical Line Features")
axes[2].axis("off")

plt.tight_layout()
plt.show()
