
import torch
import matplotlib.pyplot as plt

from src.model import MaxPoolingLayer

# 1. Create an 8x8 grayscale image
image = torch.tensor([
    [1., 2., 3., 4., 5., 6., 7., 8.],
    [2., 9., 4., 3., 6., 5., 8., 7.],
    [3., 4., 8., 2., 7., 9., 1., 5.],
    [5., 6., 3., 7., 2., 4., 6., 8.],
    [1., 3., 5., 7., 9., 2., 4., 6.],
    [2., 4., 6., 8., 3., 5., 7., 9.],
    [8., 6., 4., 2., 1., 3., 5., 7.],
    [7., 5., 3., 1., 2., 4., 6., 8.]
])

# PyTorch expects (batch, channels, height, width)
image = image.unsqueeze(0).unsqueeze(0)

# 2. Apply YOUR MaxPoolingLayer
pooling = MaxPoolingLayer(kernel_size=4, stride=4)
pooled_image = pooling(image)

print("Pooled values:")
print(pooled_image)
print("Original shape:", image.shape)
print("Pooled shape:", pooled_image.shape)
print("\nPooled values:\n", pooled_image)

# 3. Visualize before and after
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(image[0, 0], cmap="gray", vmin=0, vmax=9)
axes[0].set_title("Original Image (8x8)")
axes[0].set_xticks(range(8))
axes[0].set_yticks(range(8))
axes[0].grid()

axes[1].imshow(pooled_image[0, 0], cmap="gray", vmin=0, vmax=10)
axes[1].set_title("After Max Pooling (4x4)")
axes[1].set_xticks(range(4))
axes[1].set_yticks(range(4))
axes[1].grid()

plt.tight_layout()
plt.show()
