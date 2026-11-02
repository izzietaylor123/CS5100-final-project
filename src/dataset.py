"""
PrimusDataset gets one sample. When asked for sample number idx, it hands back:
    - the image (already resized and normalized)
    - the label (the symbols in that image, written as numbers from vocab.py)

collate_batch() packs a group of samples (e.g., 16 samples) into one batch
The images have different widths, so this function:
    - pads the short images with blank space so they are all the same width
    - puts all the labels (clef-G2, timeSignature-2/4, note-A4_quarter, ...)
      into one long list of numbers
    - writes down the real image widths and label lengths, so CTC knows
      what is real and what is just padding in each image
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocess import IMG_HEIGHT, load_image
from src.vocab import read_tokens


class PrimusDataset(Dataset):
    """Holds a list of folders and loads one image at a time."""

    def __init__(self, sample_dirs, token_to_id):
        # a list of folders (e.g., all the training folders)
        self.sample_dirs = sample_dirs

        # dictionary that turns each symbol into a number (from vocab.py)
        self.token_to_id = token_to_id

    def __len__(self):
        # how many samples there are (one folder = one sample)
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        # find the folder for sample number idx
        folder = Path(self.sample_dirs[idx])
        name = folder.name

        # loads the image
        image_path = folder / (name + ".png")
        img = load_image(str(image_path))

        # loads the label symbols
        label_path = folder / (name + ".semantic")
        tokens = read_tokens(label_path)

        # turns each symbol into its number identifier
        label = []
        for token in tokens:
            number = self.token_to_id[token]
            label.append(number)

        return img, label


def collate_batch(batch):
    """Combine a list of (image, label) samples into one batch.

    Input:  a list like [(image1, label1), (image2, label2), ...]
    Output: padded images, all labels in one list, real image widths,
            and the number of symbols in each label
    """

    # separate the images from the labels
    images = []
    labels = []
    for sample in batch:
        images.append(sample[0])
        labels.append(sample[1])

    # record each image's real width, and find the widest one
    widths = []
    for img in images:
        widths.append(img.shape[1])
    max_width = max(widths)

    # make an empty block of zeros big enough for every image
    batch_size = len(images)
    padded = np.zeros((batch_size, 1, IMG_HEIGHT, max_width), dtype=np.float32)

    # copy each image into the block (leftover space stays zero)
    for i in range(batch_size):
        img = images[i]
        width = img.shape[1]
        padded[i, 0, :, :width] = img

    # record how many symbols each label has
    label_lengths = []
    for label in labels:
        label_lengths.append(len(label))

    # join all labels into one long list
    flat_labels = []
    for label in labels:
        for number in label:
            flat_labels.append(number)

    # convert to PyTorch tensors
    images_tensor = torch.from_numpy(padded)                        # (B, 1, 64, max_width)
    labels_tensor = torch.tensor(flat_labels, dtype=torch.long)     # all labels in one list
    widths_tensor = torch.tensor(widths, dtype=torch.long)          # real pixel widths
    lengths_tensor = torch.tensor(label_lengths, dtype=torch.long)  # symbols per label

    return images_tensor, labels_tensor, widths_tensor, lengths_tensor