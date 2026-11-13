import random
from pathlib import Path
from src.dataset import PrimusDataset
from src.vocab import build_vocab

folders = sorted(p for p in Path("sheet music/primus/package_aa").iterdir() if p.is_dir())  
random.Random(42).shuffle(folders)   # seed 42 -> same shuffle every run

# create a 70/15/15 split for training, validation, and testing
n = len(folders)
train_dirs = folders[: int(0.7 * n)]
val_dirs   = folders[int(0.7 * n): int(0.85 * n)]
test_dirs  = folders[int(0.85 * n):]

# build the vocabs and the dataset objects for each split
label_paths = [d / (d.name + ".semantic") for d in folders]
token_to_id, id_to_token = build_vocab(label_paths)
train_ds = PrimusDataset(train_dirs, token_to_id)
val_ds   = PrimusDataset(val_dirs, token_to_id)
test_ds  = PrimusDataset(test_dirs, token_to_id)