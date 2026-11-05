# =============================================================================
# data.py -- CIFAR-10 loading, official CLIP preprocessing, few-shot splits
#
# Split discipline (critical for a fair comparison):
#   * support set  : k images/class drawn from the official CIFAR-10 TRAIN split
#   * validation   : 100 images/class drawn from TRAIN \ support (never reused
#                    as support; used for ALL hyperparameter selection)
#   * test         : the official 10,000-image test split, evaluated ONCE per
#                    method, only after hyperparameters are frozen.
# =============================================================================
import numpy as np
import torch
from torch.utils.data import Dataset

# NOTE: this file is executed into a single shared namespace (by run_experiments.py
# locally, or by the notebook cells), so project-level names (CLIP_MEAN, ...,
# VAL_PER_CLASS, DATA_ROOT, MOCK) are taken from config.py defined earlier.


def clip_image_transform():
    """The official OpenAI CLIP preprocessing, applied to 32x32 CIFAR images:
       resize to 224 (bicubic) -> center crop 224 -> tensor -> CLIP mean/std.
       This is identical to open_clip's preprocessing for the OpenAI weights."""
    from torchvision import transforms as T
    return T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def load_datasets(root=DATA_ROOT, mock=MOCK):
    """Returns (train_dataset, test_dataset, train_targets, test_targets).
    Datasets are None in mock mode (features are synthesized instead)."""
    if mock:
        rng = np.random.RandomState(0)
        ytr = rng.randint(0, NUM_CLASSES, 2000).astype(np.int64)
        yte = rng.randint(0, NUM_CLASSES, 500).astype(np.int64)
        print("[data] MOCK mode: synthetic targets (2000 train / 500 test)")
        return None, None, ytr, yte
    import torchvision
    tf = clip_image_transform()
    train_ds = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=tf)
    test_ds = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=tf)
    return train_ds, test_ds, np.asarray(train_ds.targets), np.asarray(test_ds.targets)


def make_few_shot_splits(targets, shots, seed, val_per_class=VAL_PER_CLASS):
    """Stratified sampling with a per-seed RNG.

    1. support: exactly `shots` images per class (disjoint across classes).
    2. validation: exactly `val_per_class` images per class from the REMAINING
       train images, guaranteed disjoint from the support set.

    Returns (support_idx, val_idx), both sorted int64 arrays.
    """
    targets = np.asarray(targets)
    rng = np.random.RandomState(seed)
    classes = np.unique(targets)

    support_per_class = []
    for c in classes:
        pool = np.flatnonzero(targets == c)
        support_per_class.append(rng.choice(pool, size=shots, replace=False))
    support_idx = np.sort(np.concatenate(support_per_class)).astype(np.int64)

    rest = np.setdiff1d(np.arange(len(targets)), support_idx)
    val_per_class_idx = []
    for c in classes:
        pool = rest[targets[rest] == c]
        val_per_class_idx.append(rng.choice(pool, size=val_per_class, replace=False))
    val_idx = np.sort(np.concatenate(val_per_class_idx)).astype(np.int64)

    assert len(support_idx) == shots * len(classes)
    assert len(val_idx) == val_per_class * len(classes)
    assert not np.intersect1d(support_idx, val_idx).size, "support/val leakage"
    return support_idx, val_idx
