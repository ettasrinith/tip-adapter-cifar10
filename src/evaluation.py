# =============================================================================
# evaluation.py -- metrics, record keeping, result tables
# =============================================================================
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

try:
    from config import NUM_CLASSES
except ImportError:
    pass  # shared-namespace mode (notebook): names already defined by config cell


def compute_metrics(y_true, y_pred, logits=None, num_classes=NUM_CLASSES):
    """Top-1, Top-5, macro precision/recall/F1, per-class accuracy, confusion."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    top1 = float((y_true == y_pred).mean())

    top5 = None
    if logits is not None:
        lg = logits if torch.is_tensor(logits) else torch.as_tensor(logits)
        k = min(5, num_classes)
        top5_idx = torch.topk(lg, k=k, dim=1).indices.numpy()
        top5 = float(np.mean([y_true[i] in top5_idx[i] for i in range(len(y_true))]))

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    return dict(top1=top1, top5=top5,
                macro_precision=float(prec), macro_recall=float(rec), macro_f1=float(f1),
                per_class_accuracy=[float(x) for x in per_class],
                confusion_matrix=cm.astype(int).tolist())


def to_py(obj):
    """Recursively convert numpy/torch containers to plain Python for JSON."""
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj


def save_records(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_py(records), f, indent=2)


def summarize_records(records):
    """Flatten records into a DataFrame without the bulky per-class fields."""
    rows = [{k: (json.dumps(v) if k == "hp" and isinstance(v, dict) else v)
             for k, v in r.items()
             if k not in ("per_class_accuracy", "confusion_matrix", "history")}
            for r in records]
    return pd.DataFrame(rows)


def pivot_mean_std(df, metric="top1", index="method", columns="shots"):
    mean = df.pivot_table(index=index, columns=columns, values=metric, aggfunc="mean")
    std = df.pivot_table(index=index, columns=columns, values=metric, aggfunc="std")
    return mean, std
