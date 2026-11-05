# =============================================================================
# visualize.py -- publication-quality figures, all saved as PNG
# =============================================================================
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter

try:
    from config import CLASSES, FIG_DIR, SEEDS, SHOTS
except ImportError:
    pass  # shared-namespace mode (notebook): names already defined by config cell

METHOD_ORDER = ["zero_shot", "linear_probe", "tip_adapter", "tip_adapter_f"]
METHOD_LABELS = {
    "zero_shot": "Zero-shot CLIP",
    "linear_probe": "Linear Probe",
    "tip_adapter": "Tip-Adapter (training-free)",
    "tip_adapter_f": "Tip-Adapter-F (fine-tuned)",
}
METHOD_COLORS = {
    "zero_shot": "#8c8c8c",
    "linear_probe": "#4477AA",
    "tip_adapter": "#EE7733",
    "tip_adapter_f": "#228833",
}


def _save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure] saved {path}")
    return path


def _shots_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(SHOTS)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Shots (support images per class)")


def _pick(records, method, shots, seed):
    return next(r for r in records
                if r["method"] == method and r["shots"] == shots and r["seed"] == seed)


def plot_accuracy_vs_shots(df, out_dir):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for m in METHOD_ORDER:
        sub = df[df.method == m].groupby("shots")["top1"]
        shots = sorted(sub.groups.keys())
        mean = sub.mean().loc[shots] * 100
        std = sub.std().loc[shots].fillna(0) * 100
        ax.errorbar(shots, mean, yerr=std, marker="o", ms=5, capsize=3, lw=1.8,
                    color=METHOD_COLORS[m], label=METHOD_LABELS[m])
    _shots_axis(ax)
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Few-shot accuracy on CIFAR-10 (CLIP ViT-B/32)\n"
                 f"mean +/- std over seeds {SEEDS}")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, out_dir, "01_accuracy_vs_shots.png")


def plot_tip_vs_tipf(df, out_dir):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    x = np.arange(len(SHOTS))
    w = 0.38
    pairs = [("tip_adapter", "Tip-Adapter (training-free)"),
             ("tip_adapter_f", "Tip-Adapter-F (fine-tuned)")]
    for k, (m, label) in enumerate(pairs):
        sub = df[df.method == m].groupby("shots")["top1"]
        mean = sub.mean().loc[SHOTS] * 100
        std = sub.std().loc[SHOTS].fillna(0) * 100
        bars = ax.bar(x + (k - 0.5) * w, mean, w, yerr=std, capsize=3,
                      color=METHOD_COLORS[m], label=label)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    lo = df[df.method.isin(["tip_adapter", "tip_adapter_f"])]["top1"].min() * 100
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}-shot" for s in SHOTS])
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_ylim(max(0.0, lo - 4), 100)
    ax.set_title("Tip-Adapter vs Tip-Adapter-F\n(error bars: std over seeds)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)
    return _save(fig, out_dir, "02_tipadapter_vs_tipadapterf.png")


def plot_tipf_curves(records, out_dir, seed=None):
    seed = SEEDS[0] if seed is None else seed
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    colors = plt.cm.viridis(np.linspace(0.08, 0.85, len(SHOTS)))
    for ci, shots in enumerate(SHOTS):
        try:
            rec = _pick(records, "tip_adapter_f", shots, seed)
        except StopIteration:
            continue
        hist = rec.get("history")
        if not hist:
            continue
        ep = [h["epoch"] for h in hist]
        axes[0].plot(ep, [h["train_loss"] for h in hist], color=colors[ci],
                     marker=".", ms=4, label=f"{shots}-shot")
        axes[1].plot(ep, [h["val_acc"] * 100 for h in hist], color=colors[ci],
                     marker=".", ms=4, label=f"{shots}-shot")
        best_acc = hist[rec["best_epoch"] - 1]["val_acc"] * 100
        axes[1].scatter([rec["best_epoch"]], [best_acc], color=colors[ci], marker="*",
                        s=120, zorder=5, edgecolor="k", linewidth=0.4)
    axes[0].set_title(f"Tip-Adapter-F training loss (seed {seed})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy on support set")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_title("Validation accuracy per epoch (star = selected checkpoint)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation top-1 (%)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    return _save(fig, out_dir, "03_tipf_training_curves.png")


def plot_confusion_grid(records, out_dir, shots=16, seed=None):
    seed = SEEDS[0] if seed is None else seed
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.2))
    last_im = None
    for ax, m in zip(axes.ravel(), METHOD_ORDER):
        rec = _pick(records, m, shots, seed)
        cm = np.array(rec["confusion_matrix"])
        last_im = ax.imshow(cm, cmap="Blues")
        thr = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6.5,
                        color="white" if cm[i, j] > thr else "#333333")
        ax.set_title(f"{METHOD_LABELS[m]}\n{shots}-shot, seed {seed} -- top-1 {rec['top1'] * 100:.2f}%",
                     fontsize=10)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(CLASSES, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.015)
    fig.suptitle(f"Confusion matrices on the CIFAR-10 test set (10,000 images, {shots}-shot, seed {seed})",
                 fontsize=12)
    return _save(fig, out_dir, "04_confusion_matrices_16shot.png")


def plot_per_class_accuracy(records, out_dir, shots=16, seed=None):
    seed = SEEDS[0] if seed is None else seed
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(CLASSES))
    w = 0.2
    lows = []
    for k, m in enumerate(METHOD_ORDER):
        acc = np.array(_pick(records, m, shots, seed)["per_class_accuracy"]) * 100
        lows.append(acc.min())
        ax.bar(x + (k - 1.5) * w, acc, w, color=METHOD_COLORS[m], label=METHOD_LABELS[m])
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=30, ha="right")
    ax.set_ylabel("Per-class accuracy (%)")
    ax.set_ylim(max(0.0, min(lows) - 4), 101)
    ax.set_title(f"Per-class accuracy on the test set ({shots}-shot, seed {seed})")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9, ncol=2)
    return _save(fig, out_dir, "05_per_class_accuracy_16shot.png")


def plot_training_time(records, out_dir):
    df = summarize_records(records)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = np.arange(len(SHOTS))
    w = 0.38
    for k, m in enumerate(["linear_probe", "tip_adapter_f"]):
        sub = df[df.method == m].groupby("shots")["train_time_s"]
        mean = sub.mean().loc[SHOTS]
        std = sub.std().loc[SHOTS].fillna(0)
        ax.bar(x + (k - 0.5) * w, mean, w, yerr=std, capsize=3,
               color=METHOD_COLORS[m], label=METHOD_LABELS[m])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}-shot" for s in SHOTS])
    ax.set_ylabel("Training time (s, mean over seeds)")
    ax.set_title("Training time: Linear Probe vs Tip-Adapter-F\n"
                 "(zero-shot and training-free Tip-Adapter need no training;\n"
                 "the Tip-Adapter (alpha, beta) search takes < 1 s per run)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)
    return _save(fig, out_dir, "06_training_time.png")


def plot_trainable_params(records, out_dir):
    df = summarize_records(records)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    labels = ["Zero-shot CLIP", "Tip-Adapter"]
    vals = [0, 0]
    lp = int(df[df.method == "linear_probe"]["trainable_params"].max())
    labels.append("Linear Probe")
    vals.append(lp)
    for s in SHOTS:
        sub = df[(df.method == "tip_adapter_f") & (df.shots == s)]
        labels.append(f"Tip-Adapter-F ({s}-shot)")
        vals.append(int(sub["trainable_params"].max()))
    y = np.arange(len(labels))
    colors = ["#8c8c8c", "#EE7733", "#4477AA"] + ["#228833"] * len(SHOTS)
    ax.barh(y, vals, color=colors)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.015 + 400, yi, f"{v:,}" if v else "0 (training-free)",
                va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Number of trainable parameters")
    ax.set_title("Trainable parameters per method\n"
                 "(CLIP image + text encoders, ~151M params, stay frozen in ALL methods)")
    ax.set_xlim(0, max(vals) * 1.2)
    ax.grid(alpha=0.3, axis="x")
    return _save(fig, out_dir, "07_trainable_params.png")


def plot_affinity_ablation(affinity_rows, out_dir):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = np.arange(len(SHOTS))
    w = 0.38
    exp_m = [r["exp_top1"] * 100 for r in affinity_rows]
    cos_m = [r["cos_top1"] * 100 for r in affinity_rows]
    b1 = ax.bar(x - w / 2, cos_m, w, color="#8c8c8c", label="Raw cosine similarity")
    b2 = ax.bar(x + w / 2, exp_m, w, color="#228833",
                label=r"Exponential: exp($-\beta$(1 - cos))")
    ax.bar_label(b1, fmt="%.1f", fontsize=8)
    ax.bar_label(b2, fmt="%.1f", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}-shot" for s in SHOTS])
    ax.set_ylabel("Test top-1 accuracy (%)")
    ax.set_ylim(max(0.0, min(cos_m + exp_m) - 4), 100)
    ax.set_title("Ablation B: affinity function (seed 42;\nhyperparameters tuned on validation)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, out_dir, "08_ablation_affinity.png")


def plot_alpha_sensitivity(sweep, out_dir, seed=None):
    seed = SEEDS[0] if seed is None else seed
    if not sweep:
        return None
    sw = pd.DataFrame(sweep)
    fig, ax = plt.subplots(figsize=(8, 5))
    for beta, g in sw.groupby("beta"):
        g = g.sort_values("alpha")
        ax.plot(g["alpha"], g["val_acc"] * 100, marker=".", ms=4, label=f"beta = {beta:g}")
    ax.set_xlabel("alpha  (weight of the cache logits)")
    ax.set_ylabel("Validation top-1 (%)")
    ax.set_title(f"Role of alpha and beta (training-free Tip-Adapter, 16-shot, seed {seed})\n"
                 "alpha scales the cache contribution; beta sharpens nearest-neighbour affinity")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, "09_alpha_beta_sensitivity.png")


def make_all_figures(records, affinity_rows, sweep_16, out_dir=FIG_DIR):
    os.makedirs(out_dir, exist_ok=True)
    df = summarize_records(records)
    plot_accuracy_vs_shots(df, out_dir)
    plot_tip_vs_tipf(df, out_dir)
    plot_tipf_curves(records, out_dir)
    plot_confusion_grid(records, out_dir)
    plot_per_class_accuracy(records, out_dir)
    plot_training_time(records, out_dir)
    plot_trainable_params(records, out_dir)
    if affinity_rows:
        plot_affinity_ablation(affinity_rows, out_dir)
    if sweep_16:
        plot_alpha_sensitivity(sweep_16, out_dir)
