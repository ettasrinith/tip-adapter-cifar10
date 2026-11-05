# =============================================================================
# make_report.py -- build RESULTS.md from the downloaded Kaggle output
#
# Usage:  python scripts/make_report.py [path-to-output-dir]
#         (default: ../kaggle/output, i.e. `kaggle kernels output` destination)
# =============================================================================
import json
import os
import shutil
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ORDER = ["zero_shot", "linear_probe", "tip_adapter", "tip_adapter_f"]
LABELS = {
    "zero_shot": "Zero-shot CLIP",
    "linear_probe": "Linear Probe",
    "tip_adapter": "Tip-Adapter (training-free)",
    "tip_adapter_f": "Tip-Adapter-F (fine-tuned)",
}
SEEDS = [42, 123, 2026]
SHOTS = [1, 2, 4, 8, 16]


def fmt(mean, std, scale=100, digits=2):
    return f"{mean * scale:.{digits}f} ± {std * scale:.{digits}f}"


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "kaggle", "output")
    res = os.path.join(out_dir, "results")
    records = json.load(open(os.path.join(res, "records.json"), encoding="utf-8"))
    df = pd.DataFrame([{k: v for k, v in r.items()
                        if k not in ("per_class_accuracy", "confusion_matrix", "history")}
                       for r in records])

    cfg = json.load(open(os.path.join(res, "run_config.json"), encoding="utf-8"))
    lines = []
    lines.append("# Results -- Tip-Adapter / Tip-Adapter-F on CIFAR-10 (CLIP ViT-B/32)\n")
    lines.append(f"*Run on Kaggle Tesla T4. Seeds {SEEDS}, shots {SHOTS}, "
                 f"validation = 100 images/class from the train split (disjoint from support), "
                 f"all hyperparameters selected on validation only.*\n")
    lines.append(f"*Environment: torch {cfg.get('torch')}, CLIP source `{cfg.get('clip_source')}`, "
                 f"logit scale {cfg.get('logit_scale')}.*\n")

    # Headline table
    lines.append("\n## Top-1 accuracy (%) on the official CIFAR-10 test set -- mean ± std over seeds\n")
    mean = df.pivot_table(index="method", columns="shots", values="top1", aggfunc="mean").loc[ORDER]
    std = df.pivot_table(index="method", columns="shots", values="top1", aggfunc="std").loc[ORDER].fillna(0)
    head = ["Method"] + [f"{s}-shot" for s in SHOTS]
    rows = [[LABELS[m]] + [fmt(mean.loc[m, s], std.loc[m, s]) for s in SHOTS] for m in ORDER]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "---|" * len(head))
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")

    # Full metrics at 16-shot
    lines.append("\n## Full test metrics at 16-shot (mean over seeds, %)\n")
    d16 = df[df.shots == 16].groupby("method")[["top1", "top5", "macro_precision",
                                                "macro_recall", "macro_f1"]].mean().loc[ORDER] * 100
    lines.append("| Method | Top-1 | Top-5 | Macro P | Macro R | Macro F1 |")
    lines.append("|---|---|---|---|---|---|")
    for m in ORDER:
        v = d16.loc[m]
        lines.append(f"| {LABELS[m]} | {v.top1:.2f} | {v.top5:.2f} | {v.macro_precision:.2f} "
                     f"| {v.macro_recall:.2f} | {v.macro_f1:.2f} |")

    # Efficiency
    lines.append("\n## Efficiency at 16-shot (mean over seeds)\n")
    aux = df[df.shots == 16].groupby("method")[["trainable_params", "train_time_s",
                                                "epochs_run", "best_epoch"]].mean().loc[ORDER]
    lines.append("| Method | Trainable params | Train time (s) | Epochs run | Best epoch |")
    lines.append("|---|---|---|---|---|")
    for m in ORDER:
        v = aux.loc[m]
        lines.append(f"| {LABELS[m]} | {int(v.trainable_params):,} | {v.train_time_s:.2f} "
                     f"| {v.epochs_run:.0f} | {v.best_epoch:.0f} |")

    # Ablations
    aff_path = os.path.join(res, "ablation_affinity.json")
    if os.path.exists(aff_path):
        aff = pd.DataFrame(json.load(open(aff_path, encoding="utf-8")))
        lines.append("\n## Ablation B -- affinity function (seed 42, test top-1 %)\n")
        lines.append("| Shots | Raw cosine | exp(-beta(1-cos)) | Delta |")
        lines.append("|---|---|---|---|")
        for _, r in aff.iterrows():
            lines.append(f"| {int(r.shots)} | {r.cos_top1 * 100:.2f} | {r.exp_top1 * 100:.2f} "
                         f"| {(r.exp_top1 - r.cos_top1) * 100:+.2f} |")

    cache_path = os.path.join(res, "ablation_cache_and_finetune.csv")
    if os.path.exists(cache_path):
        ab = pd.read_csv(cache_path)
        lines.append("\n## Ablations A & C -- cache contribution and fine-tuning effect (test top-1 %, mean over seeds)\n")
        lines.append("| Shots | Zero-shot | + Cache (A) | + Trained cache (C) | A gain | C gain |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in ab.iterrows():
            lines.append(f"| {int(r.shots)} | {r.zero_shot:.2f} | {r.clip_plus_cache:.2f} "
                         f"| {r.clip_plus_trained_cache:.2f} | {r.ablationA_cache_gain:+.2f} "
                         f"| {r.ablationC_finetune_gain:+.2f} |")

    lines.append("\n## Artifacts\n")
    fig_src = os.path.join(out_dir, "figures")
    names = sorted(os.listdir(fig_src)) if os.path.isdir(fig_src) else []
    for n in names:
        lines.append(f"- `figures/{n}`")
    lines.append("- `results/records.json`, `results/records.csv`, `results/run_config.json`, "
                 "`results/summary_top1_mean_std.csv`")
    lines.append("- `checkpoints/tip_adapter_f_<shots>shot_seed<seed>.pt` (best fine-tuned cache keys)")

    dest = os.path.join(ROOT, "RESULTS.md")
    with open(dest, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {dest}")

    local_figs = os.path.join(ROOT, "results", "figures")
    if os.path.isdir(fig_src):
        os.makedirs(local_figs, exist_ok=True)
        for n in names:
            shutil.copy2(os.path.join(fig_src, n), os.path.join(local_figs, n))
        print(f"copied {len(names)} figures to {local_figs}")


if __name__ == "__main__":
    main()
