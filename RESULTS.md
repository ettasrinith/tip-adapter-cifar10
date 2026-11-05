# Results -- Tip-Adapter / Tip-Adapter-F on CIFAR-10 (CLIP ViT-B/32)

*Run on Kaggle Tesla T4. Seeds [42, 123, 2026], shots [1, 2, 4, 8, 16], validation = 100 images/class from the train split (disjoint from support), all hyperparameters selected on validation only.*

*Environment: torch 2.11.0+cu130, CLIP source `open_clip:ViT-B-32:openai`, logit scale 100.0.*


## Top-1 accuracy (%) on the official CIFAR-10 test set -- mean ± std over seeds

| Method | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|---|---|---|---|---|---|
| Zero-shot CLIP | 85.25 ± 0.00 | 85.25 ± 0.00 | 85.25 ± 0.00 | 85.25 ± 0.00 | 85.25 ± 0.00 |
| Linear Probe | 66.34 ± 3.96 | 77.38 ± 5.40 | 83.95 ± 1.85 | 87.71 ± 0.54 | 89.67 ± 0.55 |
| Tip-Adapter (training-free) | 86.42 ± 0.94 | 87.17 ± 0.68 | 88.30 ± 0.42 | 88.56 ± 0.25 | 88.83 ± 0.53 |
| Tip-Adapter-F (fine-tuned) | 87.11 ± 1.08 | 87.71 ± 0.53 | 88.53 ± 0.55 | 89.12 ± 0.23 | 90.12 ± 0.59 |

## Full test metrics at 16-shot (mean over seeds, %)

| Method | Top-1 | Top-5 | Macro P | Macro R | Macro F1 |
|---|---|---|---|---|---|
| Zero-shot CLIP | 85.25 | 99.03 | 86.59 | 85.25 | 85.14 |
| Linear Probe | 89.67 | 99.42 | 90.26 | 89.67 | 89.76 |
| Tip-Adapter (training-free) | 88.83 | 99.45 | 89.53 | 88.83 | 88.87 |
| Tip-Adapter-F (fine-tuned) | 90.12 | 99.67 | 90.39 | 90.12 | 90.12 |

## Efficiency at 16-shot (mean over seeds)

| Method | Trainable params | Train time (s) | Epochs run | Best epoch |
|---|---|---|---|---|
| Zero-shot CLIP | 0 | 0.00 | 0 | 0 |
| Linear Probe | 5,130 | 2.02 | 300 | 133 |
| Tip-Adapter (training-free) | 0 | 0.02 | 0 | 0 |
| Tip-Adapter-F (fine-tuned) | 81,920 | 0.02 | 10 | 5 |

## Ablation B -- affinity function (seed 42, test top-1 %)

| Shots | Raw cosine | exp(-beta(1-cos)) | Delta |
|---|---|---|---|
| 1 | 85.39 | 85.41 | +0.02 |
| 2 | 86.00 | 86.38 | +0.38 |
| 4 | 87.82 | 88.13 | +0.31 |
| 8 | 88.56 | 88.71 | +0.15 |
| 16 | 88.30 | 88.78 | +0.48 |

## Ablations A & C -- cache contribution and fine-tuning effect (test top-1 %, mean over seeds)

| Shots | Zero-shot | + Cache (A) | + Trained cache (C) | A gain | C gain |
|---|---|---|---|---|---|
| 1 | 85.25 | 86.42 | 87.11 | +1.17 | +0.69 |
| 2 | 85.25 | 87.17 | 87.71 | +1.92 | +0.54 |
| 4 | 85.25 | 88.30 | 88.53 | +3.05 | +0.23 |
| 8 | 85.25 | 88.56 | 89.12 | +3.31 | +0.56 |
| 16 | 85.25 | 88.83 | 90.12 | +3.58 | +1.29 |

## Artifacts

- `figures/01_accuracy_vs_shots.png`
- `figures/02_tipadapter_vs_tipadapterf.png`
- `figures/03_tipf_training_curves.png`
- `figures/04_confusion_matrices_16shot.png`
- `figures/05_per_class_accuracy_16shot.png`
- `figures/06_training_time.png`
- `figures/07_trainable_params.png`
- `figures/08_ablation_affinity.png`
- `figures/09_alpha_beta_sensitivity.png`
- `results/records.json`, `results/records.csv`, `results/run_config.json`, `results/summary_top1_mean_std.csv`
- `checkpoints/tip_adapter_f_<shots>shot_seed<seed>.pt` (best fine-tuned cache keys)

## Run timeline (Kaggle T4 session)

| Phase | Time |
|---|---|
| CIFAR-10 download from cs.toronto.edu (slow mirror) | ~71 min |
| CLIP ViT-B/32 weights + feature extraction (60,000 images, fp16) | ~79 s |
| All 60 method evaluations (4 methods x 5 shots x 3 seeds) incl. tuning | 35 s |
| Ablations + 9 figures + summary tables | ~4 s |
| **Total session** | **~74 min** |

*Re-running with CIFAR-10 attached as a Kaggle dataset (e.g. `cifar10-python`) removes the download and the whole study finishes in ~5 minutes. The verified frozen checks from the log: `CLIP parameters: total 151,277,313 | all frozen (requires_grad=False): True` and `Tip-Adapter-F trainable parameter names: ['cache_keys']`.*
