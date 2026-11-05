# Tip-Adapter & Tip-Adapter-F on CIFAR-10 with CLIP ViT-B/32

**Training-free vs fine-tuned cache adaptation of CLIP in the few-shot regime.**
A complete, reproducible research pipeline for a **Kaggle Tesla T4 (16 GB VRAM)**, implementing
*"[Tip-Adapter: Training-free CLIP-Adapter for Better Vision-Language Modeling"](https://arxiv.org/abs/2207.09619)* (Zhang et al., ECCV 2022) and its fine-tuned extension Tip-Adapter-F.

---

## 1. Research question

> How effectively can Tip-Adapter and Tip-Adapter-F adapt a **frozen** pretrained CLIP (ViT-B/32)
> to CIFAR-10 using only *k* ∈ {1, 2, 4, 8, 16} labeled images per class?

Four approaches are compared on **identical support sets** and the **official test split**:

| # | Method | What is learned | Trainable params (16-shot) | Training? |
|---|------------------|--------------------------------------|----------------------------|-----------|
| 1 | **Zero-shot CLIP** | nothing (prompt ensemble only) | 0 | no |
| 2 | **Linear Probe** | linear head `W, b` on frozen features | 5,130 | yes (CE loss) |
| 3 | **Tip-Adapter** | nothing — non-parametric cache; α, β tuned on validation | 0 | no |
| 4 | **Tip-Adapter-F** | cache keys `K` via gradient descent (AdamW) | N×512 = 81,920 | yes (CE loss) |

## 2. Design at a glance

```
CIFAR-10 (50,000 train / 10,000 test, 32x32)
     |  resize 224x224 bicubic + center crop + official CLIP normalization
     v
FROZEN CLIP ViT-B/32  (image + text encoders; requires_grad=False)
     |                                   |
     v                                   v
image banks (disk-cached)         10 classes x 3 templates
train [50000,512]  test [10000,512]   -> averaged normalized text embeddings [10,512]
     |                                  logit_scale = exp(logit_scale) = 100
     |-- stratified splits per (shots, seed):  support [k/class]  +  validation [100/class]
     |
     +-- Zero-shot:      argmax( 100 * f_img @ T^T )
     +-- Linear probe:   argmax( W f_img + b ),  cross-entropy on support, AdamW
     +-- Tip-Adapter:    clip_logits + alpha * exp(-beta*(1 - f_img @ K^T)) @ V
     +-- Tip-Adapter-F:  same formula, but K is nn.Parameter trained with AdamW
```

Everything downstream of the first cell operates on **precomputed, disk-cached feature
banks**, so the frozen CLIP encoders run over the images exactly once per session
(~5 min on T4 with fp16 autocast). The encoders are then deleted from memory —
accidental CLIP fine-tuning is impossible by construction.

## 3. The Tip-Adapter cache model

For a support set of `N = 10k` images:

* `cache_keys  K ∈ R^{N×512}` — the L2-normalized CLIP image features of the support images
* `cache_values V ∈ {0,1}^{N×10}` — one-hot class labels of the support images

For a query image feature `q` (L2-normalized):

```
affinity       = exp(-beta * (1 - q @ K^T))          # in (0, 1], shape [Q, N]
cache_logits   = affinity @ V                        # [Q, 10]
final_logits   = clip_logits + alpha * cache_logits  # clip_logits = 100 * q @ T^T
```

**Role of the two hyperparameters**

* **β — affinity sharpness.** `cos(q, k)` ∈ [-1, 1], so `1 - cos ∈ [0, 2]`.
  Large β ⇒ only cache keys almost identical to the query contribute
  (nearest-neighbour behaviour); β → 0 ⇒ affinity ≈ uniform (every key counts ~1).
* **α — cache weight.** Scales the few-shot memory relative to CLIP's zero-shot
  logits. α = 0 recovers pure zero-shot CLIP; larger α trusts the support samples more.

Both are tuned by an exhaustive grid search on the **validation set only**
(α ∈ {0, 0.25, …, 6.00}, β ∈ {2, 3, 4, 5, 5.86, 7, 8}; 5.86 is the paper's default).

**Tip-Adapter-F** starts from the same cache initialization but makes `cache_keys` an
`nn.Parameter` — the **only** trainable tensor. `cache_values`, the text embeddings,
the logit scale, and both CLIP encoders are constants (buffers / deleted models).
Training:

* queries during training **are the support features** (official protocol: the cache is
  optimized on the few-shot samples it stores);
* loss = cross-entropy on `final_logits` (CLIP + α·cache branch together);
* AdamW over a small lr × weight-decay grid; ≤20 epochs with patience 5;
* **checkpoint selection + early stopping on validation accuracy** (ties → lower val loss);
* after training, **α is re-searched on validation** with the learned keys frozen
  (the official two-stage protocol).

## 4. Normalization strategy (exact)

* Image and text embeddings from frozen CLIP are **L2-normalized**.
* Text: per-template normalize → average the 3 templates → re-normalize per class.
* Cache keys: initialized from **L2-normalized** support features and **not**
  re-normalized during fine-tuning (matches the official implementation — the keys
  drift freely so they can become better class prototypes).
* Queries: always L2-normalized frozen features; never updated.
* CLIP's learned logit scale (`exp(logit_scale) = 100`) is kept in the CLIP logits.

## 5. Data & fairness protocol

1. **Support sets**: stratified sampling, exactly *k* per class, seeded by
   `seeds = [42, 123, 2026]`; the **same support images are used by all four methods**.
2. **Validation**: 100 images/class (1,000 total) drawn from the train split
   **minus** the support set. All hyperparameters (α, β, lr, weight decay, epochs)
   are chosen here. Validation images are never used as support.
3. **Test**: the official 10,000-image test split, evaluated **once per method,
   after** all selection is frozen. Test labels are never used for tuning,
   checkpoint selection, or early stopping.
4. **Seeds**: mean ± standard deviation over the 3 seeds for every reported number.

## 6. Evaluation metrics

Per method and (shots, seed): top-1 accuracy, top-5 accuracy, macro precision,
macro recall, macro F1, per-class accuracy, confusion matrix. For Tip-Adapter-F
additionally: training loss curve, validation accuracy curve, epochs run, best epoch,
training time, trainable-parameter count.

## 7. Ablations

| Ablation | Comparison | What it demonstrates |
|----------|------------|----------------------|
| **A. Cache contribution** | zero-shot CLIP vs CLIP + cache | how much a non-parametric few-shot memory alone adds to zero-shot CLIP |
| **B. Affinity function** | raw cosine similarity vs `exp(-β(1-cos))` (identical aggregation) | the value of the exponential kernel: it concentrates weight on the nearest support samples instead of averaging over all of them |
| **C. Fine-tuning effect** | fixed keys (Tip-Adapter) vs trainable keys (Tip-Adapter-F) | what gradient descent on the cache keys buys over pure retrieval |
| **D. Number of shots** | 1 → 2 → 4 → 8 → 16 | sample-efficiency of each method; where each plateaus |

## 8. Project structure

```
tip-adapter-cifar10/
├── README.md                     <- this document
├── src/
│   ├── config.py                 # constants, search grids, paths, seeds
│   ├── data.py                   # CIFAR-10, CLIP preprocessing, few-shot splits
│   ├── clip_features.py          # frozen CLIP loading, extraction, disk caching
│   ├── methods.py                # zero-shot, linear probe, Tip-Adapter, Tip-Adapter-F
│   ├── evaluation.py             # metrics, record keeping (JSON/CSV)
│   ├── ablations.py              # affinity ablation + derived tables
│   ├── visualize.py              # all publication figures (PNG)
│   └── run_experiments.py        # end-to-end orchestration (5 stages)
├── scripts/
│   ├── build_notebook.py         # generates the Kaggle notebook from src/ (single source of truth)
│   └── make_report.py            # local report from downloaded Kaggle outputs
├── kaggle/
│   ├── kernel-metadata.json      # T4 GPU + internet
│   ├── tip_adapter_cifar10.ipynb # generated self-contained notebook
│   └── output/                   # downloaded run artifacts (results, figures, checkpoints)
└── results/                      # local copy of key results
```

The Kaggle notebook is **generated from the same `src/` modules** by
`scripts/build_notebook.py`, so the local codebase and the cloud run cannot diverge.

## 9. How to run

### Kaggle (recommended — Tesla T4)
1. `pip install kaggle`, put your `kaggle.json` under `~/.kaggle/`.
2. `cd kaggle && kaggle kernels push -p .` (needs a phone-verified account for internet).
3. Monitor: `kaggle kernels status srinithetta/tip-adapter-cifar-10-clip-vit-b-32`.
4. Fetch outputs: `kaggle kernels output srinithetta/tip-adapter-cifar-10-clip-vit-b-32 -p kaggle/output`.

### Locally (logic test without GPU)
`TIP_MOCK=1 python src/run_experiments.py` — synthetic features exercise the complete
pipeline (splits, all four methods, tuning, ablations, figures) on CPU in ~1 minute.

### Full local run (any CUDA GPU)
`python src/run_experiments.py` — downloads CIFAR-10 + CLIP weights and executes everything.

## 10. Results

See `RESULTS.md` (generated after the Kaggle T4 run) and the artifacts in `kaggle/output/`.

## 11. Known limitations (documented design decisions)

* Training operates on **precomputed features** (as in the official implementation on
  CIFAR), so support-set data augmentation (the "augment epochs" variant of the paper)
  is not used; this affects Tip-Adapter-F more than Tip-Adapter but applies equally
  to all methods compared here.
* The validation set (1,000 images) introduces selection noise at 1-shot; three seeds
  average this out in the reported mean ± std.
* Zero-shot uses the 3 specified templates rather than the full 80-template CLIP
  ensemble, per the task specification.
