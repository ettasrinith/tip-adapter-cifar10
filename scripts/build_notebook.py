# =============================================================================
# build_notebook.py -- assemble the self-contained Kaggle notebook
#
# Single source of truth: the notebook cells are generated directly from the
# src/*.py modules, so the local codebase and the Kaggle run can never diverge.
# =============================================================================
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
OUT_DIR = os.path.join(ROOT, "kaggle")


def read(name, strip_main=False):
    text = open(os.path.join(SRC, name), encoding="utf-8").read()
    if strip_main:
        marker = 'if __name__ == "__main__":'
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].rstrip() + "\n"
    return text


TITLE_MD = """# Tip-Adapter & Tip-Adapter-F on CIFAR-10 with CLIP ViT-B/32
**Training-free vs fine-tuned cache adaptation of CLIP in the few-shot regime.**
Complete, reproducible pipeline for a Kaggle **Tesla T4 (16 GB)**.

---

## 1. Research question

> How effectively can Tip-Adapter and Tip-Adapter-F adapt a *frozen* pretrained CLIP
> to CIFAR-10 using only $k \\in \\{1,2,4,8,16\\}$ labeled images per class?

We compare four approaches on identical support sets and the official test split:

| # | Method | What is learned | Trainable params (16-shot) |
|---|--------|-----------------|----------------------------|
| 1 | **Zero-shot CLIP** | nothing (prompt ensemble only) | 0 |
| 2 | **Linear Probe** | a linear head $W,b$ on frozen features | 5,130 |
| 3 | **Tip-Adapter** | nothing (non-parametric cache, tuned $\\alpha,\\beta$) | 0 |
| 4 | **Tip-Adapter-F** | the cache keys $K$ (gradient descent) | $N \\times 512$ = 81,920 |

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
     |
     |-- stratified splits per (shots, seed):  support [k/class]  +  validation [100/class]
     |
     +-- Zero-shot:      argmax( 100 * f_img . T^T )
     +-- Linear probe:   argmax( W f_img + b ),  cross-entropy on support
     +-- Tip-Adapter:    clip_logits + alpha * exp(-beta*(1 - f_img . K^T)) . V
     +-- Tip-Adapter-F:  same formula, but K is nn.Parameter, trained with AdamW
```

## 3. The Tip-Adapter cache model (both variants)

Support set of $N = 10k$ images with CLIP features $K \\in \\mathbb{R}^{N\\times512}$
(L2-normalized) and one-hot labels $V \\in \\{0,1\\}^{N\\times10}$:

$$A(q, K) = \\exp\\big(-\\beta\\,(1 - qK^{\\top})\\big) \\qquad \\text{(affinity, in } (0,1])$$
$$\\text{cache\\_logits}(q) = A(q,K)\\,V \\qquad \\text{final}(q) = \\underbrace{100\\, qT^{\\top}}_{\\text{CLIP logits}} + \\; \\alpha\\, \\text{cache\\_logits}(q)$$

* **$\\beta$ (affinity sharpness):** large $\\beta$ -> only cache keys almost identical to the
  query contribute (nearest-neighbour behaviour); $\\beta \\to 0$ -> affinity ~ uniform.
* **$\\alpha$ (cache weight):** global weight of the few-shot memory relative to CLIP's
  zero-shot logits; $\\alpha = 0$ recovers pure zero-shot CLIP.

**Tip-Adapter-F** initializes $K$ from the support features and makes it the *only*
trainable tensor (`nn.Parameter`). $V$, the text embeddings, the logit scale and both
CLIP encoders are constants. Loss = cross-entropy on `final(q)` with the support
images themselves as queries (official protocol); AdamW; early stopping and checkpoint
selection on the **validation** set; after training $\\alpha$ is re-searched on validation
with the learned keys frozen (official two-stage protocol).

## 4. Normalization strategy (exact)

* image and text embeddings from frozen CLIP: **L2-normalized**;
* text: per-template normalize -> average over 3 templates -> re-normalize;
* cache keys: initialized from **L2-normalized** support features, **not**
  re-normalized during training (matches the official implementation);
* queries: always L2-normalized frozen features, never updated;
* CLIP's learned logit scale ($\\exp(\\text{logit\\_scale}) = 100$) is kept in the CLIP logits.

## 5. Experimental protocol (fairness rules)

1. Support sets: stratified, exactly $k$ per class, seeded by
   `seeds = [42, 123, 2026]`; **identical support images for all four methods**.
2. Validation: 100 images/class drawn from the train split **minus** the support set;
   every hyperparameter ($\\alpha, \\beta$, lr, weight decay, epochs) is chosen here.
3. Test: the official 10,000-image split, evaluated **once** per method, after all
   selection is frozen. No test labels are ever used for tuning, checkpoints, or
   early stopping.
4. Frozen CLIP embeddings are extracted once and cached to `feature_cache/*.pt`;
   all methods consume the same banks.
5. Report mean +/- std over the 3 seeds; per-run records, checkpoints, figures and
   config dumps are saved under `/kaggle/working`.

## 6. Ablations

* **A - Cache contribution:** zero-shot vs CLIP + cache -> what the few-shot memory adds.
* **B - Affinity function:** raw cosine vs $\\exp(-\\beta(1-\\cos))$ with identical
  aggregation -> the value of the exponential kernel.
* **C - Fine-tuning effect:** fixed vs trainable cache keys -> what gradients buy.
* **D - Shots:** the main 1/2/4/8/16 sweep -> sample-efficiency of each method.

---

*Everything below is executable: it runs top-to-bottom on a T4 in ~20-30 minutes.*
"""

SETUP_MD = """## 7. Setup
`open_clip_torch` is the only package that is not preinstalled on Kaggle.
Everything else (torch, torchvision, pandas, scikit-learn, matplotlib) is already there."""

CONFIG_MD = """## 8. Configuration
All constants, search grids and paths in one place. `MOCK` mode (env var `TIP_MOCK=1`)
replaces CLIP/CIFAR with synthetic features so the pipeline logic can be tested on CPU."""

DATA_MD = """## 9. Data & few-shot splits
Official CIFAR-10 train/test splits; official CLIP preprocessing (bicubic resize to
224, center crop, CLIP mean/std). `make_few_shot_splits` samples the support set and
a disjoint 1,000-image validation set from the remaining *training* images, so
validation can never leak into support and test is never touched here."""

FEAT_MD = """## 10. Frozen CLIP + cached feature banks
The image encoder is run once over all 60,000 images (fp16 autocast on the T4) and
the normalized embeddings are cached to disk. The text side encodes 10 classes x 3
templates. Every downstream method operates on these banks only - the CLIP encoders
are deleted afterwards, which makes accidental CLIP fine-tuning impossible."""

METHODS_MD = """## 11. The four methods
Zero-shot, linear probe, training-free Tip-Adapter (validation grid over
$\\alpha \\in \\{0, 0.25, ..., 6\\} \\times \\beta \\in \\{2,3,4,5,5.86,7,8\\}$), and
Tip-Adapter-F (AdamW on the trainable keys, lr x wd grid, early stopping on validation,
post-training $\\alpha$ re-search). Trainable-parameter verification is built into the
training function (`assert` that only `cache_keys` receives gradients)."""

EVAL_MD = """## 12. Metrics
Top-1 / Top-5 accuracy, macro precision / recall / F1, per-class accuracy and the
confusion matrix - computed from the saved logits, never re-tuned."""

ABL_MD = """## 13. Ablations
Ablation B (affinity function) is computed fresh for every shot setting on seed 42;
ablations A and C are derived from the main records at summary time."""

VIZ_MD = """## 14. Figures
All plots are saved as 200-dpi PNGs under `figures/`."""

PROTO_MD = """## 15. Run the experiment
One call executes: feature banks -> 60 evaluated runs (4 methods x 5 shots x 3 seeds)
-> ablations -> 9 figures -> summary tables. Records are saved incrementally, so the
run is resumable at the artifacts level if interrupted."""

RESULTS_MD = """## 16. Reading the results
* `results/records.json` / `records.csv` -- one row per (method, shots, seed) with all
  metrics, timings, trainable-parameter counts and chosen hyperparameters.
* `results/summary_top1_mean_std.csv` -- the headline mean +/- std table.
* `figures/*.png` -- the publication figures.
* `checkpoints/*.pt` -- best fine-tuned cache keys per run.
* `results/run_config.json` -- exact configuration + library versions for reproducibility.
* `feature_cache/*.pt` -- cached frozen CLIP embeddings (skip re-extraction on re-runs).
"""


def build():
    os.makedirs(OUT_DIR, exist_ok=True)
    cells = []

    def md(text):
        cells.append(dict(cell_type="markdown", metadata={}, source=text.splitlines(keepends=True)))

    def code(text):
        cells.append(dict(cell_type="code", metadata={}, execution_count=None,
                          outputs=[], source=text.splitlines(keepends=True)))

    md(TITLE_MD)
    md(SETUP_MD)
    code("!pip install -q open_clip_torch\n"
         "\n"
         "import torch, sys\n"
         "print('python', sys.version.split()[0], '| torch', torch.__version__,\n"
         "      '| cuda available:', torch.cuda.is_available())\n"
         "if torch.cuda.is_available():\n"
         "    print('GPU:', torch.cuda.get_device_name(0),\n"
         "          '| VRAM: %.1f GB' % (torch.cuda.get_device_properties(0).total_memory / 1e9))\n")
    md(CONFIG_MD)
    code(read("config.py"))
    md(DATA_MD)
    code(read("data.py"))
    md(FEAT_MD)
    code(read("clip_features.py"))
    md(METHODS_MD)
    code(read("methods.py"))
    md(EVAL_MD)
    code(read("evaluation.py"))
    md(ABL_MD)
    code(read("ablations.py"))
    md(VIZ_MD)
    code(read("visualize.py"))
    md(PROTO_MD)
    code(read("run_experiments.py", strip_main=True))
    code("records = run_all()")
    md(RESULTS_MD)

    nb = dict(cells=cells,
              metadata=dict(kernelspec=dict(display_name="Python 3", language="python",
                                            name="python3"),
                            language_info=dict(name="python")),
              nbformat=4, nbformat_minor=4)
    out = os.path.join(OUT_DIR, "tip_adapter_cifar10.ipynb")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"notebook written: {out} ({len(cells)} cells)")


if __name__ == "__main__":
    build()
