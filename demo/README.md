# Live Demo — CIFAR-10 Few-Shot Classifier (localhost web app)

Upload any image (or click one of the samples) at **http://127.0.0.1:8080** and the
fine-tuned Tip-Adapter-F model predicts its class — plus a side-by-side comparison of
all three inference methods and a cache-explainability view.

## Run it

```bash
pip install flask open_clip_torch torchvision   # one-time
python demo/app.py                              # then open http://127.0.0.1:8080
```

The server loads, from this repository's `kaggle/output/` artifacts:

| Artifact | Purpose |
|---|---|
| CLIP ViT-B/32 weights (local cache, ~354 MB) | frozen image encoder |
| `kaggle/output/feature_cache/text_feats.pt` | the 10 class text embeddings (exact run artifacts) |
| `kaggle/output/checkpoints/tip_adapter_f_16shot_seed42.pt` | fine-tuned cache keys (α=1.25, β=8.0) |
| `kaggle/output/feature_cache/train_bank.pt` | support features for the training-free Tip-Adapter baseline |
| `demo/data/cifar-10-batches-py/` | CIFAR-10 (test gallery + support images) |

The local CLIP features were verified against the Kaggle feature bank: **0.99995 mean
cosine similarity**, so the demo reproduces the reported results exactly.

## Use your own images (Google photos work)

1. Drop any photo — JPG/PNG/WebP, **any size or aspect ratio** — into
   `demo/my_images/<class>/` (e.g. a dog photo into `demo/my_images/dog/`).
2. Refresh the page; your images appear in section **2b**. Click one to predict.
3. Or just drag-and-drop any image onto the upload zone (section 1).

**No manual resizing needed**: every image is resized to 224×224 with the official
CLIP preprocessing (bicubic + center crop + CLIP normalization) inside the code
(`predict_pil()` → `clip_image_transform()`). Verified on 10 real Wikipedia photos
(77 KB – 3.4 MB, arbitrary resolutions): **10/10 correct** at 98–100 % confidence,
~100 ms per image on CPU.

Folder contents are git-ignored (keep your personal images out of the repo);
the folder structure is committed with `.gitkeep` files.

## Endpoints

| Route | What it does |
|---|---|
| `GET /` | the demo UI |
| `GET /api/health` | model info, α/β, device |
| `GET /api/gallery` | 10 CIFAR-10 test samples (one per class) |
| `GET /api/image/<test_idx>` · `/api/train_image/<train_idx>` | thumbnail of a test/train image |
| `GET /api/predict_test/<test_idx>` | predict a CIFAR test image |
| `GET /api/my_images` | scan `demo/my_images/<class>/` |
| `GET /api/my_image/<class>/<file>` | thumbnail of your image |
| `GET /api/predict_my/<class>/<file>` | predict your image (expected-class aware) |
| `POST /api/predict` (multipart `file`) | predict an uploaded image |

Each prediction returns: Tip-Adapter-F top-5 with probabilities, zero-shot /
Tip-Adapter / Tip-Adapter-F side-by-side, the 3 nearest (fine-tuned) cache keys with
their cosine similarities, and inference latency.
