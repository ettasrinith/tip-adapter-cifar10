"""
Localhost demo server -- CIFAR-10 few-shot classifier.

Loads the exact artifacts produced by the Kaggle T4 study:
  * frozen OpenAI CLIP ViT-B/32 image encoder (weights cached on this laptop)
  * cached text embeddings + logit scale from the experiment run
  * Tip-Adapter-F fine-tuned cache keys (16-shot, seed 42 checkpoint)
  * training-free Tip-Adapter keys (support features of the same split/seed)

Run:      python demo/app.py
Open:     http://127.0.0.1:8080
"""
import base64
import io
import json
import random
import sys
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch
from flask import Flask, jsonify, request, send_file, send_from_directory
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
OUT = ROOT / "kaggle" / "output"
sys.path.insert(0, str(SRC))


def _load_shared(name):
    """Exec experiment modules into this file's globals (shared-namespace
    pattern, identical to run_experiments.py)."""
    exec(compile((SRC / name).read_text(encoding="utf-8"), str(SRC / name), "exec"), globals())


for _name in ("config.py", "data.py", "clip_features.py", "methods.py"):
    _load_shared(_name)

import torchvision  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = ROOT / "demo" / "data"
CKPT_PATH = OUT / "checkpoints" / "tip_adapter_f_16shot_seed42.pt"
BANK_DIR = OUT / "feature_cache"
RECORDS_PATH = OUT / "results" / "records.json"
MY_IMAGES_DIR = ROOT / "demo" / "my_images"   # user's own photos, one folder per class
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
PORT = 8080

STATE = {}


def _valid_images_in(folder):
    """List image files in a folder whose bytes PIL can actually open."""
    out = []
    if not folder.is_dir():
        return out
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in IMG_EXTS and p.is_file():
            try:
                with Image.open(p) as im:
                    im.verify()
                out.append(p.name)
            except Exception:
                pass
    return out


def _my_image_path(cls, fname):
    """Resolve a (class, filename) pair safely inside demo/my_images."""
    if cls not in CLASSES:
        return None
    p = (MY_IMAGES_DIR / cls / fname).resolve()
    try:
        p.relative_to(MY_IMAGES_DIR.resolve())
    except ValueError:
        return None
    return p if p.is_file() else None


def load_everything():
    import open_clip

    print(f"[demo] device: {DEVICE}", flush=True)
    print("[demo] loading frozen CLIP ViT-B/32 (local weight cache) ...", flush=True)
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(DEVICE).eval()

    text_feats = torch.load(BANK_DIR / "text_feats.pt")
    logit_scale = float(torch.load(BANK_DIR / "logit_scale.pt"))
    train_bank = torch.load(BANK_DIR / "train_bank.pt")

    print(f"[demo] loading fine-tuned cache keys: {CKPT_PATH.name} ...", flush=True)
    ckpt = torch.load(CKPT_PATH)
    records = json.loads(RECORDS_PATH.read_text(encoding="utf-8"))

    def rec(method):
        return next(r for r in records
                    if r["method"] == method and r["shots"] == 16 and r["seed"] == 42)

    tip_hp = rec("tip_adapter")["hp"]
    test_acc = {m: rec(m)["top1"] for m in ("zero_shot", "tip_adapter", "tip_adapter_f")}

    print("[demo] loading CIFAR-10 (train images for support display + test gallery) ...", flush=True)
    train_ds = torchvision.datasets.CIFAR10(root=str(DATA_ROOT), train=True, download=False, transform=None)
    test_ds = torchvision.datasets.CIFAR10(root=str(DATA_ROOT), train=False, download=False, transform=None)
    ytr = np.asarray(train_ds.targets)
    yte = np.asarray(test_ds.targets)

    support_idx, _ = make_few_shot_splits(ytr, 16, 42)
    support_labels = ytr[support_idx]
    values = one_hot_values(support_labels, NUM_CLASSES)

    rng = random.Random(7)
    gallery = [int(rng.choice(np.flatnonzero(yte == c))) for c in range(NUM_CLASSES)]

    STATE.update(
        model=model,
        text_feats=text_feats.to(DEVICE),
        logit_scale=logit_scale,
        K_finetuned=ckpt["cache_keys"].to(DEVICE),
        K_trainfree=train_bank[support_idx].to(DEVICE),
        V=values.to(DEVICE),
        alpha_ta=float(tip_hp["alpha"]), beta_ta=float(tip_hp["beta"]),
        alpha_f=float(ckpt["alpha"]), beta_f=float(ckpt["beta"]),
        train_ds=train_ds, test_ds=test_ds,
        support_idx=support_idx, support_labels=support_labels,
        gallery=gallery,
        meta=dict(
            clip="OpenAI CLIP ViT-B/32 (frozen, 151,277,313 params)",
            checkpoint=CKPT_PATH.name,
            finetuned=dict(lr=ckpt["lr"], wd=ckpt["wd"], best_epoch=ckpt["best_epoch"],
                           val_acc=ckpt["val_acc"]),
            test_acc_16shot_seed42=test_acc,
            recorded_test_accuracy="Tip-Adapter-F 90.12 +/- 0.59 (mean over seeds) at 16-shot",
        ),
    )
    print(f"[demo] ready. Tip-Adapter-F: alpha={STATE['alpha_f']:.2f} beta={STATE['beta_f']:.2f} | "
          f"Tip-Adapter: alpha={STATE['alpha_ta']:.2f} beta={STATE['beta_ta']:.2f}", flush=True)


def to_thumbnail_b64(pil_img, size=192):
    img = pil_img.convert("RGB").copy()
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def dataset_thumbnail(ds, idx, size=128):
    img = ds[idx][0].convert("RGB")
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def predict_pil(pil_img):
    t0 = time.time()
    x = clip_image_transform()(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        q = STATE["model"].encode_image(x).float()
        q = q / q.norm(dim=-1, keepdim=True)
        clip_l = clip_logits(q, STATE["text_feats"], STATE["logit_scale"])
        logits = {
            "zero_shot": clip_l,
            "tip_adapter": clip_l + STATE["alpha_ta"] * cache_logits(
                q, STATE["K_trainfree"], STATE["V"], STATE["beta_ta"]),
            "tip_adapter_f": clip_l + STATE["alpha_f"] * cache_logits(
                q, STATE["K_finetuned"], STATE["V"], STATE["beta_f"]),
        }
        sims = (q @ STATE["K_finetuned"].t())[0]

    methods = {}
    for name, lg in logits.items():
        probs = torch.softmax(lg[0], dim=0)
        top = torch.topk(probs, 5)
        methods[name] = dict(
            pred=CLASSES[int(top.indices[0])],
            conf=float(probs[top.indices[0]]),
            top5=[dict(name=CLASSES[int(i)], p=float(probs[i])) for i in top.indices],
        )
    nearest = []
    for i in torch.topk(sims, 3).indices.tolist():
        nearest.append(dict(
            train_idx=int(STATE["support_idx"][i]),
            label=CLASSES[int(STATE["support_labels"][i])],
            sim=float(sims[i]),
            kind="fine-tuned keys",
        ))
    return dict(methods=methods, nearest_support=nearest,
                thumbnail=to_thumbnail_b64(pil_img),
                latency_ms=int((time.time() - t0) * 1000))


app = Flask(__name__, static_folder=None)


@app.route("/")
def index():
    return send_from_directory(ROOT / "demo" / "static", "index.html")


@app.route("/api/health")
def health():
    return jsonify(dict(status="ok", device=str(DEVICE), classes=CLASSES, meta=STATE["meta"],
                        alpha_beta=dict(tip_adapter_f=dict(alpha=STATE["alpha_f"], beta=STATE["beta_f"]),
                                        tip_adapter=dict(alpha=STATE["alpha_ta"], beta=STATE["beta_ta"]))))


@app.route("/api/gallery")
def gallery():
    out = [dict(test_idx=i, label=CLASSES[c]) for c, i in enumerate(STATE["gallery"])]
    return jsonify(out)


@app.route("/api/image/<int:idx>")
def test_image(idx):
    return send_file(io.BytesIO(dataset_thumbnail(STATE["test_ds"], idx)), mimetype="image/jpeg")


@app.route("/api/train_image/<int:idx>")
def train_image(idx):
    return send_file(io.BytesIO(dataset_thumbnail(STATE["train_ds"], idx)), mimetype="image/jpeg")


@app.route("/api/predict_test/<int:idx>")
def predict_test(idx):
    return jsonify(dict(test_idx=idx, true_label=CLASSES[int(STATE["test_ds"].targets[idx])],
                        **predict_pil(STATE["test_ds"][idx][0])))


@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify(error="no file uploaded (field name: 'file')"), 400
    try:
        img = Image.open(request.files["file"].stream)
    except Exception as e:
        return jsonify(error=f"cannot read image: {e}"), 400
    return jsonify(predict_pil(img))


@app.route("/api/my_images")
def my_images():
    """The user's own collection: demo/my_images/<class>/<file> (scanned live,
    so files dropped into the folders appear on page refresh)."""
    groups = []
    for c, cls in enumerate(CLASSES):
        files = _valid_images_in(MY_IMAGES_DIR / cls)
        groups.append(dict(label=cls, class_idx=c, images=[dict(
            name=f,
            url=f"/api/my_image/{cls}/{quote(f, safe='')}",
            predict_url=f"/api/predict_my/{cls}/{quote(f, safe='')}",
        ) for f in files]))
    return jsonify(groups)


@app.route("/api/my_image/<cls>/<fname>")
def my_image(cls, fname):
    p = _my_image_path(cls, fname)
    if p is None:
        return jsonify(error="not found"), 404
    img = Image.open(p).convert("RGB")
    img.thumbnail((160, 160))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return send_file(io.BytesIO(buf.getvalue()), mimetype="image/jpeg")


@app.route("/api/predict_my/<cls>/<fname>")
def predict_my(cls, fname):
    """Predict a user-owned photo. Any resolution/format works -- the CLIP
    transform resizes to 224x224 inside predict_pil."""
    p = _my_image_path(cls, fname)
    if p is None:
        return jsonify(error="not found"), 404
    return jsonify(dict(source="my_images", filename=fname, true_label=cls,
                        **predict_pil(Image.open(p))))


if __name__ == "__main__":
    load_everything()
    print(f"[demo] serving on http://127.0.0.1:{PORT}", flush=True)
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
