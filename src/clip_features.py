# =============================================================================
# clip_features.py -- frozen CLIP loading, feature extraction, disk caching
#
# Normalization strategy (documented once, used everywhere):
#   * image features : forward through FROZEN image encoder, then L2-normalize
#   * text features  : per-template L2-normalize -> average over templates ->
#                      re-normalize per class (standard CLIP prompt ensembling)
#   * cache keys     : initialized from L2-normalized support image features;
#                      NOT re-normalized during fine-tuning (matches the
#                      official Tip-Adapter implementation; the keys drift
#                      freely so they can point to better class prototypes)
#   * queries        : always L2-normalized frozen CLIP features (never updated)
#
# All four methods consume the SAME cached feature banks, so the frozen CLIP
# model is only run over the images once per session.
# =============================================================================
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

try:
    from config import (BATCH_SIZE, CACHE_DIR, CLASSES, CLIP_ARCH, CLIP_HF_FALLBACK,
                        CLIP_PRETRAINED, DEVICE, MOCK, NUM_CLASSES, NUM_WORKERS,
                        TEMPLATES, USE_AMP)
except ImportError:
    pass  # shared-namespace mode (notebook): names already defined by config cell


class _HFTextImageWrapper(nn.Module):
    """Minimal adapter so a HuggingFace CLIPModel exposes the same
    encode_image / encode_text / logit_scale interface as open_clip."""

    def __init__(self, hf):
        super().__init__()
        self.hf = hf

    def encode_image(self, images):
        return self.hf.get_image_features(pixel_values=images)

    def encode_text(self, input_ids):
        return self.hf.get_text_features(input_ids=input_ids)

    @property
    def logit_scale(self):
        return self.hf.logit_scale


def load_clip_model(device=DEVICE):
    """Load OpenAI CLIP ViT-B/32, freeze every parameter, return
    (model, tokenize_fn, logit_scale, source_description)."""
    try:
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(CLIP_ARCH, pretrained=CLIP_PRETRAINED)
        tokenize = open_clip.get_tokenizer(CLIP_ARCH)
        source = f"open_clip:{CLIP_ARCH}:{CLIP_PRETRAINED}"
    except Exception as e:  # pragma: no cover - fallback for restricted networks
        print(f"[load_clip_model] open_clip failed ({e}); falling back to HuggingFace transformers.")
        from transformers import CLIPModel, CLIPTokenizer
        tok = CLIPTokenizer.from_pretrained(CLIP_HF_FALLBACK)
        hf = CLIPModel.from_pretrained(CLIP_HF_FALLBACK)
        model = _HFTextImageWrapper(hf)
        tokenize = (lambda prompts: tok(prompts, padding="max_length", max_length=77,
                                        truncation=True, return_tensors="pt").input_ids)
        source = f"transformers:{CLIP_HF_FALLBACK}"

    for p in model.parameters():          # freeze the entire CLIP model
        p.requires_grad_(False)
    model = model.to(device).eval()
    with torch.no_grad():
        logit_scale = float(model.logit_scale.exp().item())
    print(f"[load_clip_model] source={source} | logit_scale={logit_scale:.2f} | device={device}")
    return model, tokenize, logit_scale, source


@torch.no_grad()
def extract_image_features(model, dataset, indices, device=DEVICE,
                           batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                           use_amp=USE_AMP, desc=""):
    """Forward images through the FROZEN image encoder; return L2-normalized
    float32 features on CPU, shape [len(indices), 512]."""
    model.eval()
    idx_list = indices.tolist() if isinstance(indices, np.ndarray) else list(indices)
    loader = DataLoader(Subset(dataset, idx_list), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"))
    feats = []
    for k, (images, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        if use_amp and device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                f = model.encode_image(images)
        else:
            f = model.encode_image(images)
        feats.append(f.float().cpu())
        if desc and k % 20 == 0:
            print(f"  [{desc}] batch {k + 1}/{len(loader)}", flush=True)
    feats = torch.cat(feats, dim=0)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


@torch.no_grad()
def encode_text_classes(model, tokenize, classes=CLASSES, templates=TEMPLATES, device=DEVICE):
    """Prompt-ensemble text embeddings: average of per-template normalized
    embeddings, re-normalized. Returns [num_classes, 512] float32 on CPU."""
    per_class = []
    for c in classes:
        prompts = [t.format(c) for t in templates]
        toks = tokenize(prompts).to(device)
        f = model.encode_text(toks).float()
        f = f / f.norm(dim=-1, keepdim=True)
        f = f.mean(dim=0)
        f = f / f.norm(dim=-1, keepdim=True)
        per_class.append(f.cpu())
    return torch.stack(per_class)


def build_feature_banks(model, tokenize, logit_scale, train_ds, test_ds, ytr, yte,
                        device=DEVICE, cache_dir=CACHE_DIR, mock=MOCK):
    """Extract (or load from disk cache) three feature banks shared by ALL methods:
         train_bank [50000, 512]  test_bank [10000, 512]  text_feats [10, 512]"""
    os.makedirs(cache_dir, exist_ok=True)
    paths = {k: os.path.join(cache_dir, f) for k, f in [
        ("train", "train_bank.pt"), ("test", "test_bank.pt"),
        ("text", "text_feats.pt"), ("scale", "logit_scale.pt")]}
    if all(os.path.exists(p) for p in paths.values()):
        print("[features] cached feature banks found on disk -- loading (skips image encoding)")
        return (torch.load(paths["train"]), torch.load(paths["test"]),
                torch.load(paths["text"]), float(torch.load(paths["scale"])))

    if mock:
        print("[features] MOCK mode: generating deterministic random feature banks")
        g = torch.Generator().manual_seed(7)
        train_bank = torch.randn(len(ytr), 512, generator=g)
        train_bank = train_bank / train_bank.norm(dim=-1, keepdim=True)
        test_bank = torch.randn(len(yte), 512, generator=g)
        test_bank = test_bank / test_bank.norm(dim=-1, keepdim=True)
        gt = torch.Generator().manual_seed(99)
        text_feats = torch.randn(NUM_CLASSES, 512, generator=gt)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        logit_scale = 100.0
    else:
        t0 = time.time()
        print(f"[features] encoding {len(ytr)} CIFAR-10 TRAIN images with frozen CLIP ...")
        train_bank = extract_image_features(model, train_ds, np.arange(len(ytr)),
                                            device=device, desc="train")
        print(f"[features]   train bank done in {time.time() - t0:.0f}s")
        t0 = time.time()
        print(f"[features] encoding {len(yte)} CIFAR-10 TEST images with frozen CLIP ...")
        test_bank = extract_image_features(model, test_ds, np.arange(len(yte)),
                                           device=device, desc="test")
        print(f"[features]   test bank done in {time.time() - t0:.0f}s")
        text_feats = encode_text_classes(model, tokenize, device=device)

    torch.save(train_bank, paths["train"])
    torch.save(test_bank, paths["test"])
    torch.save(text_feats, paths["text"])
    torch.save(torch.tensor(float(logit_scale)), paths["scale"])
    print(f"[features] banks cached to {cache_dir}")
    return train_bank, test_bank, text_feats, float(logit_scale)
