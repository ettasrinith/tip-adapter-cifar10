# =============================================================================
# methods.py -- the four adaptation methods
#
#   1) Zero-shot CLIP        : argmax over logit_scale * cos(img, class-text)
#   2) Linear Probe          : nn.Linear(512, 10) on frozen features, trained
#                              with cross-entropy on the support set
#   3) Tip-Adapter           : training-free cache model
#   4) Tip-Adapter-F         : same cache, but keys are nn.Parameter and
#                              optimized with AdamW + cross-entropy
#
# Tip-Adapter cache model (paper equations):
#   cache_keys K in R^(N x 512)  = L2-normalized support image features
#   cache_values V in {0,1}^(N x 10) = one-hot labels of the support images
#   affinity   A(q, k) = exp(-beta * (1 - cos(q, k)))          in (0, 1]
#   cache_logits(q)    = A @ V                                  [Q, 10]
#   final_logits(q)    = clip_logits(q) + alpha * cache_logits(q)
#
# Role of the two hyperparameters:
#   * beta -- sharpness of the affinity. Large beta -> only cache keys that are
#     nearly identical to the query contribute (nearest-neighbour behaviour);
#     beta -> 0 makes affinity nearly uniform (1 for every key).
#   * alpha -- global weight of the cache branch relative to CLIP's zero-shot
#     logits. alpha = 0 recovers pure zero-shot CLIP.
#
# Both are tuned on the VALIDATION set only, never on test accuracy.
# =============================================================================
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from config import (ALPHA_GRID, BETA_GRID, DEVICE, LP_EVAL_EVERY, LP_EPOCHS,
                        LP_LR_GRID, LP_WD_GRID, NUM_CLASSES, TIPF_LR_GRID,
                        TIPF_MAX_EPOCHS, TIPF_PATIENCE, TIPF_WD_GRID)
except ImportError:
    pass  # shared-namespace mode (notebook): names already defined by config cell


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------
def clip_logits(feats, text_feats, logit_scale):
    """CLIP zero-shot logits: logit_scale * cos(img, class_text).
    `feats` [B, 512] and `text_feats` [10, 512] must be L2-normalized."""
    return logit_scale * (feats @ text_feats.t())


def affinity_matrix(query_feats, cache_keys, beta=None, kind="exp"):
    """Affinity between query features and cache keys.
    kind='exp'    : exp(-beta * (1 - cos))  -- the Tip-Adapter affinity.
    kind='cosine' : raw cosine similarity    -- used only in ablation B.
    """
    sim = query_feats @ cache_keys.t()          # cosine similarity, [Q, N]
    if kind == "exp":
        return torch.exp(-beta * (1.0 - sim))
    if kind == "cosine":
        return sim
    raise ValueError(f"unknown affinity kind: {kind}")


def cache_logits(query_feats, cache_keys, cache_values, beta=None, kind="exp"):
    """cache_logits = affinity @ one-hot values -> [Q, 10] class scores."""
    return affinity_matrix(query_feats, cache_keys, beta, kind) @ cache_values


def one_hot_values(labels, num_classes=NUM_CLASSES):
    """cache_values: [N, num_classes] one-hot label matrix."""
    return F.one_hot(torch.as_tensor(labels, dtype=torch.long), num_classes).float()


def zero_shot_predict(feats, text_feats, logit_scale):
    logits = clip_logits(feats, text_feats, logit_scale)
    return logits, logits.argmax(dim=1)


# ---------------------------------------------------------------------------
# Method 2: linear probe on frozen CLIP features
# ---------------------------------------------------------------------------
def train_linear_probe(support_feats, support_labels, val_feats, val_labels,
                       num_classes=NUM_CLASSES, lr_grid=LP_LR_GRID, wd_grid=LP_WD_GRID,
                       epochs=LP_EPOCHS, eval_every=LP_EVAL_EVERY, device=DEVICE, seed=0,
                       verbose=True):
    """Full-batch linear head on frozen features. Grid-search (lr, weight_decay);
    for each config keep the epoch with the best validation accuracy. The best
    config is selected by validation accuracy (ties -> lower val loss)."""
    sf = support_feats.to(device)
    y = torch.as_tensor(support_labels, dtype=torch.long, device=device)
    vf = val_feats.to(device)
    vy = torch.as_tensor(val_labels, dtype=torch.long, device=device)

    best, t0 = None, time.time()
    for lr in lr_grid:
        for wd in wd_grid:
            torch.manual_seed(seed)
            head = nn.Linear(sf.shape[1], num_classes).to(device)
            opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
            best_ep, best_vacc, best_vloss, best_state = 0, -1.0, float("inf"), None
            for ep in range(1, epochs + 1):
                head.train()
                opt.zero_grad()
                loss = F.cross_entropy(head(sf), y)
                loss.backward()
                opt.step()
                if ep % eval_every == 0 or ep == epochs:
                    head.eval()
                    with torch.no_grad():
                        vl = head(vf)
                        vloss = F.cross_entropy(vl, vy).item()
                        vacc = (vl.argmax(1) == vy).float().mean().item()
                    if vacc > best_vacc or (vacc == best_vacc and vloss < best_vloss):
                        best_ep, best_vacc, best_vloss = ep, vacc, vloss
                        best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            if verbose:
                print(f"  [linear-probe] lr={lr:g} wd={wd:g} -> best val acc {best_vacc:.4f} @ epoch {best_ep}")
            if best is None or best_vacc > best["val_acc"] or \
                    (best_vacc == best["val_acc"] and best_vloss < best["val_loss"]):
                best = dict(state=best_state, lr=lr, wd=wd, val_acc=best_vacc,
                            val_loss=best_vloss, best_epoch=best_ep)
    train_time = time.time() - t0
    head = nn.Linear(sf.shape[1], num_classes).to(device)
    head.load_state_dict(best["state"])
    head.eval()
    return head, best, train_time


# ---------------------------------------------------------------------------
# Method 3: Tip-Adapter (training-free)
# ---------------------------------------------------------------------------
def tip_adapter_free_search(val_feats, val_labels, support_feats, support_values,
                            val_clip_logits, alpha_grid=ALPHA_GRID, beta_grid=None,
                            kind="exp", device=DEVICE, verbose=True, label="tip-adapter"):
    """Exhaustive (alpha, beta) grid search on the VALIDATION set.

    Returns (best_cfg dict with alpha/beta/val_acc, full sweep rows)."""
    if beta_grid is None:
        beta_grid = BETA_GRID
    vy = torch.as_tensor(val_labels, dtype=torch.long, device=device)
    sfd = support_feats.to(device)
    svd = support_values.to(device)
    vcl = val_clip_logits.to(device)
    sims_val = val_feats.to(device) @ sfd.t()      # [V, N] cosine similarities

    sweep, best = [], None
    for beta in beta_grid:
        if kind == "exp":
            cache_vl = torch.exp(-beta * (1.0 - sims_val)) @ svd
            alphas = alpha_grid
        else:  # raw cosine affinity: beta is irrelevant
            cache_vl = sims_val @ svd
            alphas = [a for a in alpha_grid if a > 0]
        for alpha in alphas:
            logits = vcl + alpha * cache_vl
            vacc = (logits.argmax(1) == vy).float().mean().item()
            sweep.append(dict(alpha=float(alpha), beta=float(beta), kind=kind, val_acc=vacc))
            if best is None or vacc > best["val_acc"]:
                best = dict(alpha=float(alpha), beta=float(beta), kind=kind, val_acc=vacc)
    if verbose:
        print(f"  [{label}] best ({kind}) on validation: alpha={best['alpha']:.2f} "
              f"beta={best['beta']:.2f} -> val acc {best['val_acc']:.4f}")
    return best, sweep


def tip_adapter_predict(test_feats, support_feats, support_values, text_feats,
                        logit_scale, alpha, beta, kind="exp", device=DEVICE):
    """final_logits = clip_logits + alpha * cache_logits on the test features."""
    tf = test_feats.to(device)
    logits = clip_logits(tf, text_feats.to(device), logit_scale) + \
        alpha * cache_logits(tf, support_feats.to(device), support_values.to(device),
                             beta, kind)
    return logits.cpu(), logits.argmax(1).cpu()


# ---------------------------------------------------------------------------
# Method 4: Tip-Adapter-F (fine-tuned cache keys)
# ---------------------------------------------------------------------------
class TipAdapterF(nn.Module):
    """Cache model with TRAINABLE keys.

    Only `cache_keys` is an nn.Parameter; cache_values / text_feats / logit_scale
    are registered buffers (constants). CLIP's encoders are NOT part of this
    module at all -- their logits are supplied precomputed and constant -- so it
    is impossible to update CLIP weights here by construction.

    Normalization: keys are initialized from L2-normalized support features and
    are NOT re-normalized during training (official implementation behaviour);
    queries are always L2-normalized frozen features."""

    def __init__(self, cache_keys, cache_values, text_feats, logit_scale, alpha, beta):
        super().__init__()
        self.cache_keys = nn.Parameter(cache_keys.clone())          # [N, 512] TRAINABLE
        self.register_buffer("cache_values", cache_values.clone())  # [N, 10] fixed
        self.register_buffer("text_feats", text_feats.clone())      # [10, 512] fixed
        self.register_buffer("logit_scale", torch.tensor(float(logit_scale)))
        self.alpha, self.beta = float(alpha), float(beta)

    def forward(self, query_feats):
        sim = query_feats @ self.cache_keys.t()
        cache_l = torch.exp(-self.beta * (1.0 - sim)) @ self.cache_values
        clip_l = self.logit_scale * (query_feats @ self.text_feats.t())
        return clip_l + self.alpha * cache_l

    def trainable_summary(self):
        names = [n for n, p in self.named_parameters() if p.requires_grad]
        count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return names, count


def train_tip_adapter_f(support_feats, support_labels, support_values, val_feats, val_labels,
                        text_feats, logit_scale, alpha, beta, lr, wd,
                        max_epochs=TIPF_MAX_EPOCHS, patience=TIPF_PATIENCE,
                        device=DEVICE, seed=0):
    """One Tip-Adapter-F training run. Training queries ARE the support features
    (official protocol: the cache is optimized on the few-shot samples it stores).
    Cross-entropy on final_logits; AdamW; early stopping + checkpoint selection
    on VALIDATION accuracy (ties -> lower val loss)."""
    torch.manual_seed(seed)
    model = TipAdapterF(support_feats.to(device), support_values.to(device),
                        text_feats.to(device), logit_scale, alpha, beta).to(device)

    trainable_names, trainable_count = model.trainable_summary()
    assert trainable_names == ["cache_keys"], f"unexpected trainable params: {trainable_names}"
    assert trainable_count == np.prod(support_feats.shape), \
        f"expected {np.prod(support_feats.shape)} trainable values, got {trainable_count}"

    opt = torch.optim.AdamW([model.cache_keys], lr=lr, weight_decay=wd)
    sf = support_feats.to(device)
    y = torch.as_tensor(support_labels, dtype=torch.long, device=device)
    vf = val_feats.to(device)
    vy = torch.as_tensor(val_labels, dtype=torch.long, device=device)

    history = []
    best_vacc, best_vloss, best_epoch, best_state, wait = -1.0, float("inf"), 0, None, 0
    t0 = time.time()
    for ep in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(sf), y)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            vlog = model(vf)
            vloss = F.cross_entropy(vlog, vy).item()
            vacc = (vlog.argmax(1) == vy).float().mean().item()
        history.append(dict(epoch=ep, train_loss=float(loss.item()),
                            val_acc=vacc, val_loss=vloss))

        if vacc > best_vacc or (vacc == best_vacc and vloss < best_vloss):
            best_vacc, best_vloss, best_epoch, wait = vacc, vloss, ep, 0
            best_state = model.cache_keys.detach().clone()
        else:
            wait += 1
            if wait >= patience:
                break
    train_time = time.time() - t0
    return dict(best_keys=best_state, val_acc=best_vacc, val_loss=best_vloss,
                best_epoch=best_epoch, epochs_run=history[-1]["epoch"],
                history=history, train_time=train_time, trainable_count=trainable_count)


def run_tip_adapter_f(support_feats, support_labels, support_values, val_feats, val_labels,
                      text_feats, logit_scale, init_alpha, beta,
                      lr_grid=TIPF_LR_GRID, wd_grid=TIPF_WD_GRID, alpha_grid=ALPHA_GRID,
                      device=DEVICE, seed=0, verbose=True, ckpt_path=None):
    """Full Tip-Adapter-F protocol for one (shots, seed):
      1. grid-search (lr, weight_decay); each config early-stops on validation.
      2. re-search alpha on validation with the best frozen keys (official
         two-stage protocol). beta stays fixed at its validation-tuned value
         from the training-free stage.
    Returns the best result incl. final (keys, alpha, beta) and history."""
    best = None
    for lr in lr_grid:
        for wd in wd_grid:
            res = train_tip_adapter_f(support_feats, support_labels, support_values,
                                      val_feats, val_labels, text_feats, logit_scale,
                                      alpha=init_alpha, beta=beta, lr=lr, wd=wd,
                                      device=device, seed=seed)
            if verbose:
                print(f"  [tip-adapter-F] lr={lr:g} wd={wd:g} -> val acc {res['val_acc']:.4f} "
                      f"(best epoch {res['best_epoch']}/{res['epochs_run']})")
            if best is None or res["val_acc"] > best["val_acc"] or \
                    (res["val_acc"] == best["val_acc"] and res["val_loss"] < best["val_loss"]):
                res = dict(res)
                res.update(lr=lr, wd=wd)
                best = res

    # --- post-training alpha re-search on validation (keys frozen) ---
    keys = best["best_keys"].to(device)
    vals = support_values.to(device)
    vcl = clip_logits(val_feats.to(device), text_feats.to(device), logit_scale)
    sims = val_feats.to(device) @ keys.t()
    best_alpha = None
    for a in alpha_grid:
        logits = vcl + a * (torch.exp(-beta * (1.0 - sims)) @ vals)
        vacc = (logits.argmax(1) == torch.as_tensor(val_labels, dtype=torch.long,
                                                    device=device)).float().mean().item()
        if best_alpha is None or vacc > best_alpha[1]:
            best_alpha = (float(a), vacc)
    best["final_alpha"] = best_alpha[0]
    best["val_acc_after_alpha_search"] = best_alpha[1]
    if verbose:
        print(f"  [tip-adapter-F] post-training alpha search: alpha={best['final_alpha']:.2f} "
              f"-> val acc {best['val_acc_after_alpha_search']:.4f}")

    if ckpt_path is not None:
        torch.save(dict(cache_keys=best["best_keys"].cpu(), alpha=best["final_alpha"],
                        beta=float(beta), lr=best["lr"], wd=best["wd"],
                        init_alpha=float(init_alpha), best_epoch=best["best_epoch"],
                        val_acc=best["val_acc_after_alpha_search"]), ckpt_path)
    return best


def tip_f_predict(test_feats, cache_keys, cache_values, text_feats, logit_scale,
                  alpha, beta, device=DEVICE):
    """Test-time prediction with the fine-tuned keys and re-searched alpha."""
    tf = test_feats.to(device)
    logits = clip_logits(tf, text_feats.to(device), logit_scale) + \
        alpha * cache_logits(tf, cache_keys.to(device), cache_values.to(device), beta, "exp")
    return logits.cpu(), logits.argmax(1).cpu()
