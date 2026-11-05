# =============================================================================
# ablations.py
#
# A. Cache contribution  : zero-shot CLIP  vs  CLIP + cache (Tip-Adapter).
#    -> measures how much a non-parametric few-shot memory adds to zero-shot.
# B. Affinity function   : raw cosine similarity  vs  exp(-beta*(1-cos)).
#    -> measures the effect of the exponential kernel (aggregation identical).
# C. Fine-tuning effect  : fixed cache keys (Tip-Adapter) vs trainable keys
#    (Tip-Adapter-F) -> measures what gradient descent on the cache buys.
# D. Number of shots     : covered by the main 1/2/4/8/16-shot sweep.
#
# This file is executed into the shared namespace; functions from methods.py /
# data.py / config.py are already defined when it runs.
# =============================================================================
import numpy as np
import torch


def ablate_affinity(ctx, shots_list=None, seeds=(42,)):
    """Ablation B: for each shot setting, run the SAME (alpha, beta) validation
    search twice -- once with the exponential affinity and once with raw cosine
    similarity -- then evaluate both on the test set. Aggregation (affinity @
    one-hot values, added to CLIP logits) is identical in both arms."""
    if shots_list is None:
        shots_list = globals().get("SHOTS") or [1, 2, 4, 8, 16]
    train_bank, test_bank, text_feats = ctx["train_bank"], ctx["test_bank"], ctx["text_feats"]
    logit_scale, ytr, yte = ctx["logit_scale"], ctx["ytr"], ctx["yte"]
    test_clip_l = clip_logits(test_bank, text_feats, logit_scale)

    rows = []
    for shots in shots_list:
        for seed in seeds:
            support_idx, val_idx = make_few_shot_splits(ytr, shots, seed)
            support_feats = train_bank[support_idx]
            support_values = one_hot_values(ytr[support_idx], NUM_CLASSES)
            val_feats = train_bank[val_idx]
            val_labels = torch.as_tensor(ytr[val_idx], dtype=torch.long)
            val_clip_l = clip_logits(val_feats, text_feats, logit_scale)

            best_exp, _ = tip_adapter_free_search(val_feats, val_labels, support_feats,
                                                  support_values, val_clip_l,
                                                  kind="exp", verbose=False)
            best_cos, _ = tip_adapter_free_search(val_feats, val_labels, support_feats,
                                                  support_values, val_clip_l,
                                                  kind="cosine", verbose=False)
            exp_logits = test_clip_l + best_exp["alpha"] * cache_logits(
                test_bank, support_feats, support_values, best_exp["beta"], "exp")
            cos_logits = test_clip_l + best_cos["alpha"] * cache_logits(
                test_bank, support_feats, support_values, None, "cosine")
            rows.append(dict(
                shots=int(shots), seed=int(seed),
                exp_alpha=best_exp["alpha"], exp_beta=best_exp["beta"],
                exp_val=best_exp["val_acc"],
                exp_top1=float((exp_logits.argmax(1).numpy() == yte).mean()),
                cos_alpha=best_cos["alpha"], cos_val=best_cos["val_acc"],
                cos_top1=float((cos_logits.argmax(1).numpy() == yte).mean())))
    return rows


def derive_cache_and_finetune_table(df):
    """Ablations A and C, derived from the main experiment records:
       A: zero_shot -> tip_adapter        (adding the training-free cache)
       C: tip_adapter -> tip_adapter_f    (making the cache keys trainable)"""
    rows = []
    for shots in sorted(df["shots"].unique()):
        def mean_acc(method):
            sel = df[(df.method == method) & (df.shots == shots)]
            return float(sel["top1"].mean()) * 100
        rows.append(dict(
            shots=int(shots),
            zero_shot=mean_acc("zero_shot"),
            clip_plus_cache=mean_acc("tip_adapter"),
            clip_plus_trained_cache=mean_acc("tip_adapter_f"),
            ablationA_cache_gain=mean_acc("tip_adapter") - mean_acc("zero_shot"),
            ablationC_finetune_gain=mean_acc("tip_adapter_f") - mean_acc("tip_adapter")))
    return rows
