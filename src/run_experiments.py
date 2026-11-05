# =============================================================================
# run_experiments.py -- end-to-end orchestration
#
#   Stage 1  data + frozen CLIP feature banks (cached on disk)
#   Stage 2  4 methods x 5 shot settings x 3 seeds = 60 evaluated runs
#   Stage 3  ablations (affinity function; cache contribution; fine-tuning)
#   Stage 4  figures (PNG)
#   Stage 5  summary tables (CSV + console)
#
# Shared-namespace loader: locally this file execs the other modules into its
# own globals; inside the Kaggle notebook the same code is placed in cells, the
# files are absent, and the loader is a no-op.
# =============================================================================
import json
import os
import time

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()


def _load_shared(name):
    path = os.path.join(_HERE, name)
    if os.path.exists(path):
        exec(compile(open(path, encoding="utf-8").read(), path, "exec"), globals())
    else:
        print(f"[run_experiments] {name} not on disk -- assuming notebook mode (names already defined).")


for _name in ("config.py", "data.py", "clip_features.py", "methods.py",
              "evaluation.py", "ablations.py", "visualize.py"):
    _load_shared(_name)

import numpy as np
import torch

# NOTE: after the exec loop above, every function/constant from the project
# modules is available as a plain global (shared-namespace design), so no
# project imports are needed here.


def set_all_seeds(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_environment():
    print("=" * 78)
    print("ENVIRONMENT")
    print("=" * 78)
    print(f"device: {DEVICE} | torch {torch.__version__} | mock mode: {MOCK}")
    if DEVICE.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name} | VRAM: {p.total_memory / 1e9:.1f} GB")
    try:
        import torchvision
        print(f"torchvision {torchvision.__version__}")
    except Exception:
        pass
    try:
        import open_clip
        print(f"open_clip {open_clip.__version__}")
    except Exception:
        print("open_clip: not installed (transformers fallback will be used)")


def stage_data_and_model():
    print("\n" + "=" * 78)
    print("STAGE 1 -- DATA + FROZEN CLIP FEATURE BANKS")
    print("=" * 78)
    ensure_dirs()
    set_all_seeds(SEEDS[0])
    train_ds, test_ds, ytr, yte = load_datasets()
    print(f"CIFAR-10: train {len(ytr)} | test {len(yte)} | classes {len(CLASSES)}")

    clip_source, img_params = "mock", 0
    if MOCK:
        model, tokenize, logit_scale = None, None, 100.0
    else:
        model, tokenize, logit_scale, clip_source = load_clip_model()
        named = [(n, p) for n, p in model.named_parameters()]
        img_params = sum(p.numel() for n, p in named if n.startswith("visual."))
        total = sum(p.numel() for p in model.parameters())
        frozen = all(not p.requires_grad for p in model.parameters())
        print(f"CLIP parameters: total {total:,} | all frozen (requires_grad=False): {frozen}")
        if img_params:
            print(f"  image encoder ~{img_params:,} | text encoder ~{total - img_params:,}")

    train_bank, test_bank, text_feats, logit_scale = build_feature_banks(
        model, tokenize, logit_scale, train_ds, test_ds, ytr, yte)
    for nm, m in (("train", train_bank), ("test", test_bank), ("text", text_feats)):
        err = float((m.norm(dim=-1) - 1).abs().max())
        assert err < 1e-4, f"{nm} bank rows are not unit-normalized (max err {err})"
    print(f"train bank {tuple(train_bank.shape)} | test bank {tuple(test_bank.shape)} | "
          f"text {tuple(text_feats.shape)} | logit_scale {logit_scale:.2f}")

    if not MOCK:
        print(f"CLIP frozen check (all requires_grad=False): "
              f"{all(not p.requires_grad for p in model.parameters())}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Explicit verification of what is trainable in Tip-Adapter-F (spec item)
    demo_keys = torch.randn(16, train_bank.shape[1])
    demo_vals = one_hot_values([i % NUM_CLASSES for i in range(16)], NUM_CLASSES)
    demo = TipAdapterF(demo_keys, demo_vals, text_feats, logit_scale, alpha=1.0, beta=5.86)
    tnames, tcount = demo.trainable_summary()
    print(f"Tip-Adapter-F trainable parameter names: {tnames}")
    print(f"Tip-Adapter-F trainable parameter count (demo, 16 cache keys): "
          f"{tcount} = 16 x {train_bank.shape[1]}")
    print("Everything else (cache values, text embeddings, both CLIP encoders) "
          "is frozen by construction.")
    return dict(train_bank=train_bank, test_bank=test_bank, text_feats=text_feats,
                logit_scale=logit_scale, ytr=ytr, yte=yte, clip_source=clip_source)


def dump_run_config(ctx):
    cfg = dict(seeds=SEEDS, shots=SHOTS, val_per_class=VAL_PER_CLASS,
               num_classes=NUM_CLASSES, classes=CLASSES, templates=TEMPLATES,
               alpha_grid=ALPHA_GRID, beta_grid=BETA_GRID,
               lp_lr_grid=LP_LR_GRID, lp_wd_grid=LP_WD_GRID, lp_epochs=LP_EPOCHS,
               tipf_lr_grid=TIPF_LR_GRID, tipf_wd_grid=TIPF_WD_GRID,
               tipf_max_epochs=TIPF_MAX_EPOCHS, tipf_patience=TIPF_PATIENCE,
               clip_source=ctx["clip_source"], logit_scale=ctx["logit_scale"],
               feature_dim=int(ctx["train_bank"].shape[1]),
               torch=torch.__version__, mock=bool(MOCK))
    try:
        import torchvision
        cfg["torchvision"] = torchvision.__version__
    except Exception:
        pass
    try:
        import open_clip
        cfg["open_clip"] = open_clip.__version__
    except Exception:
        pass
    with open(os.path.join(RESULTS_DIR, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"[config] run configuration saved to {os.path.join(RESULTS_DIR, 'run_config.json')}")


def make_record(method, shots, seed, metrics, hp=None, train_time_s=0.0,
                trainable_params=0, epochs_run=0, best_epoch=0, val_top1=None,
                history=None):
    rec = dict(method=method, shots=int(shots), seed=int(seed),
               top1=metrics["top1"], top5=metrics["top5"],
               macro_precision=metrics["macro_precision"],
               macro_recall=metrics["macro_recall"], macro_f1=metrics["macro_f1"],
               per_class_accuracy=metrics["per_class_accuracy"],
               confusion_matrix=metrics["confusion_matrix"],
               support_size=int(shots) * NUM_CLASSES,
               train_time_s=float(train_time_s), trainable_params=int(trainable_params),
               epochs_run=int(epochs_run), best_epoch=int(best_epoch),
               val_top1=None if val_top1 is None else float(val_top1),
               hp=hp or {})
    if history is not None:
        rec["history"] = history
    return rec


def run_one(shots, seed, ctx, verbose=True):
    """One (shots, seed) run: all four methods on the SAME support samples."""
    train_bank, test_bank, text_feats = ctx["train_bank"], ctx["test_bank"], ctx["text_feats"]
    logit_scale, ytr, yte = ctx["logit_scale"], ctx["ytr"], ctx["yte"]

    support_idx, val_idx = make_few_shot_splits(ytr, shots, seed)
    support_feats = train_bank[support_idx]
    support_labels = ytr[support_idx]
    support_values = one_hot_values(support_labels, NUM_CLASSES)
    val_feats, val_labels = train_bank[val_idx], ytr[val_idx]
    test_feats, test_labels = test_bank, yte

    test_clip_l = clip_logits(test_feats, text_feats, logit_scale)
    val_clip_l = clip_logits(val_feats, text_feats, logit_scale)
    n_support, dim = support_feats.shape
    if verbose:
        print(f"\n--- shots={shots} seed={seed} | support={n_support} images "
              f"({shots}/class) | val={len(val_idx)} images ---")
    recs = []

    # ---- 1) Zero-shot CLIP ---------------------------------------------------
    t0 = time.time()
    zs_logits, zs_pred = zero_shot_predict(test_feats, text_feats, logit_scale)
    zs_time = time.time() - t0
    m = compute_metrics(test_labels, zs_pred.numpy(), zs_logits.numpy())
    recs.append(make_record("zero_shot", shots, seed, m, hp={}, train_time_s=zs_time))
    if verbose:
        print(f"  zero-shot     : test top1 {m['top1'] * 100:.2f}% | trainable params 0")

    # ---- 2) Linear probe -------------------------------------------------------
    head, lp_best, lp_time = train_linear_probe(
        support_feats, torch.as_tensor(support_labels), val_feats,
        torch.as_tensor(val_labels), seed=seed, verbose=False)
    with torch.no_grad():
        lp_logits = head(test_feats.to(DEVICE)).cpu()
    m = compute_metrics(test_labels, lp_logits.argmax(1).numpy(), lp_logits.numpy())
    lp_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    recs.append(make_record("linear_probe", shots, seed, m,
                            hp=dict(lr=lp_best["lr"], wd=lp_best["wd"]),
                            train_time_s=lp_time, trainable_params=lp_params,
                            epochs_run=LP_EPOCHS, best_epoch=lp_best["best_epoch"],
                            val_top1=lp_best["val_acc"]))
    if verbose:
        print(f"  linear-probe  : test top1 {m['top1'] * 100:.2f}% | val {lp_best['val_acc'] * 100:.2f}% "
              f"| trainable params {lp_params} | time {lp_time:.1f}s")

    # ---- 3) Tip-Adapter (training-free) ---------------------------------------
    t0 = time.time()
    best_free, sweep = tip_adapter_free_search(
        val_feats, torch.as_tensor(val_labels), support_feats, support_values,
        val_clip_l, verbose=verbose)
    search_time = time.time() - t0
    tip_logits_test = test_clip_l + best_free["alpha"] * cache_logits(
        test_feats, support_feats, support_values, best_free["beta"])
    m = compute_metrics(test_labels, tip_logits_test.argmax(1).numpy(), tip_logits_test.numpy())
    recs.append(make_record("tip_adapter", shots, seed, m, hp=best_free,
                            train_time_s=search_time, val_top1=best_free["val_acc"]))
    if verbose:
        print(f"  tip-adapter   : test top1 {m['top1'] * 100:.2f}% | alpha={best_free['alpha']:.2f} "
              f"beta={best_free['beta']:.2f} | trainable params 0 | search {search_time:.2f}s")
    sweep_save = sweep if (shots == 16 and seed == SEEDS[0]) else None

    # ---- 4) Tip-Adapter-F -------------------------------------------------------
    ckpt = os.path.join(CKPT_DIR, f"tip_adapter_f_{shots}shot_seed{seed}.pt")
    init_alpha = max(float(best_free["alpha"]), 0.5)   # guard: keep gradient flowing to keys
    res = run_tip_adapter_f(
        support_feats, torch.as_tensor(support_labels), support_values, val_feats,
        torch.as_tensor(val_labels), text_feats, logit_scale,
        init_alpha=init_alpha, beta=float(best_free["beta"]), seed=seed, ckpt_path=ckpt)
    tf_logits, tf_pred = tip_f_predict(test_feats, res["best_keys"], support_values,
                                       text_feats, logit_scale, res["final_alpha"],
                                       float(best_free["beta"]))
    m = compute_metrics(test_labels, tf_pred.numpy(), tf_logits.numpy())
    tf_params = n_support * dim
    recs.append(make_record("tip_adapter_f", shots, seed, m,
                            hp=dict(lr=res["lr"], wd=res["wd"], init_alpha=init_alpha,
                                    beta=float(best_free["beta"]), final_alpha=res["final_alpha"],
                                    max_epochs=TIPF_MAX_EPOCHS, patience=TIPF_PATIENCE),
                            train_time_s=res["train_time"], trainable_params=tf_params,
                            epochs_run=res["epochs_run"], best_epoch=res["best_epoch"],
                            val_top1=res["val_acc_after_alpha_search"],
                            history=res["history"]))
    if verbose:
        print(f"  tip-adapter-F : test top1 {m['top1'] * 100:.2f}% | val {res['val_acc_after_alpha_search'] * 100:.2f}% "
              f"| lr={res['lr']:g} wd={res['wd']:g} final_alpha={res['final_alpha']:.2f} "
              f"| best epoch {res['best_epoch']}/{res['epochs_run']} | time {res['train_time']:.1f}s "
              f"| trainable params {tf_params} ({n_support} keys x {dim})")
    return recs, sweep_save


def stage_experiments(ctx):
    print("\n" + "=" * 78)
    print("STAGE 2 -- FEW-SHOT EXPERIMENTS (4 methods x 5 shots x 3 seeds)")
    print("=" * 78)
    records, sweep_16 = [], None
    total = len(SHOTS) * len(SEEDS)
    done, t_start = 0, time.time()
    for shots in SHOTS:
        for seed in SEEDS:
            recs, sweep = run_one(shots, seed, ctx)
            records.extend(recs)
            if sweep is not None:
                sweep_16 = sweep
                save_records(sweep, os.path.join(RESULTS_DIR, "alpha_beta_sweep_16shot_seed42.json"))
            done += 1
            save_records(records, os.path.join(RESULTS_DIR, "records.json"))
            print(f"[progress] {done}/{total} runs done | elapsed {time.time() - t_start:.0f}s", flush=True)
    return records


def stage_ablations(ctx):
    print("\n" + "=" * 78)
    print("STAGE 3 -- ABLATIONS")
    print("=" * 78)
    rows = ablate_affinity(ctx)
    save_records(rows, os.path.join(RESULTS_DIR, "ablation_affinity.json"))
    for r in rows:
        print(f"  {r['shots']:>2}-shot | exp  (alpha={r['exp_alpha']:.2f}, beta={r['exp_beta']:.2f}) "
              f"-> test {r['exp_top1'] * 100:.2f}%   |   cosine (alpha={r['cos_alpha']:.2f}) "
              f"-> test {r['cos_top1'] * 100:.2f}%")
    return rows


def stage_visualize(records, affinity_rows, sweep_16):
    print("\n" + "=" * 78)
    print("STAGE 4 -- FIGURES")
    print("=" * 78)
    make_all_figures(records, affinity_rows, sweep_16, FIG_DIR)


def stage_summary(records, affinity_rows):
    print("\n" + "=" * 78)
    print("STAGE 5 -- SUMMARY TABLES")
    print("=" * 78)
    df = summarize_records(records)
    df.to_csv(os.path.join(RESULTS_DIR, "records.csv"), index=False)

    order = [m for m in ("zero_shot", "linear_probe", "tip_adapter", "tip_adapter_f")]
    mean, std = pivot_mean_std(df, "top1", "method", "shots")
    mean, std = mean.loc[order], std.loc[order]

    print(f"\nTop-1 accuracy (%) on the official CIFAR-10 test set -- mean +/- std "
          f"over seeds {SEEDS}")
    tbl = pd.DataFrame(index=mean.index, columns=mean.columns, dtype=object)
    for c in mean.columns:
        tbl[c] = [f"{m * 100:.2f} +/- {s * 100:.2f}"
                  for m, s in zip(mean[c], std[c].fillna(0))]
    tbl.index = [METHOD_LABELS[i] for i in tbl.index]
    print(tbl.to_string())
    tbl.to_csv(os.path.join(RESULTS_DIR, "summary_top1_mean_std.csv"))

    d16 = df[df.shots == 16]
    show = d16.groupby("method")[["top1", "top5", "macro_precision", "macro_recall",
                                  "macro_f1"]].mean().loc[order] * 100
    print("\nFull test metrics at 16-shot (mean over seeds, %):")
    print(show.round(2).to_string())
    aux = d16.groupby("method")[["trainable_params", "train_time_s"]].mean().loc[order]
    print("\nTrainable parameters / training time at 16-shot (mean over seeds):")
    print(aux.round(2).to_string())

    cache_tbl = pd.DataFrame(derive_cache_and_finetune_table(df))
    cache_tbl.to_csv(os.path.join(RESULTS_DIR, "ablation_cache_and_finetune.csv"), index=False)
    print("\nAblation A (cache contribution) and C (fine-tuning effect), test top-1 %:")
    print(cache_tbl.round(2).to_string(index=False))

    aff = pd.DataFrame(affinity_rows)
    print("\nAblation B (affinity function), seed 42, test top-1 %:")
    print(aff[["shots", "cos_top1", "exp_top1"]].assign(
        cos_top1=lambda d: d.cos_top1 * 100, exp_top1=lambda d: d.exp_top1 * 100
    ).round(2).to_string(index=False))


def list_outputs():
    print("\nOutput files:")
    for base in (RESULTS_DIR, FIG_DIR, CKPT_DIR):
        if not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            for fn in sorted(files):
                p = os.path.join(root, fn)
                try:
                    print(f"  {p} ({os.path.getsize(p) / 1e6:.2f} MB)")
                except OSError:
                    pass


def run_all():
    t0 = time.time()
    print_environment()
    ctx = stage_data_and_model()
    dump_run_config(ctx)
    records = stage_experiments(ctx)
    affinity_rows = stage_ablations(ctx)
    sweep_path = os.path.join(RESULTS_DIR, "alpha_beta_sweep_16shot_seed42.json")
    sweep_16 = json.load(open(sweep_path, encoding="utf-8")) if os.path.exists(sweep_path) else None
    stage_visualize(records, affinity_rows, sweep_16)
    stage_summary(records, affinity_rows)
    print(f"\nTotal wall-clock: {(time.time() - t0) / 60:.1f} min")
    list_outputs()
    print("\nALL DONE")
    return records


if __name__ == "__main__":
    run_all()
