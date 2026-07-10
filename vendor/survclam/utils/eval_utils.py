import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from models.model_clam_family import CLAMFamilySB
from models.model_mil import MIL_fc, MIL_fc_mc

from utils.utils import get_split_loader, print_network
from utils.core_utils import summarize_all  


########################################
# Batch unpacking (supports institution_id)
########################################

def _unpack_batch(batch, task_type: str):
    """
    Supports dataset outputs with optional covariates and optional institution_id.

    Classification:
      - (feats, y)
      - (feats, cov, y)
      - (feats, y, inst_id)
      - (feats, cov, y, inst_id)

    Survival:
      - (feats, surv)
      - (feats, cov, surv)
      - (feats, surv, inst_id)
      - (feats, cov, surv, inst_id)

    Regression:
      - (feats, target)
      - (feats, cov, target)
      - (feats, target, inst_id)
      - (feats, cov, target, inst_id)

    Returns:
      feats, cov_or_none, y, inst_id_or_none
    """
    if not isinstance(batch, (tuple, list)):
        raise ValueError(f"Expected batch tuple/list, got {type(batch)}")

    n = len(batch)
    if n == 2:
        feats, y = batch
        return feats, None, y, None

    if n == 3:
        feats, a, b = batch
        # disambiguate: (feats, cov, y) vs (feats, y, inst_id)
        # inst_id is expected to be integer-like (LongTensor scalar)
        # cov is float tensor vector
        if torch.is_tensor(b) and b.dtype in (torch.int64, torch.long) and b.numel() == 1:
            # (feats, y, inst_id)
            return feats, None, a, b
        else:
            # (feats, cov, y)
            return feats, a, b, None

    if n == 4:
        feats, cov, y, inst_id = batch
        return feats, cov, y, inst_id

    raise ValueError(f"Unexpected batch arity={n}. Expected 2/3/4.")


########################################
# Model init / checkpoint load
########################################

def initiate_model(args, ckpt_path, device=None):
    """
    Build the correct model for the chosen task, load weights, return model.eval().

    Mirrors core_utils.train() model init conventions so eval == training.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Init Model")

    clam_kwargs = dict(
        task_type=args.task_type,
        embed_dim=args.embed_dim,
        size_arg=getattr(args, "model_size", "small"),
        gate=True,
        dropout=args.drop_out,
        n_classes=getattr(args, "n_classes", 2),
        k_sample=getattr(args, "B", 8),
        instance_loss_fn=nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 10.0], device=device)
        ),
        subtyping=getattr(args, "subtyping", False),

        cov_dim=getattr(args, "cov_dim", 0),
        cov_hidden=getattr(args, "cov_hidden", 16),
        cov_use_layernorm=getattr(args, "cov_use_layernorm", True),
        cov_dropout=getattr(args, "cov_dropout", 0.0),
        cov_fusion=getattr(args, "cov_fusion", "concat"),
    )

    if args.model_type in ['clam_family', 'clam_sb', 'clam_mb', 'clam_sb_surv']:
        model = CLAMFamilySB(**clam_kwargs)
    else:
        if args.task_type != "classification":
            raise NotImplementedError("Legacy MIL baselines here only cover classification.")
        mil_kwargs = dict(
            embed_dim=args.embed_dim,
            dropout=args.drop_out,
            n_classes=getattr(args, "n_classes", 2)
        )
        model = MIL_fc_mc(**mil_kwargs) if mil_kwargs["n_classes"] > 2 else MIL_fc(**mil_kwargs)

    print_network(model)

    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_clean = {k.replace('module.', ''): v for k, v in ckpt.items()}

    missing, unexpected = model.load_state_dict(ckpt_clean, strict=False)
    if len(missing) > 0:
        print(" [initiate_model] Missing keys:", missing)
    if len(unexpected) > 0:
        print(" [initiate_model] Unexpected keys:", unexpected)

    model = model.to(device).eval()
    return model


########################################
# High-level evaluation entrypoint
########################################

def evaluate_model(dataset_split, args, ckpt_path):
    """
    Unified evaluation wrapper that REUSES the same logic as training exports.

    Returns:
        model, metrics_dict, df_results

    df_results schema matches summarize_all():
      - classification: slide_id, case_id, label, pred, prob_class_*
      - survival:       slide_id, case_id, time, event, risk
      - regression:     slide_id, case_id, target, pred, abs_err, squared_err

    NOTE:
      dataset may yield institution_id; we ignore it for metrics by default,
      but you can later extend summarize_all to record it.
    """
    device = torch.device(getattr(args, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = initiate_model(args, ckpt_path, device=device)

    loader = get_split_loader(
        dataset_split,
        training=False,
        testing=getattr(args, "testing", False),
        weighted=False,
        task_type=args.task_type
    )

    # ----------------------------------------------------------
    # If summarize_all already supports your new collate outputs,
    # you can just call it.
    #
    # But if summarize_all assumes fixed tuple sizes, we wrap the
    # loader batches into a compatible (feats, cov?, y) format here.
    # ----------------------------------------------------------
    def _batch_adapter():
        for batch in loader:
            feats, cov, y, inst_id = _unpack_batch(batch, args.task_type)
            # yield back a tuple shape summarize_all is guaranteed to understand:
            if cov is None:
                yield (feats, y)
            else:
                yield (feats, cov, y)

    # Use adapter only if return_institution might be on
    needs_adapter = bool(getattr(args, "return_institution", False))

    if needs_adapter:
        # Create a lightweight iterable that mimics a loader
        class _IterWrapper:
            def __init__(self, it):
                self.it = it
            def __iter__(self):
                return iter(self.it)

        adapted_loader = _IterWrapper(_batch_adapter())
        metrics, df = summarize_all(
            model=model,
            loader=adapted_loader,
            task_type=args.task_type,
            return_df=True
        )
    else:
        metrics, df = summarize_all(
            model=model,
            loader=loader,
            task_type=args.task_type,
            return_df=True
        )

    # Optional: KM plot for survival (saved per fold/split if args.save_dir is provided)
    if args.task_type == "survival" and getattr(args, "save_dir", None):
        _save_km(df, args)

    return model, metrics, df


########################################
# Survival: optional KM plot helper
########################################

def _save_km(df_surv, args):
    """
    Saves a Kaplan-Meier median-risk split curve.
    IMPORTANT: Use unique filename to avoid overwriting across folds/splits.
    """
    if df_surv is None or df_surv.empty:
        return
    if not all(c in df_surv.columns for c in ["time", "event", "risk"]):
        return

    os.makedirs(args.save_dir, exist_ok=True)

    fold_tag  = getattr(args, "current_fold", None)
    split_tag = getattr(args, "split", None)

    if fold_tag is None:
        fname = "km_curve.png"
    else:
        fname = f"km_curve_fold{fold_tag}_{split_tag if split_tag else 'split'}.png"

    save_path = os.path.join(args.save_dir, fname)
    plot_kaplan_meier(
        risk_scores=df_surv["risk"].to_numpy(),
        time=df_surv["time"].to_numpy(),
        event=df_surv["event"].to_numpy(),
        save_path=save_path,
        title=f"Kaplan-Meier (median risk split){'' if fold_tag is None else f' | fold {fold_tag}'}"
    )


def plot_kaplan_meier(risk_scores, time, event, save_path=None, title='Kaplan-Meier Survival Curve'):
    """
    risk_scores: higher = riskier (worse survival). We median-split on risk_scores.
    """
    risk_scores = np.asarray(risk_scores, dtype=float).flatten()
    time  = np.asarray(time, dtype=float).flatten()
    event = np.asarray(event, dtype=float).flatten()

    median_risk = np.median(risk_scores)
    group = risk_scores >= median_risk  # True = high risk

    plt.figure(figsize=(8, 6))
    kmf = KaplanMeierFitter()

    for label_bool, disp in [(True, "High risk"), (False, "Low risk")]:
        mask = (group == label_bool)
        if mask.sum() == 0:
            continue
        kmf.fit(time[mask], event[mask], label=f"{disp} (n={int(mask.sum())})")
        kmf.plot(ci_show=True, linewidth=2, alpha=0.9)

    high_mask = (group == True)
    low_mask  = (group == False)

    p_value = np.nan
    if high_mask.sum() > 0 and low_mask.sum() > 0:
        try:
            result = logrank_test(
                time[high_mask], time[low_mask],
                event[high_mask], event[low_mask]
            )
            p_value = float(result.p_value)
        except Exception:
            p_value = np.nan

    plt.text(
        0.05, 0.05,
        f"Log-rank p = {p_value:.4g}" if np.isfinite(p_value) else "Log-rank p = NA",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8)
    )

    plt.title(title, fontsize=14)
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
