# core_utils.py
import os
import math
import copy
from contextlib import contextmanager
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index

# external/local utils
from utils.utils import (
    calculate_error,
    get_split_loader,
    get_optim,
    print_network,
    seed_torch,
)
from dataset_modules.dataset_generic import save_splits

from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam_family import CLAMFamilySB
from torch.optim.lr_scheduler import ReduceLROnPlateau


# =============================================================================
# EMA
# =============================================================================

class ModelEMA:
    """
    Exponential Moving Average of model parameters.
    Keeps a shadow copy of the model weights updated after each optimizer step.
    """
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: Optional[torch.device] = None
    ):
        self.decay = float(decay)
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.device = device
        if self.device is not None:
            self.ema.to(self.device)

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        esd = self.ema.state_dict()
        for k, v in esd.items():
            if k not in msd:
                continue
            model_v = msd[k].detach()
            if self.device is not None:
                model_v = model_v.to(self.device, non_blocking=True)

            if not torch.is_floating_point(model_v):
                v.copy_(model_v)
            else:
                v.mul_(self.decay).add_(model_v, alpha=1.0 - self.decay)


@contextmanager
def ema_apply_context(ema: Optional[ModelEMA], model: nn.Module):
    """
    Temporarily swap model weights with EMA weights for eval/export.
    """
    if ema is None:
        yield
        return
    backup = copy.deepcopy(model.state_dict())
    model.load_state_dict(ema.ema.state_dict(), strict=False)
    try:
        yield
    finally:
        model.load_state_dict(backup, strict=False)


# =============================================================================
# Survival / Cox utils
# =============================================================================

def coxph_loss(y_pred, y_time, y_event, eps=1e-8):
    # sort by descending time
    order = torch.argsort(y_time, descending=True)
    risk  = y_pred[order].view(-1)
    event = y_event[order].view(-1)

    # log cumulative sum exp for risk set
    log_cum = torch.logcumsumexp(risk, dim=0)

    pll = (risk - log_cum) * event
    denom = torch.clamp(event.sum(), min=1.0)
    return -pll.sum() / denom


def _extract_attention_from_out(out: dict):
    """
    Try common CLAM keys. Returns a 1D torch.Tensor or None.
    """
    for k in ["attn", "A", "attention", "attn_weights", "attn_raw"]:
        if k in out and out[k] is not None:
            a = out[k]
            if torch.is_tensor(a):
                return a.view(-1)
            try:
                return torch.as_tensor(a).view(-1)
            except Exception:
                return None
    return None


def _to_1d_np(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x).reshape(-1)
    return x


def attn_entropy_reg(attn, eps=1e-8):
    # attn assumed nonnegative; normalize
    p = attn / (attn.sum() + eps)
    return -(p * (p + eps).log()).sum()


def safe_concordance_index(event_times, predicted_scores, event_observed, *, debug_tag=""):
    t = _to_1d_np(event_times)
    s = _to_1d_np(predicted_scores)
    e = _to_1d_np(event_observed).astype(float)

    mask = np.isfinite(t) & np.isfinite(s) & np.isfinite(e)
    n_bad = int((~mask).sum())
    if n_bad > 0:
        print(f"[CINDEX WARNING]{' ' + debug_tag if debug_tag else ''} Dropping {n_bad}/{len(mask)} rows due to NaNs/Infs.")

    t2, s2, e2 = t[mask], s[mask], e[mask]
    if len(t2) < 2:
        return float("nan")

    return float(concordance_index(t2, s2, e2))


def _parse_survival_target(survival, device=None, *, debug_tag=""):
    """
    Strict survival parser: returns (t, e) as floats.

    Accepts:
      - Tensor/np/list shape (2,), (1,2), (B,2), (B,1,2)
      - tuple/list like (time, event)
      - dict-like with keys (time,event) variants
    """
    if isinstance(survival, (tuple, list)) and len(survival) >= 2:
        return float(survival[0]), float(survival[1])

    if isinstance(survival, dict):
        for tk, ek in [
            ("time", "event"),
            ("t", "e"),
            ("y_time", "y_event"),
            ("surv_time", "surv_event"),
        ]:
            if tk in survival and ek in survival:
                return float(survival[tk]), float(survival[ek])
        raise ValueError(f"[survival]{' '+debug_tag if debug_tag else ''} dict missing time/event keys: {list(survival.keys())}")

    surv = torch.as_tensor(survival)

    if device is not None and torch.is_tensor(surv):
        try:
            surv = surv.to(device, non_blocking=True)
        except Exception:
            pass

    if surv.ndim == 0:
        raise ValueError(f"[survival]{' '+debug_tag if debug_tag else ''} got scalar survival={surv}. Expected (time,event).")

    if surv.ndim == 3 and surv.size(1) == 1:
        surv = surv[:, 0, :]  # (B,2)

    if surv.ndim == 2:
        if surv.size(1) < 2:
            raise ValueError(f"[survival]{' '+debug_tag if debug_tag else ''} shape={tuple(surv.shape)} last dim <2.")
        return float(surv[0, 0].item()), float(surv[0, 1].item())

    if surv.ndim == 1:
        v = surv.view(-1)
        if v.numel() < 2:
            raise ValueError(f"[survival]{' '+debug_tag if debug_tag else ''} 1D numel<2. shape={tuple(surv.shape)}.")
        return float(v[0].item()), float(v[1].item())

    raise ValueError(f"[survival]{' '+debug_tag if debug_tag else ''} unsupported surv shape={tuple(surv.shape)} dtype={surv.dtype}")


def _try_parse_survival_target(survival, device=None, *, debug_tag=""):
    """
    Non-throwing wrapper: returns (t,e,ok).
    """
    try:
        t, e = _parse_survival_target(survival, device=device, debug_tag=debug_tag)
        return t, e, True
    except Exception as ex:
        print(f"[survival WARNING]{' '+debug_tag if debug_tag else ''} skipping sample: survival={survival} (type={type(survival)}). Err={ex}")
        return None, None, False


# =============================================================================
# Domain-adversarial (institution) helpers
# =============================================================================

def _has_domain_adapt(args) -> bool:
    return bool(getattr(args, "domain_adapt", False))


def _domain_lambda(epoch: int, args) -> float:
    """
    GRL lambda schedule.

    Args supported:
      - domain_lambda_max (float): cap for GRL lambda (default 0.05)
      - domain_lambda_start_epoch (int): epoch to start applying domain GRL (default 3)
      - domain_lambda_ramp_epochs (int): number of epochs to ramp to max (default 12)

    Backwards compat:
      - domain_warmup_epochs -> ramp with start_epoch=0
    """
    lam_max = float(getattr(args, "domain_lambda_max", 0.05))

    start_epoch = getattr(args, "domain_lambda_start_epoch", None)
    ramp_epochs = getattr(args, "domain_lambda_ramp_epochs", None)

    if start_epoch is None or ramp_epochs is None:
        warm = int(getattr(args, "domain_warmup_epochs", 5))
        start_epoch = 0
        ramp_epochs = max(1, warm)

    start_epoch = int(start_epoch)
    ramp_epochs = int(ramp_epochs)

    if epoch < start_epoch:
        return 0.0

    t = (epoch - start_epoch) / max(1, ramp_epochs)  # 0..1
    t = max(0.0, min(1.0, float(t)))
    s = 0.5 * (1.0 - math.cos(math.pi * t))  # cosine ease-in
    return lam_max * s


def _domain_weight(args) -> float:
    """Multiply the *domain CE loss* by this weight in the total objective."""
    return float(getattr(args, "domain_loss_weight", 1.0))


def _make_domain_weight_vector_from_counts(counts: np.ndarray, mode: str) -> np.ndarray:
    """
    Domain-class weights *inside CE*.

    Per your note: down-weight small institutions, up-weight big ones.
    """
    counts = np.asarray(counts, dtype=float)
    counts = np.maximum(counts, 0.0)
    mode = str(mode).lower()

    if mode == "none":
        w = np.ones_like(counts, dtype=float)
    elif mode == "sqrt":
        w = np.sqrt(np.maximum(counts, 1.0))
    elif mode == "count":
        w = np.maximum(counts, 1.0)
    else:
        # backwards-compat
        if mode == "sqrt_inv":
            w = 1.0 / np.sqrt(np.maximum(counts, 1.0))
        elif mode == "inv":
            w = 1.0 / np.maximum(counts, 1.0)
        else:
            print(f"[DomainAdapt WARNING] Unknown domain_class_weighting='{mode}', using 'none'.")
            w = np.ones_like(counts, dtype=float)

    m = float(np.mean(w)) if w.size else 1.0
    if m > 0:
        w = w / m
    return w


def _infer_inst_id_series_from_split(split_ds, args):
    """
    Try to get per-sample institution IDs for THIS split without loading .pt files.
    Returns (inst_ids, counts) or (None,None).
    """
    K = int(getattr(args, "num_institutions", 0))
    inst_col = str(getattr(args, "institution_col", "institution"))

    cand_int_cols = ["inst_id", "institution_id", "site_id", "institution_id_int"]

    def _try_df(df: pd.DataFrame):
        if df is None or not isinstance(df, pd.DataFrame):
            return (None, None)

        for c in cand_int_cols:
            if c in df.columns:
                v = df[c].dropna().astype(int).to_numpy()
                if v.size and K > 0:
                    v = np.clip(v, 0, K - 1)
                counts = np.bincount(v, minlength=K) if K > 0 else None
                return v, counts

        if inst_col in df.columns:
            inst_vals = df[inst_col].dropna().astype(str).to_numpy()

            for map_attr in ["institution2id", "inst2id", "institution_dict", "inst_dict"]:
                if hasattr(split_ds, map_attr):
                    m = getattr(split_ds, map_attr)
                    try:
                        v = np.array([int(m[str(x)]) for x in inst_vals], dtype=int)
                        if v.size and K > 0:
                            v = np.clip(v, 0, K - 1)
                        counts = np.bincount(v, minlength=K) if K > 0 else None
                        return v, counts
                    except Exception:
                        pass

            uniq = sorted(list(pd.unique(inst_vals)))
            if K > 0 and len(uniq) != K:
                print(f"[DomainAdapt WARNING] split has {len(uniq)} unique institutions but num_institutions={K}. "
                      f"Mask/weights may be off. Consider storing an explicit institution->id mapping in the dataset.")
            map_fallback = {u: i for i, u in enumerate(uniq)}
            v = np.array([map_fallback[str(x)] for x in inst_vals], dtype=int)
            if v.size and K > 0:
                v = np.clip(v, 0, K - 1)
            counts = np.bincount(v, minlength=K) if K > 0 else None
            return v, counts

        return (None, None)

    inst_ids, counts = _try_df(getattr(split_ds, "slide_data", None))
    if inst_ids is not None:
        return inst_ids, counts

    inst_ids, counts = _try_df(getattr(split_ds, "patient_data", None))
    if inst_ids is not None:
        return inst_ids, counts

    return None, None


def _prepare_domain_stats_for_fold(train_split, args):
    """
    Precompute (TRAIN-only) keep_mask + per-class weights for domain CE.
    """
    K = int(getattr(args, "num_institutions", 0))
    if K <= 1:
        return None

    _, counts = _infer_inst_id_series_from_split(train_split, args)
    if counts is None:
        print("[DomainAdapt WARNING] Could not infer institution counts from train_split metadata. "
              "Will run domain loss without size-aware mask/weights.")
        return None

    min_count = int(getattr(args, "domain_min_count", 0))
    keep_mask = np.ones((K,), dtype=bool)
    if min_count and min_count > 0:
        keep_mask = counts >= min_count

    mode = str(getattr(args, "domain_class_weighting", "none")).lower()
    w = _make_domain_weight_vector_from_counts(counts.astype(float), mode=mode)

    stats = {
        "counts": counts.astype(int),
        "keep_mask": keep_mask.astype(bool),
        "weight_vec": w.astype(np.float32),
        "label_smoothing": float(getattr(args, "domain_label_smoothing", 0.0)),
    }
    return stats


def _ce_with_optional_weights_and_smoothing(logits, targets, *, weight_vec=None, label_smoothing=0.0):
    if label_smoothing and label_smoothing > 0:
        return F.cross_entropy(logits, targets, weight=weight_vec, label_smoothing=float(label_smoothing))
    return F.cross_entropy(logits, targets, weight=weight_vec)


def _domain_loss_and_acc(
    out_dict: dict,
    inst_id,
    device,
    *,
    domain_level: str,
    domain_stats: Optional[dict],
    tile_lambda_mult: float = 1.0,
):
    """
    Compute domain CE loss and domain accuracy (bag and/or tile).
    """
    if inst_id is None:
        return None, None

    if not torch.is_tensor(inst_id):
        inst_id = torch.tensor(int(inst_id), device=device, dtype=torch.long)
    else:
        inst_id = inst_id.to(device=device, dtype=torch.long).view(-1)[0]

    weight_vec_t = None
    keep_mask = None
    label_smoothing = 0.0
    if domain_stats is not None:
        keep_mask = domain_stats.get("keep_mask", None)
        w = domain_stats.get("weight_vec", None)
        label_smoothing = float(domain_stats.get("label_smoothing", 0.0))
        if w is not None:
            weight_vec_t = torch.tensor(w, device=device, dtype=torch.float32)

    if keep_mask is not None:
        inst_int = int(inst_id.detach().cpu().item())
        if inst_int < 0 or inst_int >= len(keep_mask) or (not bool(keep_mask[inst_int])):
            return None, None

    losses, accs, acc_weights = [], [], []

    if domain_level in ("bag", "both"):
        logits_bag = out_dict.get("domain_logits_bag", None)
        if logits_bag is not None:
            lb = logits_bag.view(1, -1)
            tgt = inst_id.view(1)
            loss_b = _ce_with_optional_weights_and_smoothing(lb, tgt, weight_vec=weight_vec_t, label_smoothing=label_smoothing)
            pred_b = int(torch.argmax(lb, dim=1).detach().cpu().item())
            acc_b = 1.0 if pred_b == int(tgt.detach().cpu().item()) else 0.0
            losses.append(loss_b)
            accs.append(acc_b)
            acc_weights.append(1.0)

    if domain_level in ("tile", "both"):
        logits_tile = out_dict.get("domain_logits_tile", None)
        if logits_tile is not None and logits_tile.numel() > 0:
            lt = logits_tile
            k = int(lt.shape[0])
            tgt = inst_id.repeat(k)
            loss_t = _ce_with_optional_weights_and_smoothing(lt, tgt, weight_vec=weight_vec_t, label_smoothing=label_smoothing)
            pred_t = torch.argmax(lt, dim=1).detach().cpu().numpy()
            tgt_np = tgt.detach().cpu().numpy()
            acc_t = float((pred_t == tgt_np).mean())

            tw = float(tile_lambda_mult)
            losses.append(loss_t * tw)
            accs.append(acc_t)
            acc_weights.append(tw)

    if len(losses) == 0:
        return None, None

    loss = sum(losses)
    den = float(sum(acc_weights))
    acc = float(sum(a * w for a, w in zip(accs, acc_weights)) / den) if den > 0 else float("nan")
    return loss, acc


def _domain_accuracy(out_dict: dict, inst_id, device, *, domain_level: str) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    if inst_id is None:
        return stats

    if not torch.is_tensor(inst_id):
        inst_id_t = torch.tensor(int(inst_id), device=device, dtype=torch.long).view(-1)[0]
    else:
        inst_id_t = inst_id.to(device=device, dtype=torch.long).view(-1)[0]

    if domain_level in ("bag", "both"):
        lb = out_dict.get("domain_logits_bag", None)
        if lb is not None:
            pred_b = int(torch.argmax(lb.view(1, -1), dim=1).detach().cpu().item())
            tgt_b = int(inst_id_t.detach().cpu().item())
            stats["dom_acc_bag"] = 1.0 if pred_b == tgt_b else 0.0

    if domain_level in ("tile", "both"):
        lt = out_dict.get("domain_logits_tile", None)
        if lt is not None and lt.numel() > 0:
            pred_t = torch.argmax(lt, dim=1).detach().cpu().numpy()
            tgt_t = int(inst_id_t.detach().cpu().item())
            stats["dom_acc_tile"] = float((pred_t == tgt_t).mean())

    return stats


def _update_dom_counts(per_inst_counts: dict, inst_id: int, correct: int):
    if inst_id not in per_inst_counts:
        per_inst_counts[inst_id] = [0, 0]
    per_inst_counts[inst_id][0] += int(correct)
    per_inst_counts[inst_id][1] += 1


def _macro_acc_from_counts(per_inst_counts: dict) -> float:
    accs = []
    for _, (c, n) in per_inst_counts.items():
        if n > 0:
            accs.append(float(c) / float(n))
    return float(np.mean(accs)) if len(accs) else float("nan")


def _num_present_insts(per_inst_counts: dict) -> int:
    return int(sum(1 for _, (_, n) in per_inst_counts.items() if n > 0))


def _chance_from_counts(per_inst_counts: dict, K_total: int = None) -> float:
    k_present = _num_present_insts(per_inst_counts)
    k = int(K_total) if (K_total is not None and K_total > 1) else k_present
    return float(1.0 / k) if k and k > 0 else float("nan")


# =============================================================================
# Logging helpers / early stopping
# =============================================================================

class Accuracy_Logger(object):
    """Tracks per-class accuracy for classification tasks."""
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for _ in range(self.n_classes)]

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]
        acc = float(correct) / count if count > 0 else None
        return acc, correct, count


class EarlyStopping:
    """
    Early stop on a metric that we want to MAXIMIZE (e.g., c-index, AUC, R2).

    - Saves checkpoint when metric improves by >= min_delta
    - Starts counting "no improvement" only after min_epoch
    """
    def __init__(self, patience=15, min_epoch=20, min_delta=1e-4, verbose=False):
        self.patience = int(patience)
        self.min_epoch = int(min_epoch)
        self.min_delta = float(min_delta)
        self.verbose = bool(verbose)

        self.counter = 0
        self.best_score = -float("inf")
        self.early_stop = False

    def __call__(self, epoch, metric_value, model, ckpt_name="checkpoint.pt"):
        # metric_value is a float where higher is better
        if metric_value is None or (isinstance(metric_value, float) and not np.isfinite(metric_value)):
            if self.verbose:
                print(f"[EarlyStopping] metric is invalid ({metric_value}); not updating.")
            return

        score = float(metric_value)

        improved = (score - self.best_score) >= self.min_delta
        if improved:
            self.best_score = score
            self.counter = 0
            torch.save(model.state_dict(), ckpt_name)
            if self.verbose:
                print(f"[EarlyStopping] metric improved to {score:.6f}. Saving: {ckpt_name}")
        else:
            if epoch >= self.min_epoch:
                self.counter += 1
                if self.verbose:
                    print(
                        f"[EarlyStopping] no improvement "
                        f"(metric={score:.6f}, best={self.best_score:.6f}) "
                        f"counter {self.counter}/{self.patience}"
                    )
                if self.counter >= self.patience:
                    self.early_stop = True


# =============================================================================
# ID accessors from loader
# =============================================================================

def _id_accessors_from_loader(loader):
    """
    Returns three callables:
      get_slide_id(row_idx), get_case_id(row_idx), get_institution(row_idx)
    Backward-compatible: if institution is unavailable, returns None.
    """
    ds = loader.dataset
    bag_level = getattr(ds, 'bag_level', 'slide')

    case_to_inst = {}
    if hasattr(ds, "slide_data") and isinstance(ds.slide_data, pd.DataFrame):
        sd = ds.slide_data
        if ("case_id" in sd.columns) and ("institution" in sd.columns):
            tmp = sd[["case_id", "institution"]].dropna()
            for _, r in tmp.iterrows():
                cid = str(r["case_id"])
                if cid not in case_to_inst:
                    case_to_inst[cid] = r["institution"]

    if bag_level == 'patient' and hasattr(ds, 'patient_data'):
        _pd = ds.patient_data
        if isinstance(_pd, dict) and 'case_id' in _pd:
            _case_id_vals = [str(x) for x in _pd['case_id']]
        elif hasattr(_pd, 'columns') and 'case_id' in _pd.columns:
            _case_id_vals = list(map(str, _pd['case_id'].tolist()))
        else:
            _case_id_vals = None

        if _case_id_vals is not None:
            case_ids = _case_id_vals

            def get_case(i):  return case_ids[i] if i < len(case_ids) else f"case_{i}"
            def get_slide(i): return f"{get_case(i)}_BAG"
            def get_inst(i):
                cid = get_case(i)
                return case_to_inst.get(cid, None)
            return get_slide, get_case, get_inst

    if hasattr(ds, 'slide_data') and isinstance(ds.slide_data, pd.DataFrame):
        slide_ids = ds.slide_data['slide_id'].tolist() if 'slide_id' in ds.slide_data else None
        case_ids = ds.slide_data['case_id'].tolist() if 'case_id' in ds.slide_data else None
        insts = ds.slide_data['institution'].tolist() if 'institution' in ds.slide_data else None

        def get_slide(i):
            if slide_ids is not None and i < len(slide_ids):
                return slide_ids[i]
            return f"slide_{i}"

        def get_case(i):
            if case_ids is not None and i < len(case_ids):
                return case_ids[i]
            return f"case_{i}"

        def get_inst(i):
            if insts is not None and i < len(insts):
                return insts[i]
            return case_to_inst.get(str(get_case(i)), None)

        return get_slide, get_case, get_inst

    return (lambda i: f"slide_{i}"), (lambda i: f"case_{i}"), (lambda i: None)


# =============================================================================
# Batch unpack helpers (tuple-format loaders)
# =============================================================================

def _unpack_batch(batch, device, *, expect_domain: bool, task_type: str = None):
    """
    Backward-compatible tuple unpacker.

    Supported batch formats:
      - (data, target)
      - (data, covariates, target)
      - (data, target, inst_id)
      - (data, covariates, target, inst_id)

    For survival: target should be survival-like (time,event) if possible.
    """
    covariates = None
    inst_id = None

    def _is_scalar_like(x):
        if torch.is_tensor(x):
            return (x.ndim == 0) or (x.numel() == 1)
        return isinstance(x, (int, np.integer, float, np.floating))

    def _is_surv_target_like(x):
        if torch.is_tensor(x):
            return (x.ndim >= 1) and (x.numel() == 2)
        if isinstance(x, (list, tuple, np.ndarray)):
            return np.asarray(x).size == 2
        return False

    def _to_device_if_tensor(x):
        return x.to(device, non_blocking=True) if torch.is_tensor(x) else x

    if len(batch) == 4:
        data, covariates, target, inst_id = batch
        covariates = _to_device_if_tensor(covariates)

    elif len(batch) == 3:
        a, b, c = batch

        if expect_domain and _is_scalar_like(c):
            # likely (data, target, inst_id)
            data, target, inst_id = a, b, c
        else:
            # likely (data, cov, target)
            if task_type == "survival":
                # try disambiguate
                if _is_surv_target_like(b) and not _is_surv_target_like(c):
                    data, target = a, b
                    covariates = c
                elif _is_surv_target_like(c) and not _is_surv_target_like(b):
                    data, covariates, target = a, b, c
                else:
                    data, covariates, target = a, b, c
            else:
                data, covariates, target = a, b, c

            covariates = _to_device_if_tensor(covariates) if torch.is_tensor(covariates) else covariates

    else:
        data, target = batch

    data = _to_device_if_tensor(data)
    if inst_id is not None:
        inst_id = _to_device_if_tensor(inst_id)

    return data, covariates, target, inst_id


# =============================================================================
# Summaries (used for fold export)
# =============================================================================

def summarize_classification(model, loader, return_df=True):
    device = next(model.parameters()).device
    model.eval()
    get_slide_id, get_case_id, get_inst = _id_accessors_from_loader(loader)

    all_probs_list, all_labels_list, all_preds_list = [], [], []
    per_row_slide, per_row_case, per_row_inst = [], [], []

    with torch.no_grad():
        row_idx = 0
        for batch in loader:
            data, covariates, target, _inst_id = _unpack_batch(batch, device, expect_domain=False)

            out = model(h_instances=data, cov=covariates, instance_eval=False)

            probs = out["probs"].detach().cpu().numpy().reshape(1, -1)
            pred = int(out["pred_label"].view(-1)[0].detach().cpu().item())
            label_int = int(target.detach().cpu().item()) if torch.is_tensor(target) else int(target)

            all_probs_list.append(probs[0])
            all_labels_list.append(label_int)
            all_preds_list.append(pred)

            per_row_slide.append(get_slide_id(row_idx))
            per_row_case.append(get_case_id(row_idx))
            per_row_inst.append(get_inst(row_idx))
            row_idx += 1

    all_probs_arr = np.array(all_probs_list)
    all_labels_arr = np.array(all_labels_list)
    all_preds_arr = np.array(all_preds_list)

    accuracy = float((all_labels_arr == all_preds_arr).mean())

    per_class_acc = {}
    for cls_id in np.unique(all_labels_arr):
        mask = (all_labels_arr == cls_id)
        per_class_acc[int(cls_id)] = float((all_preds_arr[mask] == all_labels_arr[mask]).mean())

    try:
        n_classes = all_probs_arr.shape[1]
        if n_classes == 2:
            auc_val = float(roc_auc_score(all_labels_arr, all_probs_arr[:, 1]))
        else:
            auc_val = float(roc_auc_score(all_labels_arr, all_probs_arr, multi_class='ovr'))
    except Exception as e:
        print("[summarize_classification] AUC failed:", e)
        auc_val = float("nan")

    metrics = {"auc": auc_val, "accuracy": accuracy, "per_class_acc": per_class_acc}

    if not return_df:
        return metrics, None

    df_dict = {
        "slide_id": per_row_slide,
        "case_id": per_row_case,
        "institution": per_row_inst,
        "label": all_labels_arr,
        "pred": all_preds_arr,
    }
    for c in range(all_probs_arr.shape[1]):
        df_dict[f"prob_class_{c}"] = all_probs_arr[:, c]

    return metrics, pd.DataFrame(df_dict)


def summarize_survival(model, loader, return_df=True):
    device = next(model.parameters()).device
    model.eval()
    get_slide_id, get_case_id, get_inst = _id_accessors_from_loader(loader)

    risks_list, times_list, events_list = [], [], []
    per_row_slide, per_row_case, per_row_inst = [], [], []

    n_skipped = 0

    with torch.no_grad():
        row_idx = 0
        for batch in loader:
            data, covariates, survival, _inst_id = _unpack_batch(batch, device, expect_domain=False, task_type="survival")

            out = model(h_instances=data, cov=covariates, instance_eval=False)
            risk_val = float(out["risk"].view(-1)[0].detach().cpu().item())

            t_i, e_i, ok = _try_parse_survival_target(survival, device=device, debug_tag="summarize_survival")
            if not ok:
                n_skipped += 1
                row_idx += 1
                continue

            risks_list.append(risk_val)
            times_list.append(float(t_i))
            events_list.append(float(e_i))

            per_row_slide.append(get_slide_id(row_idx))
            per_row_case.append(get_case_id(row_idx))
            per_row_inst.append(get_inst(row_idx))
            row_idx += 1

    if len(risks_list) < 2:
        print(f"[summarize_survival] Not enough valid samples (n={len(risks_list)}). Skipped={n_skipped}.")
        metrics = {"cindex": float("nan"), "coxloss": float("nan")}
        df = pd.DataFrame({
            "slide_id": per_row_slide,
            "case_id": per_row_case,
            "institution": per_row_inst,
            "time": times_list,
            "event": events_list,
            "risk": risks_list,
        })
        return metrics, df if return_df else (metrics, None)

    y_pred = torch.tensor(risks_list, dtype=torch.float32, device=device)
    y_time = torch.tensor(times_list, dtype=torch.float32, device=device)
    y_event = torch.tensor(events_list, dtype=torch.float32, device=device)

    coxloss_val = coxph_loss(y_pred, y_time, y_event)

    cindex_val = safe_concordance_index(
        np.asarray(times_list, dtype=float),
        -np.asarray(risks_list, dtype=float),
        np.asarray(events_list, dtype=float),
        debug_tag="summarize_survival",
    )

    metrics = {"cindex": float(cindex_val), "coxloss": float(coxloss_val.detach().cpu().item())}

    if not return_df:
        return metrics, None

    df = pd.DataFrame({
        "slide_id": per_row_slide,
        "case_id": per_row_case,
        "institution": per_row_inst,
        "time": times_list,
        "event": events_list,
        "risk": risks_list,
    })

    if n_skipped > 0:
        print(f"[summarize_survival] Skipped invalid survival rows: {n_skipped}")

    return metrics, df


def summarize_regression(model, loader, return_df=True):
    device = next(model.parameters()).device
    model.eval()
    get_slide_id, get_case_id, get_inst = _id_accessors_from_loader(loader)

    preds_list, targets_list = [], []
    per_row_slide, per_row_case, per_row_inst = [], [], []

    with torch.no_grad():
        row_idx = 0
        for batch in loader:
            data, covariates, target, _inst_id = _unpack_batch(batch, device, expect_domain=False)

            out = model(h_instances=data, cov=covariates, instance_eval=False)
            pred_val = float(out["pred"].view(-1)[0].detach().cpu().item())

            tgt_val = float(target.view(-1)[0].detach().cpu().item()) if torch.is_tensor(target) else float(target)

            preds_list.append(pred_val)
            targets_list.append(tgt_val)

            per_row_slide.append(get_slide_id(row_idx))
            per_row_case.append(get_case_id(row_idx))
            per_row_inst.append(get_inst(row_idx))
            row_idx += 1

    preds_arr = np.asarray(preds_list, dtype=float)
    targets_arr = np.asarray(targets_list, dtype=float)

    mse = float(np.mean((preds_arr - targets_arr) ** 2))
    mae = float(np.mean(np.abs(preds_arr - targets_arr)))
    rmse = float(np.sqrt(mse))

    y_true_mean = np.mean(targets_arr) if targets_arr.size > 0 else 0.0
    ss_res = np.sum((preds_arr - targets_arr) ** 2)
    ss_tot = np.sum((targets_arr - y_true_mean) ** 2)
    r2 = float("nan") if ss_tot == 0 else float(1.0 - (ss_res / ss_tot))

    metrics = {"rmse": rmse, "mae": mae, "mse": mse, "r2": r2}

    if not return_df:
        return metrics, None

    df = pd.DataFrame({
        "slide_id": per_row_slide,
        "case_id": per_row_case,
        "institution": per_row_inst,
        "target": targets_arr,
        "pred": preds_arr,
        "abs_err": np.abs(preds_arr - targets_arr),
        "squared_err": (preds_arr - targets_arr) ** 2,
    })

    return metrics, df


def summarize_all(model, loader, task_type, return_df=True):
    if task_type == 'classification':
        return summarize_classification(model, loader, return_df=return_df)
    if task_type == 'survival':
        return summarize_survival(model, loader, return_df=return_df)
    if task_type == 'regression':
        return summarize_regression(model, loader, return_df=return_df)
    raise ValueError(f"Unknown task_type {task_type}")


def _main_metric_for_task(task_type, metrics):
    if task_type == 'survival':
        return metrics.get("cindex", float("nan"))
    if task_type == 'classification':
        return metrics.get("auc", float("nan"))
    if task_type == 'regression':
        return metrics.get("r2", float("nan"))
    raise ValueError(f"Unknown task_type {task_type}")


# =============================================================================
# Fold-level export (EMA-aware via caller context)
# =============================================================================

def export_fold_summaries(fold, model, train_loader, val_loader, test_loader, task_type, args_like):
    results_dir = getattr(args_like, 'results_dir', None)
    if results_dir is None:
        raise ValueError("results_dir not found on args_like. Expect args_like.results_dir")

    os.makedirs(results_dir, exist_ok=True)

    def run_and_collect(split_name, loader_obj):
        if loader_obj is None:
            return None
        metrics, df = summarize_all(model, loader_obj, task_type, return_df=True)
        df = df.copy()
        df.insert(0, "split", split_name)

        row_summary = {"split": split_name}
        for k, v in metrics.items():
            row_summary[k] = v

        return {"metrics": row_summary, "df": df}

    train_sum = run_and_collect("train", train_loader)
    val_sum = run_and_collect("val", val_loader)
    test_sum = run_and_collect("test", test_loader)

    dfs_to_concat, metric_rows = [], []
    for obj in (train_sum, val_sum, test_sum):
        if obj is None:
            continue
        dfs_to_concat.append(obj["df"])
        metric_rows.append(obj["metrics"])

    preds_all_df = pd.concat(dfs_to_concat, axis=0, ignore_index=True) if len(dfs_to_concat) else pd.DataFrame()
    metrics_all_df = pd.DataFrame(metric_rows)

    preds_path = os.path.join(results_dir, f"fold_{fold}_predictions.csv")
    metrics_path = os.path.join(results_dir, f"fold_{fold}_metrics.csv")

    preds_all_df.to_csv(preds_path, index=False)
    metrics_all_df.to_csv(metrics_path, index=False)

    if val_sum is not None:
        return _main_metric_for_task(task_type, val_sum["metrics"])
    if train_sum is not None:
        return _main_metric_for_task(task_type, train_sum["metrics"])
    return float("nan")


# =============================================================================
# Classification train/val
# =============================================================================

def train_loop_classification(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None, args=None, ema: Optional[ModelEMA] = None):
    device = next(model.parameters()).device
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)

    epoch_loss = 0.0
    epoch_error = 0.0

    # domain logging
    epoch_dom = 0.0
    dom_n = 0
    epoch_dom_acc_sum = 0.0
    dom_acc_n = 0

    expect_domain = _has_domain_adapt(args) if args is not None else False
    dom_level = str(getattr(args, "domain_level", "bag")).lower() if args is not None else "bag"
    dom_w = _domain_weight(args) if args is not None else 0.0
    dom_tile_k = int(getattr(args, "domain_tile_k", 256)) if args is not None else 256
    tile_mult = float(getattr(args, "domain_tile_lambda_mult", 1.0)) if args is not None else 1.0
    dom_stats = getattr(args, "_domain_stats", None)

    if expect_domain and hasattr(model, "set_domain_lambda"):
        model.set_domain_lambda(_domain_lambda(epoch, args))

    for batch_idx, batch in enumerate(loader):
        data, covariates, label, inst_id = _unpack_batch(batch, device, expect_domain=expect_domain)

        label = label.to(device, non_blocking=True) if torch.is_tensor(label) else torch.tensor(int(label), device=device)

        out = model(
            h_instances=data,
            label=label,
            cov=covariates,
            instance_eval=False,
            return_domain=bool(expect_domain),
            domain_tile_k=dom_tile_k,
        )

        logits = out["logits"]
        Y_hat = out["pred_label"]

        label_for_loss = label.view(1) if label.dim() == 0 else label
        loss_task = loss_fn(logits, label_for_loss)
        loss = loss_task

        if expect_domain and inst_id is not None and dom_w > 0:
            dom_loss, dom_acc = _domain_loss_and_acc(
                out, inst_id, device,
                domain_level=dom_level,
                domain_stats=dom_stats,
                tile_lambda_mult=tile_mult,
            )
            if dom_loss is not None:
                loss = loss + dom_w * dom_loss
                epoch_dom += float(dom_loss.detach().cpu())
                dom_n += 1
                if dom_acc is not None:
                    epoch_dom_acc_sum += float(dom_acc)
                    dom_acc_n += 1

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        clip = float(getattr(args, "grad_clip_norm", 1.0)) if args is not None else 1.0
        if clip and clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        if ema is not None:
            ema.update(model)

        epoch_loss += float(loss_task.detach().cpu())
        acc_logger.log(Y_hat, label)
        epoch_error += calculate_error(Y_hat, label)

        if (batch_idx + 1) % 20 == 0:
            print(
                f"batch {batch_idx+1}, loss_task {float(loss_task):.4f}, "
                f"label {int(label.view(-1)[0].item())}, bag_size {data.size(0)}"
            )

    epoch_loss /= max(len(loader), 1)
    epoch_error /= max(len(loader), 1)

    msg = f"Epoch {epoch} | train_loss {epoch_loss:.4f} | train_error {epoch_error:.4f}"
    if expect_domain and dom_n > 0:
        dom_ce = epoch_dom / max(dom_n, 1)
        dom_acc = epoch_dom_acc_sum / max(dom_acc_n, 1) if dom_acc_n > 0 else float("nan")
        msg += f" | DomCE {dom_ce:.4f} | DomAcc {dom_acc:.4f} (w={dom_w:.3g}, lam={_domain_lambda(epoch, args):.3g})"
    print(msg)

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print(f"class {i}: acc={acc}, correct={correct}/{count}")
        if writer:
            writer.add_scalar(f"train/class_{i}_acc", acc if acc is not None else 0.0, epoch)

    if writer:
        writer.add_scalar("train/loss", epoch_loss, epoch)
        writer.add_scalar("train/error", epoch_error, epoch)
        if expect_domain and dom_n > 0:
            writer.add_scalar("train/domain_ce", epoch_dom / max(dom_n, 1), epoch)
            if dom_acc_n > 0:
                writer.add_scalar("train/domain_acc", epoch_dom_acc_sum / max(dom_acc_n, 1), epoch)
            writer.add_scalar("train/domain_grl_lambda", _domain_lambda(epoch, args), epoch)


def validate_classification(cur, epoch, model, loader, n_classes, early_stopping=None, writer=None, loss_fn=None, results_dir=None, args=None):
    device = next(model.parameters()).device
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)

    val_loss = 0.0
    val_error = 0.0

    all_probs = np.zeros((len(loader), n_classes), dtype=float)
    all_labels = np.zeros(len(loader), dtype=int)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            data, covariates, label, _inst_id = _unpack_batch(batch, device, expect_domain=False)

            label = label.to(device, non_blocking=True) if torch.is_tensor(label) else torch.tensor(int(label), device=device)

            out = model(h_instances=data, label=label, cov=covariates, instance_eval=False)

            logits = out["logits"]
            probs = out["probs"]
            Y_hat = out["pred_label"]

            acc_logger.log(Y_hat, label)

            label_for_loss = label.view(1) if label.dim() == 0 else label
            loss = loss_fn(logits, label_for_loss)

            all_probs[batch_idx] = probs.detach().cpu().numpy()
            all_labels[batch_idx] = int(label.view(-1)[0].item())

            val_loss += float(loss.detach().cpu())
            val_error += calculate_error(Y_hat, label)

    val_loss /= max(len(loader), 1)
    val_error /= max(len(loader), 1)

    try:
        if n_classes == 2:
            auc_val = float(roc_auc_score(all_labels, all_probs[:, 1]))
        else:
            auc_val = float(roc_auc_score(all_labels, all_probs, multi_class='ovr'))
    except Exception as e:
        print("[validate_classification] AUC failed:", e)
        auc_val = float("nan")

    if writer:
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/auc", float(auc_val), epoch)
        writer.add_scalar("val/error", val_error, epoch)

    print(f"\nVal Set | loss {val_loss:.4f} | error {val_error:.4f} | auc {float(auc_val):.4f}")
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print(f"class {i}: acc={acc}, correct={correct}/{count}")

    if early_stopping:
        assert results_dir is not None
        ckpt_name = os.path.join(results_dir, f"s_{cur}_checkpoint.pt")
        # EarlyStopping MAXIMIZES its metric, so monitor AUC (higher is better),
        # not val_loss (CE, lower is better). Passing val_loss here kept saving
        # the highest-loss / most-overfit checkpoint.
        early_stopping(epoch, auc_val, model, ckpt_name=ckpt_name)
        if early_stopping.early_stop:
            print("Early stopping")
            return True, float(val_loss), auc_val

    return False, float(val_loss), auc_val


# =============================================================================
# Survival train/val (FIXED: tuple loader + correct dom/L2/EMA + scheduler metric)
# =============================================================================

def _pairwise_rank_loss(risk, time, event, margin=0.0):
    """
    Simple pairwise ranking loss: for comparable pairs where i had event and t_i < t_j,
    encourage risk_i > risk_j.
    """
    n = risk.numel()
    if n < 2:
        return risk.sum() * 0.0

    r_i = risk.view(n, 1)
    r_j = risk.view(1, n)

    t_i = time.view(n, 1)
    t_j = time.view(1, n)

    e_i = event.view(n, 1)

    mask = (e_i > 0.5) & (t_i < t_j)
    if mask.sum() == 0:
        return risk.sum() * 0.0

    diff = r_i - r_j
    return F.relu(margin - diff)[mask].mean()


def train_loop_survival(
    epoch,
    model,
    loader,
    optimizer,
    alpha,
    writer=None,
    group_size=10,
    steps_per_update=2,
    args=None,
    ema: Optional[ModelEMA] = None,
):
    """
    Survival loop with group-wise Cox loss on small batches, gradient accumulation,
    optional:
      - L2 weight penalty (alpha) on model parameters
      - attention entropy regularization
      - pairwise ranking loss
      - domain-adversarial loss via model's domain heads (if enabled)
      - EMA updates after optimizer steps
    """
    device = next(model.parameters()).device
    model.train()

    expect_domain = _has_domain_adapt(args) if args is not None else False
    dom_level = str(getattr(args, "domain_level", "bag")).lower() if args is not None else "bag"
    dom_w = _domain_weight(args) if args is not None else 0.0
    dom_tile_k = int(getattr(args, "domain_tile_k", 256)) if args is not None else 256
    tile_mult = float(getattr(args, "domain_tile_lambda_mult", 1.0)) if args is not None else 1.0
    dom_stats = getattr(args, "_domain_stats", None)

    if expect_domain and hasattr(model, "set_domain_lambda"):
        model.set_domain_lambda(_domain_lambda(epoch, args))

    lam_ent = float(getattr(args, "attn_entropy_lambda", 0.0)) if args else 0.0
    lam_rank = float(getattr(args, "rank_lambda", 0.0)) if args else 0.0
    rank_margin = float(getattr(args, "rank_margin", 0.0)) if args else 0.0

    # buffers
    buf_risk, buf_time, buf_event = [], [], []
    buf_attn_reg = []
    buf_dom_loss = []

    optimizer.zero_grad(set_to_none=True)
    steps_done = 0

    # logging (optional light)
    running_task = 0.0
    running_dom = 0.0
    n_groups = 0

    def flush_group():
        nonlocal steps_done, running_task, running_dom, n_groups

        if len(buf_risk) == 0:
            return

        y_pred = torch.stack(buf_risk).view(-1)
        y_time = torch.stack(buf_time).view(-1)
        y_event = torch.stack(buf_event).view(-1)

        # base Cox
        loss = coxph_loss(y_pred, y_time, y_event)
        task_loss = loss

        # attention entropy reg (maximize entropy): add (-entropy) * lam_ent
        if lam_ent > 0 and len(buf_attn_reg) > 0:
            loss = loss + lam_ent * torch.stack(buf_attn_reg).mean()

        # pairwise ranking loss
        if lam_rank > 0:
            loss = loss + lam_rank * _pairwise_rank_loss(y_pred, y_time, y_event, margin=rank_margin)

        # domain loss (already masked/weighted inside helper)
        if expect_domain and dom_w > 0 and len(buf_dom_loss) > 0:
            dom_loss_group = torch.stack(buf_dom_loss).mean()
            loss = loss + dom_w * dom_loss_group
            running_dom += float(dom_loss_group.detach().cpu())

        # L2 penalty (lightweight: only when alpha>0)
        if alpha and float(alpha) > 0:
            l2_reg = torch.zeros((), dtype=torch.float32, device=device)
            for p in model.parameters():
                if p.requires_grad:
                    l2_reg = l2_reg + p.pow(2).sum()
            loss = loss + float(alpha) * l2_reg

        loss.backward()
        steps_done += 1
        running_task += float(task_loss.detach().cpu())
        n_groups += 1

        if steps_done % steps_per_update == 0:
            clip = float(getattr(args, "grad_clip_norm", 1.0)) if args is not None else 1.0
            if clip and clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()
            if ema is not None:
                ema.update(model)
            optimizer.zero_grad(set_to_none=True)

        buf_risk.clear()
        buf_time.clear()
        buf_event.clear()
        buf_attn_reg.clear()
        buf_dom_loss.clear()

    for batch in loader:
        data, covariates, survival, inst_id = _unpack_batch(batch, device, expect_domain=expect_domain, task_type="survival")

        t, e, ok = _try_parse_survival_target(survival, device=device, debug_tag="train_loop_survival")
        if not ok:
            continue

        out = model(
            h_instances=data,
            cov=covariates,
            instance_eval=False,
            return_domain=bool(expect_domain),
            domain_tile_k=dom_tile_k,
        )

        risk = out["risk"].view(-1)[0]

        buf_risk.append(risk)
        buf_time.append(torch.tensor(float(t), device=device, dtype=torch.float32))
        buf_event.append(torch.tensor(float(e), device=device, dtype=torch.float32))

        if lam_ent > 0:
            attn = _extract_attention_from_out(out)
            if attn is not None:
                attn = attn.to(device)
                # add (-entropy) so minimizing loss increases entropy
                buf_attn_reg.append(-attn_entropy_reg(attn))

        if expect_domain and dom_w > 0 and inst_id is not None:
            dom_loss, _dom_acc = _domain_loss_and_acc(
                out, inst_id, device,
                domain_level=dom_level,
                domain_stats=dom_stats,
                tile_lambda_mult=tile_mult,
            )
            if dom_loss is not None:
                buf_dom_loss.append(dom_loss)

        if len(buf_risk) >= int(group_size):
            flush_group()

    flush_group()

    # if gradients pending (not multiple of steps_per_update), do a final step
    if steps_done % steps_per_update != 0:
        clip = float(getattr(args, "grad_clip_norm", 1.0)) if args is not None else 1.0
        if clip and clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        optimizer.zero_grad(set_to_none=True)

    if n_groups > 0:
        msg = f"Epoch {epoch} | train_surv_coxloss {running_task/max(n_groups,1):.4f}"
        if expect_domain and dom_w > 0:
            msg += f" | train_dom_ce {running_dom/max(n_groups,1):.4f} (w={dom_w:.3g}, lam={_domain_lambda(epoch,args):.3g})"
        print(msg)

    if writer and n_groups > 0:
        writer.add_scalar("train_surv/coxloss", running_task / max(n_groups, 1), epoch)
        if expect_domain and dom_w > 0:
            writer.add_scalar("train_surv/domain_ce_group", running_dom / max(n_groups, 1), epoch)
            writer.add_scalar("train_surv/domain_grl_lambda", _domain_lambda(epoch, args), epoch)


def validate_survival(cur, epoch, model, loader, alpha, early_stopping=None, writer=None, results_dir=None, args=None):
    device = next(model.parameters()).device
    model.eval()

    risks, times, events = [], [], []
    n_skipped = 0

    # domain acc logging (micro + macro)
    expect_domain = _has_domain_adapt(args) if args is not None else False
    dom_level = str(getattr(args, "domain_level", "bag")).lower() if args is not None else "bag"
    dom_tile_k = int(getattr(args, "domain_tile_k", 256)) if args is not None else 256
    K_total = int(getattr(args, "num_institutions", 0)) if args is not None else 0

    dom_acc_bag_sum = 0.0
    dom_acc_tile_sum = 0.0
    dom_acc_bag_n = 0
    dom_acc_tile_n = 0

    bag_counts = {}
    tile_counts = {}

    if expect_domain and hasattr(model, "set_domain_lambda"):
        model.set_domain_lambda(_domain_lambda(epoch, args))

    with torch.no_grad():
        for batch in loader:
            data, covariates, survival, inst_id = _unpack_batch(batch, device, expect_domain=expect_domain, task_type="survival")

            out = model(
                h_instances=data,
                cov=covariates,
                instance_eval=False,
                return_domain=bool(expect_domain),
                domain_tile_k=dom_tile_k,
            )
            risk_val = out["risk"].view(-1)[0]

            t, e, ok = _try_parse_survival_target(survival, device=device, debug_tag="validate_survival")
            if not ok:
                n_skipped += 1
                continue

            risks.append(risk_val.detach())
            times.append(float(t))
            events.append(float(e))

            if expect_domain and inst_id is not None:
                accs = _domain_accuracy(out, inst_id, device, domain_level=dom_level)

                if not torch.is_tensor(inst_id):
                    inst_int = int(inst_id)
                else:
                    inst_int = int(inst_id.detach().cpu().view(-1)[0].item())

                if "dom_acc_bag" in accs:
                    dom_acc_bag_sum += float(accs["dom_acc_bag"])
                    dom_acc_bag_n += 1
                    _update_dom_counts(bag_counts, inst_int, int(accs["dom_acc_bag"] >= 0.5))

                if "dom_acc_tile" in accs:
                    dom_acc_tile_sum += float(accs["dom_acc_tile"])
                    dom_acc_tile_n += 1
                    _update_dom_counts(tile_counts, inst_int, int(accs["dom_acc_tile"] >= 0.5))

    if len(risks) < 2:
        raise ValueError(
            f"[validate_survival] 0 valid survival samples after parsing. "
            f"Skipped={n_skipped}, total_batches={len(loader)}"
        )

    y_pred = torch.stack(risks, dim=0).to(device=device, dtype=torch.float32)
    y_time = torch.tensor(times, dtype=torch.float32, device=device)
    y_event = torch.tensor(events, dtype=torch.float32, device=device)

    loss_val = coxph_loss(y_pred, y_time, y_event)

    # optional L2 weight penalty
    if alpha and float(alpha) > 0:
        l2_reg = torch.zeros((), dtype=torch.float32, device=device)
        for p in model.parameters():
            if p.requires_grad:
                l2_reg = l2_reg + p.pow(2).sum()
        loss_val = loss_val + float(alpha) * l2_reg

    cindex = safe_concordance_index(
        np.asarray(times, dtype=float),
        -np.asarray(y_pred.detach().cpu().numpy().ravel(), dtype=float),
        np.asarray(events, dtype=float),
        debug_tag="validate_survival",
    )

    msg = (
        f"Valid: Epoch {epoch} | "
        f"coxloss {float(loss_val.detach().cpu()):.4f} | "
        f"cindex {cindex:.4f} | "
        f"skipped {n_skipped}"
    )

    if expect_domain and (dom_acc_bag_n > 0 or dom_acc_tile_n > 0):
        if dom_acc_bag_n > 0:
            micro_bag = dom_acc_bag_sum / max(dom_acc_bag_n, 1)
            macro_bag = _macro_acc_from_counts(bag_counts)
            chance_bag = _chance_from_counts(bag_counts, K_total if K_total > 1 else None)
            msg += f" | DomAcc_bag micro {micro_bag:.3f} macro {macro_bag:.3f} chance {chance_bag:.3f}"
        if dom_acc_tile_n > 0:
            micro_tile = dom_acc_tile_sum / max(dom_acc_tile_n, 1)
            macro_tile = _macro_acc_from_counts(tile_counts)
            chance_tile = _chance_from_counts(tile_counts, K_total if K_total > 1 else None)
            msg += f" | DomAcc_tile micro {micro_tile:.3f} macro {macro_tile:.3f} chance {chance_tile:.3f}"

    print(msg)

    if writer:
        writer.add_scalar("val/coxloss", float(loss_val.detach().cpu()), epoch)
        writer.add_scalar("val/cindex", cindex, epoch)
        writer.add_scalar("val/skipped_invalid_survival", n_skipped, epoch)

    if writer and expect_domain:
        if dom_acc_bag_n > 0:
            writer.add_scalar("val/domain_acc_bag_micro", dom_acc_bag_sum / max(dom_acc_bag_n, 1), epoch)
            writer.add_scalar("val/domain_acc_bag_macro", _macro_acc_from_counts(bag_counts), epoch)
        if dom_acc_tile_n > 0:
            writer.add_scalar("val/domain_acc_tile_micro", dom_acc_tile_sum / max(dom_acc_tile_n, 1), epoch)
            writer.add_scalar("val/domain_acc_tile_macro", _macro_acc_from_counts(tile_counts), epoch)

    val_loss_float = float(loss_val.detach().cpu().item())

    if early_stopping:
        assert results_dir is not None
        ckpt_name = os.path.join(results_dir, f"s_{cur}_checkpoint.pt")
        early_stopping(epoch, cindex, model, ckpt_name=ckpt_name)
        if early_stopping.early_stop:
            print("Early stopping (metric=cindex)")
            return True, val_loss_float, cindex

    return False, val_loss_float, cindex


# =============================================================================
# Regression train/val
# =============================================================================

def train_loop_regression(epoch, model, loader, optimizer, writer=None, mse_loss_fn=None, lambda_pred_l2: float = 0.01, args=None, ema: Optional[ModelEMA] = None):
    device = next(model.parameters()).device
    model.train()

    epoch_loss = 0.0

    # domain logging
    epoch_dom = 0.0
    dom_n = 0
    epoch_dom_acc_sum = 0.0
    dom_acc_n = 0

    expect_domain = _has_domain_adapt(args) if args is not None else False
    dom_level = str(getattr(args, "domain_level", "bag")).lower() if args is not None else "bag"
    dom_w = _domain_weight(args) if args is not None else 0.0
    dom_tile_k = int(getattr(args, "domain_tile_k", 256)) if args is not None else 256
    tile_mult = float(getattr(args, "domain_tile_lambda_mult", 1.0)) if args is not None else 1.0
    dom_stats = getattr(args, "_domain_stats", None)

    if expect_domain and hasattr(model, "set_domain_lambda"):
        model.set_domain_lambda(_domain_lambda(epoch, args))

    for batch_idx, batch in enumerate(loader):
        data, covariates, target, inst_id = _unpack_batch(batch, device, expect_domain=expect_domain)

        target = target.to(device, non_blocking=True).view(-1).float() if torch.is_tensor(target) else torch.tensor(float(target), device=device).view(-1)

        out = model(
            h_instances=data,
            cov=covariates,
            instance_eval=False,
            return_domain=bool(expect_domain),
            domain_tile_k=dom_tile_k,
        )

        # Train in standardized target space: the head emits pred_z (unit scale)
        # and the model un-standardizes to out["pred"]. When standardization is
        # off the buffers are (0,1), so pred_z == pred and tz == target.
        ymean = getattr(model, "reg_y_mean", None)
        ystd = getattr(model, "reg_y_std", None)
        if ymean is not None and ystd is not None:
            t = target
            _log1p = getattr(model, "reg_y_log1p", None)
            if _log1p is not None and float(_log1p.item()) > 0.5:
                t = torch.log1p(torch.clamp(target, min=-1.0 + 1e-6))
            tz = (t - ymean) / ystd
        else:
            tz = target
        pred_z = out.get("pred_z", out["pred"]).view(-1)

        lam = float(getattr(args, "lambda_pred_l2", lambda_pred_l2)) if args is not None else lambda_pred_l2
        base_loss = mse_loss_fn(pred_z, tz)
        pred_l2 = (pred_z ** 2).mean()   # shrinks toward the target mean
        loss = base_loss + lam * pred_l2

        if expect_domain and inst_id is not None and dom_w > 0:
            dom_loss, dom_acc = _domain_loss_and_acc(
                out, inst_id, device,
                domain_level=dom_level,
                domain_stats=dom_stats,
                tile_lambda_mult=tile_mult,
            )
            if dom_loss is not None:
                loss = loss + dom_w * dom_loss
                epoch_dom += float(dom_loss.detach().cpu())
                dom_n += 1
                if dom_acc is not None:
                    epoch_dom_acc_sum += float(dom_acc)
                    dom_acc_n += 1

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        clip = float(getattr(args, "grad_clip_norm", 1.0)) if args is not None else 1.0
        if clip and clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        if ema is not None:
            ema.update(model)

        epoch_loss += float(loss.detach().cpu())

        if (batch_idx + 1) % 20 == 0:
            print(
                f"batch {batch_idx+1}, loss {float(loss):.4f}, "
                f"target {float(target.view(-1)[0].item()):.4f}, "
                f"pred {float(out['pred'].view(-1)[0].item()):.4f}, "
                f"bag_size {data.size(0)}"
            )

    epoch_loss /= max(len(loader), 1)

    msg = f"Epoch {epoch} | train_reg_loss {epoch_loss:.4f}"
    if expect_domain and dom_n > 0:
        dom_ce = epoch_dom / max(dom_n, 1)
        dom_acc = epoch_dom_acc_sum / max(dom_acc_n, 1) if dom_acc_n > 0 else float("nan")
        msg += f" | DomCE {dom_ce:.4f} | DomAcc {dom_acc:.4f} (w={dom_w:.3g}, lam={_domain_lambda(epoch, args):.3g})"
    print(msg)

    if writer:
        writer.add_scalar("train_reg/loss", epoch_loss, epoch)
        if expect_domain and dom_n > 0:
            writer.add_scalar("train_reg/domain_ce", epoch_dom / max(dom_n, 1), epoch)
            if dom_acc_n > 0:
                writer.add_scalar("train_reg/domain_acc", epoch_dom_acc_sum / max(dom_acc_n, 1), epoch)
            writer.add_scalar("train_reg/domain_grl_lambda", _domain_lambda(epoch, args), epoch)


def validate_regression(cur, epoch, model, loader, early_stopping=None, writer=None, mse_loss_fn=None, results_dir=None):
    device = next(model.parameters()).device
    model.eval()

    val_loss = 0.0
    preds_list, targets_list = [], []

    with torch.no_grad():
        for batch in loader:
            data, covariates, target, _inst_id = _unpack_batch(batch, device, expect_domain=False)

            target = target.to(device, non_blocking=True).view(-1).float() if torch.is_tensor(target) else torch.tensor(float(target), device=device).view(-1)

            out = model(h_instances=data, cov=covariates, instance_eval=False)
            pred = out["pred"].view(-1)

            loss = mse_loss_fn(pred, target)
            val_loss += float(loss.detach().cpu())

            preds_list.append(float(pred.detach().cpu().view(-1)[0].item()))
            targets_list.append(float(target.detach().cpu().view(-1)[0].item()))

    val_loss /= max(len(loader), 1)

    preds_arr = np.array(preds_list, dtype=float)
    targets_arr = np.array(targets_list, dtype=float)

    mse = float(np.mean((preds_arr - targets_arr) ** 2))
    mae = float(np.mean(np.abs(preds_arr - targets_arr)))
    rmse = float(np.sqrt(mse))

    y_true_mean = np.mean(targets_arr) if targets_arr.size > 0 else 0.0
    ss_res = np.sum((preds_arr - targets_arr) ** 2)
    ss_tot = np.sum((targets_arr - y_true_mean) ** 2)
    r2 = float("nan") if ss_tot == 0 else float(1.0 - (ss_res / ss_tot))

    print(f"\nVal (regression): loss {val_loss:.4f}, mse {mse:.4f}, mae {mae:.4f}, rmse {rmse:.4f}, r2 {r2:.4f}")

    if writer:
        writer.add_scalar("val_reg/loss", val_loss, epoch)
        writer.add_scalar("val_reg/mse", mse, epoch)
        writer.add_scalar("val_reg/mae", mae, epoch)
        writer.add_scalar("val_reg/rmse", rmse, epoch)
        writer.add_scalar("val_reg/r2", r2, epoch)

    # Model selection metric: -RMSE. EarlyStopping/scheduler MAXIMIZE, and on a
    # small val set RMSE is a far less noisy criterion than R2 (R2's denominator
    # is the val-set target variance, which swings wildly with ~40 samples and
    # made selection essentially random). -RMSE is monotone with val MSE.
    sel_metric = -rmse

    if early_stopping:
        assert results_dir is not None
        ckpt_name = os.path.join(results_dir, f"s_{cur}_checkpoint.pt")
        early_stopping(epoch, sel_metric, model, ckpt_name=ckpt_name)
        if early_stopping.early_stop:
            print("Early stopping (regression)")
            return True, float(val_loss), sel_metric

    return False, float(val_loss), sel_metric


# =============================================================================
# High-level train() that runs 1 fold end-to-end (EMA + scheduler FIXES)
# =============================================================================

def train(datasets, cur, args):
    print(f"\nTraining Fold {cur}!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if getattr(args, "log_data", False):
        from tensorboardX import SummaryWriter
        writer_dir = os.path.join(args.results_dir, str(cur))
        os.makedirs(writer_dir, exist_ok=True)
        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None

    # -------------------------------------------------------------------------
    # 1. Save splits
    # -------------------------------------------------------------------------
    print("\nInit train/val/test splits...", end=" ")
    train_split, val_split, test_split = datasets
    split_keys = ["train", "val"] + (["test"] if test_split is not None else [])
    save_splits(
        datasets[:len(split_keys)],
        split_keys,
        os.path.join(args.results_dir, f"splits_{cur}.csv")
    )
    print("Done!")

    print(f"Training on {len(train_split)} samples")
    print(f"Validating on {len(val_split)} samples")
    if test_split is not None:
        print(f"Testing on {len(test_split)} samples")

    # -------------------------------------------------------------------------
    # 2. Loss fns
    # -------------------------------------------------------------------------
    print("\nInit loss functions...", end=" ")

    if getattr(args, "bag_loss", "ce") == "svm":
        from topk import SmoothTop1SVM
        class_loss_fn = SmoothTop1SVM(n_classes=args.n_classes).to(device)
    else:
        if args.n_classes == 2:
            class_weights = torch.tensor([1.0, 10.0], dtype=torch.float32, device=device)
        else:
            class_weights = None
        class_loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(device)

    if str(getattr(args, "reg_loss", "mse")).lower() == "huber":
        reg_loss_fn = nn.SmoothL1Loss(beta=float(getattr(args, "huber_delta", 1.0))).to(device)
        print(f"[Regression] loss=Huber/SmoothL1 (beta={float(getattr(args, 'huber_delta', 1.0)):.3f})")
    else:
        reg_loss_fn = nn.MSELoss().to(device)
    print("Done!")

    # -------------------------------------------------------------------------
    # 3. DataLoaders
    # -------------------------------------------------------------------------
    print("\nInit Loaders...", end=" ")
    train_loader = get_split_loader(
        train_split,
        training=True,
        testing=getattr(args, "testing", False),
        weighted=getattr(args, "weighted_sample", False),
        task_type=args.task_type
    )
    val_loader = get_split_loader(
        val_split,
        testing=getattr(args, "testing", False),
        weighted=False,
        task_type=args.task_type
    )
    test_loader = get_split_loader(
        test_split,
        testing=getattr(args, "testing", False),
        weighted=False,
        task_type=args.task_type
    ) if test_split is not None else None
    print("Done!")

    # -------------------------------------------------------------------------
    # 3b. Domain stats (TRAIN-only) for size-aware masking/weights
    # -------------------------------------------------------------------------
    if _has_domain_adapt(args):
        args._domain_stats = _prepare_domain_stats_for_fold(train_split, args)
        if args._domain_stats is not None:
            cts = args._domain_stats["counts"]
            keep = args._domain_stats["keep_mask"]
            kept_n = int(np.sum(keep))
            min_count = int(getattr(args, "domain_min_count", 0))
            mode = str(getattr(args, "domain_class_weighting", "none")).lower()
            print(f"[DomainAdapt] TRAIN stats: K={len(cts)}, kept={kept_n}/{len(cts)} (min_count={min_count}), weighting='{mode}', "
                  f"label_smoothing={args._domain_stats['label_smoothing']}")
        else:
            print("[DomainAdapt] TRAIN stats unavailable -> no mask/weights for domain CE.")

    # -------------------------------------------------------------------------
    # 4. Init model
    # -------------------------------------------------------------------------
    print("\nInit Model...", end=" ")

    clam_kwargs = dict(
        embed_dim=args.embed_dim,
        size_arg=args.model_size if args.model_size else "small",
        gate=True,
        dropout=args.drop_out,
        n_classes=args.n_classes,
        k_sample=args.B if hasattr(args, "B") and args.B > 0 else 8,
        instance_loss_fn=nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0], device=device)),
        subtyping=getattr(args, "subtyping", False),

        # covariates
        cov_dim=getattr(args, "cov_dim", 0),
        cov_hidden=getattr(args, "cov_hidden", 16),
        cov_use_layernorm=getattr(args, "cov_use_layernorm", True),
        cov_dropout=getattr(args, "cov_dropout", 0.0),
        cov_fusion=getattr(args, "cov_fusion", "concat"),

        # domain-adversarial (institution)
        domain_adapt=bool(getattr(args, "domain_adapt", False)),
        num_institutions=int(getattr(args, "num_institutions", 0)),
        domain_level=str(getattr(args, "domain_level", "bag")),
        domain_hidden=int(getattr(args, "domain_hidden", 256)),
        domain_dropout=float(getattr(args, "domain_dropout", 0.25)),
    )

    if args.task_type in ("classification", "survival", "regression"):
        model = CLAMFamilySB(task_type=args.task_type, **clam_kwargs)
    else:
        if args.model_type == "mil":
            model = MIL_fc(embed_dim=args.embed_dim, dropout=args.drop_out, n_classes=args.n_classes)
        elif args.model_type == "mil_mc":
            model = MIL_fc_mc(embed_dim=args.embed_dim, dropout=args.drop_out, n_classes=args.n_classes)
        else:
            raise ValueError(f"Unknown task_type {args.task_type} and model_type {args.model_type}")

    model = model.to(device)

    # -------------------------------------------------------------------------
    # 4a. Regression target standardization (train-fold stats, baked into model)
    # -------------------------------------------------------------------------
    if args.task_type == "regression" and bool(getattr(args, "standardize_target", False)):
        _use_log1p = (str(getattr(args, "target_transform", "none")).lower() == "log1p")
        _tgts = []
        for _batch in train_loader:
            _, _, _t, _ = _unpack_batch(_batch, device, expect_domain=False)
            _tgts.append(float(_t.detach().cpu().view(-1)[0].item()) if torch.is_tensor(_t) else float(_t))
        _tgts = np.asarray(_tgts, dtype=float)
        # stats are computed in the (optionally log1p-transformed) target space
        _tgts_tf = np.log1p(np.clip(_tgts, -1.0 + 1e-6, None)) if _use_log1p else _tgts
        _mean = float(np.mean(_tgts_tf)) if _tgts_tf.size else 0.0
        _std = float(np.std(_tgts_tf)) if _tgts_tf.size else 1.0
        model.set_target_normalization(_mean, _std, log1p=_use_log1p)
        print(f"[Regression] target standardization: transform={'log1p' if _use_log1p else 'none'}, "
              f"mean={_mean:.4f}, std={_std:.4f} (train n={_tgts.size})")

    print("Done!")
    print_network(model)
    print(f"[Model config] task_type={args.task_type}, cov_dim={getattr(args,'cov_dim',0)}, cov_fusion={getattr(args,'cov_fusion','concat')}")

    if _has_domain_adapt(args):
        print(
            f"[DomainAdapt] enabled | num_institutions={getattr(args,'num_institutions',None)} | "
            f"level={getattr(args,'domain_level','bag')} | "
            f"lambda_max={getattr(args,'domain_lambda_max',1.0)} | "
            f"start={getattr(args,'domain_lambda_start_epoch',None)} | "
            f"ramp={getattr(args,'domain_lambda_ramp_epochs',None)} | "
            f"loss_weight={getattr(args,'domain_loss_weight',1.0)} | "
            f"tile_k={getattr(args,'domain_tile_k',256)} | "
            f"tile_mult={getattr(args,'domain_tile_lambda_mult',1.0)}"
        )

    # -------------------------------------------------------------------------
    # 4b. EMA init
    # -------------------------------------------------------------------------
    ema = None
    if bool(getattr(args, "use_ema", False)):
        ema_dev = getattr(args, "ema_device", None)
        ema_device = torch.device(ema_dev) if (ema_dev is not None and str(ema_dev).strip() != "") else None
        ema = ModelEMA(model, decay=float(getattr(args, "ema_decay", 0.999)), device=ema_device)
        print(f"[EMA] enabled | decay={float(getattr(args,'ema_decay',0.999))} | device={ema_device}")

    use_ema_eval = bool(int(getattr(args, "use_ema_eval", 1))) if ema is not None else False
    save_ema_ckpt = bool(int(getattr(args, "save_ema_ckpt", 1))) if ema is not None else False

    # -------------------------------------------------------------------------
    # 5. Optimizer
    # -------------------------------------------------------------------------
    print("\nInit optimizer ...", end=" ")
    optimizer = get_optim(model, args)

    # -------------------------------------------------------------------------
    # 5b. Scheduler (FIXED: monitor correct direction)
    # -------------------------------------------------------------------------
    use_plateau = bool(getattr(args, "use_plateau", True))
    scheduler = None
    if use_plateau:
        if args.task_type == "survival":
            mode = "max"   # monitor c-index
        elif args.task_type == "classification":
            mode = "max"   # monitor AUC
        else:
            mode = "max"   # monitor R2
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=float(getattr(args, "plateau_factor", 0.5)),
            patience=int(getattr(args, "plateau_patience", 2)),
            threshold=float(getattr(args, "plateau_threshold", 1e-4)),
            cooldown=int(getattr(args, "plateau_cooldown", 0)),
            min_lr=float(getattr(args, "plateau_min_lr", 1e-6)),
            verbose=True,
        )

    print("Done!")

    # -------------------------------------------------------------------------
    # 6. Early stopping (still on val loss, as implemented)
    # -------------------------------------------------------------------------
    print("\nSetup EarlyStopping...", end=" ")
    early_stopping = (
        EarlyStopping(
            patience=int(getattr(args, "es_patience", 15)),
            min_epoch=int(getattr(args, "es_min_epoch", 20)),
            min_delta=1e-4,    # require a real improvement
            verbose=True,
        )
        if getattr(args, "early_stopping", False)
        else None
    )
    print("Done!")

    # -------------------------------------------------------------------------
    # 7. Epoch loop
    # -------------------------------------------------------------------------
    for epoch in range(args.max_epochs):

        if args.task_type == "survival":
            train_loop_survival(
                epoch,
                model,
                train_loader,
                optimizer,
                alpha=getattr(args, "alpha", 0.0),
                writer=writer,
                group_size=int(getattr(args, "surv_group_size", 10)),
                steps_per_update=int(getattr(args, "surv_steps_per_update", 2)),
                args=args,
                ema=ema,
            )

            # validate/export with EMA weights if requested
            with ema_apply_context(ema if use_ema_eval else None, model):
                stop, val_loss_float, val_cindex = validate_survival(
                    cur,
                    epoch,
                    model,
                    val_loader,
                    alpha=getattr(args, "alpha", 0.0),
                    early_stopping=early_stopping,
                    writer=writer,
                    results_dir=args.results_dir,
                    args=args,
                )

            if scheduler is not None:
                scheduler.step(val_cindex)
                if writer:
                    writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        elif args.task_type == "regression":
            train_loop_regression(
                epoch,
                model,
                train_loader,
                optimizer,
                writer=writer,
                mse_loss_fn=reg_loss_fn,
                args=args,
                ema=ema,
            )

            with ema_apply_context(ema if use_ema_eval else None, model):
                stop, val_loss_float, val_sel = validate_regression(
                    cur,
                    epoch,
                    model,
                    val_loader,
                    early_stopping=early_stopping,
                    writer=writer,
                    mse_loss_fn=reg_loss_fn,
                    results_dir=args.results_dir
                )

            if scheduler is not None:
                # val_sel = -RMSE (higher is better), matches scheduler mode="max"
                scheduler.step(val_sel)
                if writer:
                    writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        else:  # classification
            train_loop_classification(
                epoch,
                model,
                train_loader,
                optimizer,
                n_classes=args.n_classes,
                writer=writer,
                loss_fn=class_loss_fn,
                args=args,
                ema=ema,
            )

            with ema_apply_context(ema if use_ema_eval else None, model):
                stop, val_loss_float, val_auc = validate_classification(
                    cur,
                    epoch,
                    model,
                    val_loader,
                    n_classes=args.n_classes,
                    early_stopping=early_stopping,
                    writer=writer,
                    loss_fn=class_loss_fn,
                    results_dir=args.results_dir,
                    args=args,
                )

            if scheduler is not None:
                scheduler.step(val_auc)
                if writer:
                    writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if stop:
            break

    # -------------------------------------------------------------------------
    # 8. Load best weights (early stopping ckpt) or save final
    # -------------------------------------------------------------------------
    ckpt_path = os.path.join(args.results_dir, f"s_{cur}_checkpoint.pt")
    if getattr(args, "early_stopping", False) and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        torch.save(model.state_dict(), ckpt_path)

    # optionally save EMA checkpoint
    if ema is not None and save_ema_ckpt:
        ema_path = os.path.join(args.results_dir, f"s_{cur}_checkpoint_ema.pt")
        torch.save(ema.ema.state_dict(), ema_path)

    # -------------------------------------------------------------------------
    # 9. End-of-fold export (use EMA weights if requested)
    # -------------------------------------------------------------------------
    with ema_apply_context(ema if use_ema_eval else None, model):
        val_metric_for_loop = export_fold_summaries(
            fold=cur,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            task_type=args.task_type,
            args_like=args
        )

        if test_loader is not None:
            test_metrics, _ = summarize_all(model, test_loader, args.task_type, return_df=True)
            test_metric_for_loop = _main_metric_for_task(args.task_type, test_metrics)
        else:
            test_metric_for_loop = float("nan")

    if writer:
        writer.add_scalar("fold/val_main_metric", val_metric_for_loop, cur)
        writer.add_scalar("fold/test_main_metric", test_metric_for_loop, cur)
        writer.close()

    fold_txt_path = os.path.join(args.results_dir, f"fold_{cur}.txt")
    with open(fold_txt_path, "w") as f:
        if args.task_type == "survival":
            f.write(
                f"Fold:{cur}, Alpha:{getattr(args, 'alpha', 0.0)}, "
                f"CovFusion:{args.cov_fusion}, "
                f"Val C-Index:{val_metric_for_loop:.4f}, "
                f"Test C-Index:{test_metric_for_loop:.4f}\n"
            )
        elif args.task_type == "classification":
            f.write(
                f"Fold:{cur}, "
                f"Val AUC:{val_metric_for_loop:.4f}, "
                f"Test AUC:{test_metric_for_loop:.4f}\n"
            )
        else:
            f.write(
                f"Fold:{cur}, "
                f"Val R2:{val_metric_for_loop:.4f}, "
                f"Test R2:{test_metric_for_loop:.4f}\n"
            )

    print("Fold summary:")
    if args.task_type == "survival":
        print(f"Val C-Index: {val_metric_for_loop:.4f}, Test C-Index: {test_metric_for_loop:.4f}")
    elif args.task_type == "classification":
        print(f"Val AUC: {val_metric_for_loop:.4f}, Test AUC: {test_metric_for_loop:.4f}")
    else:
        print(f"Val R2: {val_metric_for_loop:.4f}, Test R2: {test_metric_for_loop:.4f}")

    return val_metric_for_loop, test_metric_for_loop