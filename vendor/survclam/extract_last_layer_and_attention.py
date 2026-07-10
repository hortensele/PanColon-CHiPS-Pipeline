#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
from glob import glob
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dataset_modules.dataset_generic import Generic_WSIFamilyDataset
from utils.pca_utils import load_pca_npz, wrap_dataset_with_pca


# ----------------------------
# helpers
# ----------------------------
def resolve_feature_leaf_dir(features_root: str, dataset_name: str, feature_key: str):
    """
    Feature-store layout:
      {features_root}/{dataset_name}/{feature_key}/pt_files

    Returns the LEAF directory containing *.pt.
    """
    cand = os.path.join(features_root, dataset_name, feature_key, "pt_files")
    if os.path.isdir(cand):
        return cand
    return cand


def _find_pca_npz(
    *,
    results_dir: str,
    pca_cache_root: str,
    dataset_name: str,
    feature_key: str,
    pca_k: int,
    fold: int,
) -> str:
    """
    Find fold-specific PCA model file.

    Supports:
      - NEW cache system (pca_cache_root):
          {pca_cache_root}/{dataset}/{feature_key}/ipca_k{K}/(fold files...)
      - LEGACY (inside results_dir):
          {results_dir}/pca_models_final/fold_{fold}_ipca_k{K}.npz
    """
    patterns = []

    if pca_cache_root and pca_cache_root.strip():
        base = os.path.join(pca_cache_root, dataset_name, feature_key, f"ipca_k{int(pca_k)}")
        patterns += [
            os.path.join(base, f"fold_{fold}_ipca_k{int(pca_k)}.npz"),
            os.path.join(base, f"fold_{fold}_ipca*.npz"),
            os.path.join(base, f"fold_{fold}*.npz"),
            os.path.join(base, f"*fold_{fold}*k{int(pca_k)}*.npz"),
            os.path.join(base, "*.npz"),
        ]

    if results_dir and results_dir.strip():
        base_old = os.path.join(results_dir, "pca_models_final")
        patterns += [
            os.path.join(base_old, f"fold_{fold}_ipca_k{int(pca_k)}.npz"),
            os.path.join(base_old, f"fold_{fold}*.npz"),
        ]

    hits = []
    for pat in patterns:
        hits.extend(sorted(glob(pat)))

    hits = [h for h in hits if os.path.isfile(h)]
    if not hits:
        raise FileNotFoundError(
            "[PCA] Could not find PCA .npz.\n"
            f"  tried pca_cache_root='{pca_cache_root}' (new cache) and results_dir='{results_dir}' (legacy)\n"
            f"  dataset_name='{dataset_name}', feature_key='{feature_key}', pca_k={pca_k}, fold={fold}\n"
            "  Tip: if you want NO PCA, pass --pca_k 0."
        )

    def _score(path: str) -> int:
        name = os.path.basename(path).lower()
        s = 0
        if f"fold_{fold}" in name:
            s += 50
        if f"k{int(pca_k)}" in name:
            s += 50
        if "ipca" in name:
            s += 10
        return s

    hits = sorted(hits, key=_score, reverse=True)
    return hits[0]


def load_tile_ids_for_slide(slide_id: str, tile_locs_dir: str, tile_id_col: str = "tile_id"):
    """
    UNITED tile locations:
      {tile_locs_dir}/{slide_id}__tile_locations*.csv

    Returns: (tile_ids:list[str], csv_path:str). If not found -> ([], "").
    """
    if not tile_locs_dir or not tile_locs_dir.strip():
        return [], ""

    pattern = os.path.join(tile_locs_dir, f"{slide_id}__tile_locations*.csv")
    hits = sorted(glob(pattern))
    if len(hits) == 0:
        return [], ""

    csv_path = hits[0]
    df = pd.read_csv(csv_path)

    if tile_id_col not in df.columns:
        raise KeyError(f"'{tile_id_col}' not found in {csv_path}. Columns={list(df.columns)}")

    tile_ids = df[tile_id_col].astype(str).tolist()
    return tile_ids, csv_path


def _load_slide_feats_from_pt(pt_path: str) -> torch.Tensor:
    """
    Load slide-level features from a .pt file. Supports common CLAM formats:
      - torch.Tensor
      - numpy.ndarray
      - dict with keys like ["features","feats","x","embeddings","embedding"] containing Tensor/ndarray
    Returns a CPU float tensor [N, D].
    """
    obj = torch.load(pt_path, map_location="cpu")

    feats = None

    # 1) direct tensor
    if torch.is_tensor(obj):
        feats = obj

    # 2) direct numpy
    elif isinstance(obj, np.ndarray):
        feats = torch.from_numpy(obj)

    # 3) dict container
    elif isinstance(obj, dict):
        for k in ["features", "feats", "x", "embeddings", "embedding"]:
            if k not in obj:
                continue
            v = obj[k]
            if torch.is_tensor(v):
                feats = v
                break
            if isinstance(v, np.ndarray):
                feats = torch.from_numpy(v)
                break
            if isinstance(v, list):
                feats = torch.from_numpy(np.asarray(v))
                break

        if feats is None:
            raise RuntimeError(f"Unrecognized .pt dict format at {pt_path}. Keys={list(obj.keys())}")

    # 4) raw list (rare but possible)
    elif isinstance(obj, list):
        feats = torch.from_numpy(np.asarray(obj))

    else:
        raise RuntimeError(f"Unrecognized .pt format at {pt_path} (type={type(obj)})")

    feats = feats.float()

    # squeeze common singleton batch dim
    if feats.ndim == 3 and feats.shape[0] == 1:
        feats = feats.squeeze(0)

    if feats.ndim != 2:
        raise RuntimeError(f"Expected [N, D] feats from {pt_path}, got shape={tuple(feats.shape)}")

    return feats


def _apply_pca_tensor(feats_raw: torch.Tensor, pca) -> torch.Tensor:
    """
    Apply PCA transformer to a slide tensor.
    """
    if feats_raw is None:
        raise ValueError("feats_raw is None")

    x_t = feats_raw if torch.is_tensor(feats_raw) else torch.as_tensor(feats_raw)
    if x_t.ndim != 2:
        raise ValueError(f"Expected feats_raw [N,D], got shape={tuple(x_t.shape)}")

    x = x_t.detach().cpu().numpy()

    if hasattr(pca, "transform") and callable(getattr(pca, "transform")):
        x2 = pca.transform(x)
        return torch.from_numpy(np.asarray(x2)).float()

    for fn in ("transform_np", "apply", "forward"):
        if hasattr(pca, fn) and callable(getattr(pca, fn)):
            x2 = getattr(pca, fn)(x)
            return torch.from_numpy(np.asarray(x2)).float()

    if callable(pca):
        try:
            x2 = pca(x)
            return torch.from_numpy(np.asarray(x2)).float()
        except Exception:
            pass

    mean = None
    comps = None
    for mname in ("mean_", "mean", "mu", "center"):
        if hasattr(pca, mname):
            mean = getattr(pca, mname)
            break
    for cname in ("components_", "components", "Vt", "basis", "W"):
        if hasattr(pca, cname):
            comps = getattr(pca, cname)
            break
    if mean is None or comps is None:
        raise AttributeError(
            "PCA object does not support .transform and is missing mean_/components_. "
            f"type(pca)={type(pca)}"
        )

    mean = np.asarray(mean).reshape(1, -1)
    comps = np.asarray(comps)
    if comps.ndim != 2:
        raise ValueError(f"components has unexpected ndim={comps.ndim}")
    if comps.shape[1] != x.shape[1] and comps.shape[0] == x.shape[1]:
        comps = comps.T
    if comps.shape[1] != x.shape[1]:
        raise ValueError(f"components shape {comps.shape} incompatible with input dim {x.shape[1]}")

    x2 = (x - mean).dot(comps.T)
    return torch.from_numpy(np.asarray(x2)).float()


# ----------------------------
# checkpoint + model loading
# ----------------------------
def _find_checkpoint(results_dir: str, fold: int) -> str:
    candidates = []
    candidates += glob(os.path.join(results_dir, f"fold_{fold}*checkpoint*.pt"))
    candidates += glob(os.path.join(results_dir, f"s_{fold}*checkpoint*.pt"))
    candidates += glob(os.path.join(results_dir, f"*fold_{fold}*.pt"))
    candidates += glob(os.path.join(results_dir, "best_model*.pt"))
    candidates += glob(os.path.join(results_dir, "best_checkpoint*.pt"))
    candidates += glob(os.path.join(results_dir, "checkpoint*.pt"))

    def _score(p):
        name = os.path.basename(p).lower()
        s = 0
        if "best" in name:
            s += 100
        if f"fold_{fold}" in name or f"s_{fold}" in name:
            s += 50
        if "checkpoint" in name:
            s += 10
        return s

    candidates = sorted(set(candidates), key=_score, reverse=True)
    if len(candidates) == 0:
        raise FileNotFoundError(f"Could not find a checkpoint under: {results_dir}")
    return candidates[0]


def _load_checkpoint(path: str, device: str):
    return torch.load(path, map_location=device)


def _infer_embed_dim_from_state_dict(sd: dict) -> int:
    k = "pre_attn.0.weight"
    if k in sd and torch.is_tensor(sd[k]) and sd[k].ndim == 2:
        return int(sd[k].shape[1])

    embed_dim = None
    for kk, v in sd.items():
        if torch.is_tensor(v) and kk.endswith(".weight") and v.ndim == 2:
            in_features = int(v.shape[1])
            if embed_dim is None or in_features > embed_dim:
                embed_dim = in_features
    if embed_dim is None:
        raise RuntimeError("Could not infer embed_dim from state_dict.")
    return embed_dim


def _infer_hidden_dim(sd: dict) -> int:
    k = "pre_attn.0.weight"
    if k in sd and torch.is_tensor(sd[k]) and sd[k].ndim == 2:
        return int(sd[k].shape[0])
    raise RuntimeError("Could not infer hidden_dim from state_dict (missing pre_attn.0.weight).")


def _infer_attn_dim_and_gate(sd: dict):
    gated = any(k.startswith("attn_mod.attention_a.") for k in sd.keys()) and \
            any(k.startswith("attn_mod.attention_b.") for k in sd.keys()) and \
            any(k.startswith("attn_mod.attention_c.") for k in sd.keys())

    if gated:
        k = "attn_mod.attention_c.weight"
        attn_dim = int(sd[k].shape[1]) if (k in sd and torch.is_tensor(sd[k]) and sd[k].ndim == 2) else None
        return attn_dim, True

    attn_dim = None
    for kk, v in sd.items():
        if kk.startswith("attn_mod.attention.") and kk.endswith(".weight") and torch.is_tensor(v) and v.ndim == 2:
            out_f, in_f = int(v.shape[0]), int(v.shape[1])
            if out_f == 1:
                attn_dim = in_f
                break
    return attn_dim, False


def _infer_size_arg(embed_dim: int, hidden_dim: int, attn_dim: int) -> str:
    if hidden_dim != 512:
        return "small"
    if attn_dim == 384:
        return "big"
    return "small"


def _infer_cov_dim(sd: dict) -> int:
    k = "cov_mlp.0.weight"
    if k in sd and torch.is_tensor(sd[k]) and sd[k].ndim == 2:
        return int(sd[k].shape[1])
    return 0


def _infer_cov_hidden(sd: dict) -> int:
    k = "cov_mlp.0.weight"
    if k in sd and torch.is_tensor(sd[k]) and sd[k].ndim == 2:
        return int(sd[k].shape[0])
    return 16


def _infer_cov_fusion(sd: dict, cov_dim: int) -> str:
    if cov_dim <= 0:
        return "concat"
    if any(k.startswith("cov_to_risk.") for k in sd.keys()):
        return "cox_additive"
    return "concat"


def _infer_n_classes(task_type: str, sd: dict) -> int:
    if task_type == "classification":
        k = "head.weight"
        if k in sd and torch.is_tensor(sd[k]) and sd[k].ndim == 2:
            return int(sd[k].shape[0])
        return 2
    return 2


def _build_model_from_state_dict(state_dict: dict, device: str, task_type: str):
    from models.model_clam_family import CLAMFamilySB

    embed_dim = _infer_embed_dim_from_state_dict(state_dict)
    hidden_dim = _infer_hidden_dim(state_dict)
    attn_dim, gate = _infer_attn_dim_and_gate(state_dict)
    if attn_dim is None:
        attn_dim = 256

    size_arg = _infer_size_arg(embed_dim, hidden_dim, attn_dim)

    cov_dim = _infer_cov_dim(state_dict)
    cov_hidden = _infer_cov_hidden(state_dict)
    cov_fusion = _infer_cov_fusion(state_dict, cov_dim)
    n_classes = _infer_n_classes(task_type, state_dict)

    model = CLAMFamilySB(
        task_type=task_type,
        embed_dim=embed_dim,
        size_arg=size_arg,
        gate=gate,
        dropout=0.25,
        n_classes=n_classes,
        k_sample=8,
        instance_loss_fn=nn.CrossEntropyLoss(),
        subtyping=False,
        cov_dim=cov_dim,
        cov_hidden=cov_hidden,
        cov_use_layernorm=True,
        cov_dropout=0.0,
        cov_fusion=cov_fusion,
        nan_debug=False,
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    print(
        f"[INFO] Rebuilt CLAMFamilySB from state_dict: task_type={task_type} "
        f"embed_dim={embed_dim} size_arg={size_arg} gate={gate} "
        f"cov_dim={cov_dim} cov_hidden={cov_hidden} cov_fusion={cov_fusion} "
        f"n_classes={n_classes} | missing={len(missing)} unexpected={len(unexpected)} (strict=False)"
    )
    return model


def _infer_model_from_checkpoint(ckpt, device: str, task_type: str):
    if isinstance(ckpt, OrderedDict):
        return _build_model_from_state_dict(ckpt, device=device, task_type=task_type)

    if isinstance(ckpt, dict):
        for k in ["model", "clam_model", "network"]:
            if k in ckpt and hasattr(ckpt[k], "to"):
                model = ckpt[k].to(device)
                model.eval()
                return model

        for k in ["model_state_dict", "state_dict", "model_sd"]:
            if k in ckpt and isinstance(ckpt[k], (dict, OrderedDict)):
                return _build_model_from_state_dict(ckpt[k], device=device, task_type=task_type)

    if hasattr(ckpt, "to"):
        model = ckpt.to(device)
        model.eval()
        return model

    raise RuntimeError(f"Unrecognized checkpoint format at {type(ckpt)}")


# ----------------------------
# forward parsing + hooks
# ----------------------------
class LastLayerHook:
    """
    Captures the INPUT to the final linear layer (prefer out_features==1).
    """
    def __init__(self):
        self.last_x = None
        self.handle = None

    def attach(self, model: torch.nn.Module):
        linear_layers = []
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                linear_layers.append((name, m))

        chosen = None
        for name, m in linear_layers:
            if m.out_features == 1:
                chosen = (name, m)
                break
        if chosen is None and len(linear_layers) > 0:
            chosen = sorted(linear_layers, key=lambda x: x[1].out_features)[0]

        if chosen is None:
            raise RuntimeError("Could not find any nn.Linear layer to hook for last-layer embedding.")

        name, layer = chosen

        def _hook(module, inputs, outputs):
            x = inputs[0]
            if x is not None and torch.is_tensor(x):
                self.last_x = x.detach()

        self.handle = layer.register_forward_hook(_hook)
        return name

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def _extract_attention_from_forward_output(out):
    if isinstance(out, dict):
        for k in ["A", "attn", "attention", "A_raw", "alpha", "weights", "attn_norm"]:
            if k in out and torch.is_tensor(out[k]):
                return out[k]
        return None

    if isinstance(out, (tuple, list)):
        for item in out:
            if torch.is_tensor(item) and item.numel() > 0 and item.ndim in (1, 2):
                x = item.detach().float()
                x1 = x[0] if (x.ndim == 2 and x.shape[0] == 1) else x
                if torch.all(torch.isfinite(x1)) and (x1.min() >= -1e-4) and (x1.max() <= 1 + 1e-4):
                    s = float(x1.sum().cpu())
                    if 0.8 <= s <= 1.2:
                        return item.detach()
        return None

    return None


def _extract_risk_from_forward_output(out):
    if torch.is_tensor(out):
        return out.detach()

    if isinstance(out, dict):
        for k in ["logits", "risk", "hazard", "y_pred", "pred", "score"]:
            if k in out and torch.is_tensor(out[k]):
                return out[k].detach()
        return None

    if isinstance(out, (tuple, list)):
        for item in out:
            if torch.is_tensor(item) and item.numel() > 0:
                return item.detach()
        return None

    return None


def _build_requested_split_df(train_split, val_split, test_split, which_sets: str) -> pd.DataFrame:
    split_map = {
        "train": train_split,
        "val": val_split,
        "test": test_split,
    }

    if which_sets == "train_val":
        chosen = ["train", "val"]
    elif which_sets == "all":
        chosen = ["train", "val", "test"]
    else:
        chosen = [which_sets]

    dfs = []
    for split_name in chosen:
        split_obj = split_map.get(split_name, None)
        if split_obj is None:
            continue
        if not hasattr(split_obj, "slide_data") or split_obj.slide_data is None:
            continue

        df = split_obj.slide_data.copy()
        if "split" not in df.columns:
            df["split"] = split_name
        else:
            df["split"] = df["split"].fillna(split_name)

        dfs.append(df)

    if len(dfs) == 0:
        raise ValueError(f"No data found for requested split selection: {which_sets}")

    return pd.concat(dfs, axis=0).reset_index(drop=True)

# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--results_dir", type=str, required=True)
    ap.add_argument("--split_csv", type=str, required=True)

    ap.add_argument("--dataset_name", type=str, required=True,
                    help="e.g. colon_united / colon_tcga / colon_avant")
    ap.add_argument("--clinical_csv", type=str, required=True)

    ap.add_argument("--feature_dir", type=str, default="",
                    help="Explicit LEAF dir containing *.pt (overrides features_root/feature_key).")
    ap.add_argument("--features_root", type=str, default="")
    ap.add_argument("--feature_key", type=str, default="")

    ap.add_argument("--task_type", type=str, required=True, choices=["survival", "classification", "regression"])
    ap.add_argument("--time_col", type=str, default="time")
    ap.add_argument("--event_col", type=str, default="event")

    ap.add_argument("--bag_level", type=str, default="slide", choices=["patient", "slide"])

    ap.add_argument("--bag_id_col", type=str, default="case_id",
                    help="Patient/case id column in clinical CSV (patient bags).")
    ap.add_argument("--slide_pt_col", type=str, default="slide_id",
                    help="Slide id column in clinical CSV (matches <slide_id>.pt).")

    ap.add_argument("--pt_id_col", type=str, default="slide_id",
                    help="Clinical CSV column that matches the .pt filename stem for each slide.")

    ap.add_argument("--pca_k", type=int, default=0,
                    help="If >0, apply PCA to embeddings. If 0, use raw embeddings directly.")
    ap.add_argument("--pca_cache_root", type=str, default="",
                    help="Root that contains pca_models/... (NEW cache). If omitted, we also try legacy results_dir/pca_models.")
    ap.add_argument("--fold", type=int, default=0)

    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--tile_locs_dir", type=str, default="",
                    help="Directory containing tile locations CSVs named {slide_id}__tile_locations*.csv")
    ap.add_argument("--tile_id_col", type=str, default="tile_id",
                    help="Column name in tile_locations CSV that stores tile ids.")

    ap.add_argument("--patient_slide_sort", type=str, default="as_is",
                    choices=["as_is", "sort"],
                    help="When building patient bags, keep slide order as in CSV ('as_is') or sort slide_ids.")
    ap.add_argument("--skip_missing_slide_pt", action="store_true", default=True,
                    help="In patient mode: skip slides whose <slide_id>.pt is missing instead of crashing.")

    # NEW: when running patient-level splits, save outputs PER SLIDE
    ap.add_argument("--patient_mode_save", type=str, default="per_slide", choices=["per_slide", "per_patient"],
                    help="In patient mode, save outputs per_slide (recommended) or per_patient (legacy).")

    ap.add_argument("--which_sets",
        type=str,
        default="train_val",
        choices=["train", "val", "test", "train_val", "all"],
        help=(
            "Which split(s) to extract from. "
            "'train', 'val', 'test', 'train_val', or 'all'."
        ),
    )

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    emb_dir = os.path.join(args.out_dir, "embeddings")
    att_dir = os.path.join(args.out_dir, "attention")
    os.makedirs(emb_dir, exist_ok=True)
    os.makedirs(att_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------
    # resolve feature leaf dir
    # -------------------------
    if args.feature_dir and args.feature_dir.strip():
        feature_leaf = args.feature_dir
    else:
        if not args.features_root or not args.features_root.strip():
            raise ValueError("Provide --feature_dir OR --features_root + --feature_key.")
        if not args.feature_key or not args.feature_key.strip():
            raise ValueError("Provide --feature_dir OR --features_root + --feature_key.")
        feature_leaf = resolve_feature_leaf_dir(args.features_root, args.dataset_name, args.feature_key)

    if not os.path.isdir(feature_leaf):
        raise FileNotFoundError(f"Feature dir not found: {feature_leaf}")

    # -------------------------
    # PCA model (optional)
    # -------------------------
    use_pca = int(args.pca_k) > 0
    pca = None
    pca_npz = ""
    if use_pca:
        if not args.feature_key or not args.feature_key.strip():
            raise ValueError(
                "PCA enabled (pca_k>0) but --feature_key is empty. "
                "Provide --feature_key so we can locate PCA in the new cache, "
                "or pass --pca_k 0 to disable PCA."
            )
        pca_npz = _find_pca_npz(
            results_dir=args.results_dir,
            pca_cache_root=args.pca_cache_root,
            dataset_name=args.dataset_name,
            feature_key=args.feature_key,
            pca_k=int(args.pca_k),
            fold=int(args.fold),
        )
        print(f"[INFO] Using PCA: {pca_npz}")
        pca = load_pca_npz(pca_npz)
    else:
        print("[INFO] PCA disabled: using raw embeddings directly.")

    # -------------------------
    # load model
    # -------------------------
    ckpt_path = _find_checkpoint(args.results_dir, args.fold)
    print(f"[INFO] Using checkpoint: {ckpt_path}")
    ckpt = _load_checkpoint(ckpt_path, device=device)
    model = _infer_model_from_checkpoint(ckpt, device=device, task_type=args.task_type)

    hook = LastLayerHook()
    hooked_name = hook.attach(model)
    print(f"[INFO] Hooked last-layer input at: {hooked_name}")

    rows = []
    model.eval()

    # =========================================================================
    # SLIDE MODE (unchanged)
    # =========================================================================
    if args.bag_level == "slide":
        if args.task_type == "classification":
            raise ValueError(
                "This script currently does not support classification because it needs a proper label_dict. "
                "Add label_map logic like main.py if you want classification extraction."
            )

        dataset = Generic_WSIFamilyDataset(
            csv_path=args.clinical_csv,
            task_type=args.task_type,
            label_col="label",
            label_dict={},      # survival/regression
            ignore=[],
            time_col=args.time_col,
            event_col=args.event_col,
            target_col="target",
            covariate_cols=[],
            shuffle=False,
            seed=1,
            print_info=True,
            patient_strat=True,
            patient_voting="max",
            bag_level="slide",
            pt_id_col=args.pt_id_col,
        )
        dataset.data_dir = feature_leaf

        train_split, val_split, test_split = dataset.return_splits(
            from_id=False,
            csv_path=args.split_csv,
            bag_level="slide"
        )
        
        full_df = _build_requested_split_df(
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            which_sets=args.which_sets,
        )

        from dataset_modules.dataset_generic import FamilySplit
        full_split = FamilySplit(
            slide_data=full_df,
            data_dir=feature_leaf,
            task_type=args.task_type,
            time_col=args.time_col,
            event_col=args.event_col,
            target_col="target",
            covariate_cols=[],
            bag_level="slide",
            pt_id_col=args.pt_id_col,
        )

        if use_pca:
            full_split = wrap_dataset_with_pca(full_split, pca)

        with torch.no_grad():
            for idx in range(len(full_split)):
                item = full_split[idx]
                row = full_df.iloc[idx]
                slide_id = str(row[args.pt_id_col]).strip()

                if isinstance(item, (tuple, list)) and len(item) == 2:
                    feats, y = item
                    cov = None
                elif isinstance(item, (tuple, list)) and len(item) == 3:
                    feats, cov, y = item
                else:
                    raise RuntimeError(f"Unexpected item format at idx={idx}: {type(item)} len={len(item)}")

                feats = feats.to(device)
                if feats.ndim == 3 and feats.shape[0] == 1:
                    feats = feats.squeeze(0)

                if cov is not None:
                    cov = cov.to(device)
                    if cov.ndim == 2 and cov.shape[0] == 1:
                        cov = cov.squeeze(0)

                out = model(feats) if cov is None else model(feats, cov=cov)
                attn = _extract_attention_from_forward_output(out)
                risk = _extract_risk_from_forward_output(out)

                if hook.last_x is None:
                    raise RuntimeError("Hook did not capture last-layer input.")
                last_x = hook.last_x
                last_x_vec = last_x[0].detach().cpu() if (last_x.ndim == 2 and last_x.shape[0] == 1) else last_x.detach().cpu()

                tile_ids, tile_csv = load_tile_ids_for_slide(
                    slide_id=slide_id,
                    tile_locs_dir=args.tile_locs_dir,
                    tile_id_col=args.tile_id_col
                )
                n_instances = int(feats.shape[0]) if feats.ndim == 2 else int(feats.shape[1])
                if tile_ids and len(tile_ids) != n_instances:
                    raise ValueError(
                        f"[tile_id mismatch] slide_id={slide_id}: "
                        f"{len(tile_ids)} tile_ids in '{tile_csv}' but {n_instances} instances."
                    )

                emb_path = os.path.join(emb_dir, f"{slide_id}.pt")
                torch.save({"slide_id": slide_id, "last_layer": last_x_vec}, emb_path)

                att_path = os.path.join(att_dir, f"{slide_id}.pt")
                torch.save(
                    {
                        "slide_id": slide_id,
                        "attention": (attn.detach().cpu() if torch.is_tensor(attn) else None),
                        "tile_ids": tile_ids,
                        "tile_locations_csv": tile_csv,
                        "n_instances": n_instances,
                        "pca_enabled": int(use_pca),
                        "pca_k": int(args.pca_k) if use_pca else 0,
                        "pca_npz": pca_npz if use_pca else "",
                    },
                    att_path
                )

                time_v = np.nan
                event_v = np.nan
                target_v = np.nan
                if torch.is_tensor(y):
                    y_flat = y.detach().cpu().reshape(-1)
                    if args.task_type == "survival":
                        if y_flat.numel() >= 2:
                            time_v = float(y_flat[0].item())
                            event_v = float(y_flat[1].item())
                        elif y_flat.numel() == 1:
                            time_v = float(y_flat[0].item())
                    elif args.task_type == "regression":
                        if y_flat.numel() >= 1:
                            target_v = float(y_flat[0].item())

                risk_scalar = None
                if torch.is_tensor(risk):
                    r = risk.reshape(-1).detach().cpu().numpy()
                    risk_scalar = float(r[0]) if r.size > 0 else None

                row_out = {
                    "id": slide_id,
                    "case_id": str(row.get("case_id", "")),
                    "embedding_pt": emb_path,
                    "attention_pt": att_path,
                    "n_instances": n_instances,
                    "attn_available": int(attn is not None and torch.is_tensor(attn)),
                    "pca_enabled": int(use_pca),
                    "pca_k": int(args.pca_k) if use_pca else 0,
                    "pca_npz": pca_npz if use_pca else "",
                    "tile_locations_csv": tile_csv,
                    "n_tile_ids": len(tile_ids),
                    "split": str(row.get("split", "")),
                }
                if args.task_type == "survival":
                    row_out.update({"time": time_v, "event": event_v, "risk": risk_scalar})
                elif args.task_type == "regression":
                    row_out.update({"target": target_v, "pred": risk_scalar})

                rows.append(row_out)

    # =========================================================================
    # PATIENT MODE (UPDATED): save per-slide outputs while respecting patient splits
    # =========================================================================
    else:
        if args.task_type == "classification":
            raise ValueError("Patient-mode builder currently supports survival/regression only.")

        dataset = Generic_WSIFamilyDataset(
            csv_path=args.clinical_csv,
            task_type=args.task_type,
            label_col="label",
            label_dict={},
            ignore=[],
            time_col=args.time_col,
            event_col=args.event_col,
            target_col="target",
            covariate_cols=[],
            shuffle=False,
            seed=1,
            print_info=True,
            patient_strat=True,
            patient_voting="max",
            bag_level="slide",
            pt_id_col=args.slide_pt_col
        )
        dataset.data_dir = feature_leaf

        train_split, val_split, test_split = dataset.return_splits(
            from_id=False,
            csv_path=args.split_csv,
            bag_level="slide"
        )
        
        full_df = _build_requested_split_df(
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            which_sets=args.which_sets,
        )

        for col in [args.bag_id_col, args.slide_pt_col]:
            if col not in full_df.columns:
                raise KeyError(f"Column '{col}' missing from clinical CSV (needed for patient bags).")

        # map patient -> list of slides (within the patient-level split universe)
        patient_to_slides = (
            full_df.groupby(args.bag_id_col)[args.slide_pt_col]
                  .apply(lambda s: [str(x).strip() for x in s.tolist()])
                  .to_dict()
        )

        # one row per patient label (time/event or target)
        label_cols = [args.time_col, args.event_col] if args.task_type == "survival" else ["target"]
        patient_label_df = (
            full_df.groupby(args.bag_id_col)[label_cols]
                  .first()
                  .reset_index()
        )

        # optional: capture split label per patient if present
        patient_split_map = {}
        if "split" in full_df.columns:
            patient_split_map = full_df.groupby(args.bag_id_col)["split"].first().to_dict()

        # optional: per-slide split (if you have it)
        slide_split_map = {}
        if "split" in full_df.columns:
            slide_split_map = full_df.groupby(args.slide_pt_col)["split"].first().to_dict()

        with torch.no_grad():
            for _, prow in patient_label_df.iterrows():
                patient_id = str(prow[args.bag_id_col]).strip()
                slide_ids = patient_to_slides.get(patient_id, [])

                if args.patient_slide_sort == "sort":
                    slide_ids = sorted(slide_ids)

                if len(slide_ids) == 0:
                    continue

                time_v = np.nan
                event_v = np.nan
                target_v = np.nan
                if args.task_type == "survival":
                    time_v = float(prow[args.time_col])
                    event_v = float(prow[args.event_col])
                elif args.task_type == "regression":
                    target_v = float(prow["target"]) if "target" in prow.index else float(prow.get("target", np.nan))

                # -------------------------------------------------------------
                # NEW: save PER SLIDE (recommended)
                # -------------------------------------------------------------
                if args.patient_mode_save == "per_slide":
                    for sid in slide_ids:
                        pt_path = os.path.join(feature_leaf, f"{sid}.pt")
                        if not os.path.isfile(pt_path):
                            if args.skip_missing_slide_pt:
                                continue
                            raise FileNotFoundError(f"Missing slide .pt for patient={patient_id}: {pt_path}")

                        feats_raw = _load_slide_feats_from_pt(pt_path)   # [n_i, D_raw]
                        feats = _apply_pca_tensor(feats_raw, pca) if use_pca else feats_raw
                        n_i = int(feats.shape[0])

                        tile_ids, tile_csv = load_tile_ids_for_slide(
                            slide_id=sid,
                            tile_locs_dir=args.tile_locs_dir,
                            tile_id_col=args.tile_id_col
                        )
                        if tile_ids and len(tile_ids) != n_i:
                            raise ValueError(
                                f"[tile_id mismatch] patient={patient_id} slide={sid}: "
                                f"{len(tile_ids)} tile_ids in '{tile_csv}' but {n_i} instances in {pt_path}."
                            )
                        if not tile_ids:
                            tile_ids = [""] * n_i

                        feats = feats.to(device)

                        out = model(feats)
                        attn = _extract_attention_from_forward_output(out)
                        risk = _extract_risk_from_forward_output(out)

                        if hook.last_x is None:
                            raise RuntimeError("Hook did not capture last-layer input.")
                        last_x = hook.last_x
                        last_x_vec = last_x[0].detach().cpu() if (last_x.ndim == 2 and last_x.shape[0] == 1) else last_x.detach().cpu()

                        # scalar risk/pred
                        risk_scalar = None
                        if torch.is_tensor(risk):
                            r = risk.reshape(-1).detach().cpu().numpy()
                            risk_scalar = float(r[0]) if r.size > 0 else None

                        emb_path = os.path.join(emb_dir, f"{sid}.pt")
                        torch.save(
                            {
                                "slide_id": sid,
                                "case_id": patient_id,
                                "last_layer": last_x_vec,
                                "pca_enabled": int(use_pca),
                                "pca_k": int(args.pca_k) if use_pca else 0,
                                "pca_npz": pca_npz if use_pca else "",
                            },
                            emb_path
                        )

                        att_path = os.path.join(att_dir, f"{sid}.pt")
                        torch.save(
                            {
                                "slide_id": sid,
                                "case_id": patient_id,
                                "attention": (attn.detach().cpu().reshape(-1) if torch.is_tensor(attn) else None),
                                "tile_ids": tile_ids,
                                "tile_locations_csv": tile_csv,
                                "n_instances": int(n_i),
                                "pt_path": pt_path,
                                "pca_enabled": int(use_pca),
                                "pca_k": int(args.pca_k) if use_pca else 0,
                                "pca_npz": pca_npz if use_pca else "",
                            },
                            att_path
                        )

                        row_out = {
                            "id": sid,  # <-- slide id (important)
                            "slide_id": sid,
                            "case_id": patient_id,
                            "embedding_pt": emb_path,
                            "attention_pt": att_path,
                            "n_instances": int(n_i),
                            "attn_available": int(attn is not None and torch.is_tensor(attn)),
                            "n_tile_ids": int(len(tile_ids)),
                            "tile_locations_csv": tile_csv,
                            "pca_enabled": int(use_pca),
                            "pca_k": int(args.pca_k) if use_pca else 0,
                            "pca_npz": pca_npz if use_pca else "",
                            "patient_split": str(patient_split_map.get(patient_id, "")),
                            "slide_split": str(slide_split_map.get(sid, "")),
                        }
                        if args.task_type == "survival":
                            row_out.update({"time": time_v, "event": event_v, "risk": risk_scalar})
                        elif args.task_type == "regression":
                            row_out.update({"target": target_v, "pred": risk_scalar})

                        rows.append(row_out)

                # -------------------------------------------------------------
                # LEGACY: save PER PATIENT (kept as option)
                # -------------------------------------------------------------
                else:
                    all_feats = []
                    all_tile_ids = []
                    all_slide_ids_per_instance = []
                    slide_offsets = []
                    cursor = 0
                    n_missing = 0

                    for sid in slide_ids:
                        pt_path = os.path.join(feature_leaf, f"{sid}.pt")
                        if not os.path.isfile(pt_path):
                            n_missing += 1
                            if args.skip_missing_slide_pt:
                                continue
                            raise FileNotFoundError(f"Missing slide .pt for patient={patient_id}: {pt_path}")

                        feats_raw = _load_slide_feats_from_pt(pt_path)
                        feats = _apply_pca_tensor(feats_raw, pca) if use_pca else feats_raw
                        n_i = int(feats.shape[0])

                        tile_ids, tile_csv = load_tile_ids_for_slide(
                            slide_id=sid,
                            tile_locs_dir=args.tile_locs_dir,
                            tile_id_col=args.tile_id_col
                        )
                        if tile_ids and len(tile_ids) != n_i:
                            raise ValueError(
                                f"[tile_id mismatch] patient={patient_id} slide={sid}: "
                                f"{len(tile_ids)} tile_ids in '{tile_csv}' but {n_i} instances in {pt_path}."
                            )

                        all_feats.append(feats)
                        if not tile_ids:
                            tile_ids = [""] * n_i
                        all_tile_ids.extend(tile_ids)
                        all_slide_ids_per_instance.extend([sid] * n_i)

                        slide_offsets.append({
                            "slide_id": sid,
                            "tile_locations_csv": tile_csv,
                            "start": cursor,
                            "end": cursor + n_i,
                            "n_instances": n_i,
                            "pt_path": pt_path,
                        })
                        cursor += n_i

                    if len(all_feats) == 0:
                        continue

                    feats_cat = torch.cat(all_feats, dim=0).to(device)

                    out = model(feats_cat)
                    attn = _extract_attention_from_forward_output(out)
                    risk = _extract_risk_from_forward_output(out)

                    if hook.last_x is None:
                        raise RuntimeError("Hook did not capture last-layer input.")
                    last_x = hook.last_x
                    last_x_vec = last_x[0].detach().cpu() if (last_x.ndim == 2 and last_x.shape[0] == 1) else last_x.detach().cpu()

                    emb_path = os.path.join(emb_dir, f"{patient_id}.pt")
                    torch.save({"case_id": patient_id, "last_layer": last_x_vec}, emb_path)

                    att_cpu = attn.detach().cpu().reshape(-1) if torch.is_tensor(attn) else None
                    att_path = os.path.join(att_dir, f"{patient_id}.pt")
                    torch.save(
                        {
                            "case_id": patient_id,
                            "attention": att_cpu,
                            "slide_ids_per_instance": all_slide_ids_per_instance,
                            "tile_ids": all_tile_ids,
                            "slide_offsets": slide_offsets,
                            "n_instances_total": int(feats_cat.shape[0]),
                            "n_slides_total": int(len(slide_ids)),
                            "n_slides_missing_pt": int(n_missing),
                            "pca_enabled": int(use_pca),
                            "pca_k": int(args.pca_k) if use_pca else 0,
                            "pca_npz": pca_npz if use_pca else "",
                        },
                        att_path
                    )

                    risk_scalar = None
                    if torch.is_tensor(risk):
                        r = risk.reshape(-1).detach().cpu().numpy()
                        risk_scalar = float(r[0]) if r.size > 0 else None

                    row_out = {
                        "id": patient_id,
                        "case_id": patient_id,
                        "embedding_pt": emb_path,
                        "attention_pt": att_path,
                        "n_instances": int(feats_cat.shape[0]),
                        "attn_available": int(att_cpu is not None),
                        "n_slides_in_bag": int(len(slide_offsets)),
                        "n_slides_missing_pt": int(n_missing),
                        "n_tile_ids": int(len(all_tile_ids)),
                        "pca_enabled": int(use_pca),
                        "pca_k": int(args.pca_k) if use_pca else 0,
                        "pca_npz": pca_npz if use_pca else "",
                        "patient_split": str(patient_split_map.get(patient_id, "")),
                    }
                    if args.task_type == "survival":
                        row_out.update({"time": time_v, "event": event_v, "risk": risk_scalar})
                    elif args.task_type == "regression":
                        row_out.update({"target": target_v, "pred": risk_scalar})

                    rows.append(row_out)

    hook.remove()

    out_csv = os.path.join(args.out_dir, "full_last_layer.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[OK] Wrote: {out_csv}")
    print(f"[OK] Embeddings dir: {emb_dir}")
    print(f"[OK] Attention dir : {att_dir}")


if __name__ == "__main__":
    main()