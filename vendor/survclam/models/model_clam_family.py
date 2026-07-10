#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# Gradient Reversal Layer (GRL) for domain-adversarial training
# ============================================================
class _GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GRL(nn.Module):
    """
    Gradient reversal layer.
    - call set_lambda(lam) from the training loop each epoch/step.
    """
    def __init__(self, lambd: float = 0.0):
        super().__init__()
        self.lambd = float(lambd)

    def set_lambda(self, lambd: float):
        self.lambd = float(lambd)

    def forward(self, x):
        return _GradReverseFn.apply(x, self.lambd)


# =========================
# Attention submodules
# =========================
class Attn_Net(nn.Module):
    def __init__(self, L=1024, D=256, dropout=0.25, n_classes=1):
        super().__init__()
        layers = [
            nn.Linear(L, D),
            nn.Tanh()
        ]
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(D, n_classes))
        self.attention = nn.Sequential(*layers)

    def forward(self, x):
        # x: [N_inst, L]
        return self.attention(x), x


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=0.25, n_classes=1):
        super().__init__()
        a_layers = [
            nn.Linear(L, D),
            nn.Tanh()
        ]
        b_layers = [
            nn.Linear(L, D),
            nn.Sigmoid()
        ]
        if dropout and dropout > 0:
            a_layers.append(nn.Dropout(dropout))
            b_layers.append(nn.Dropout(dropout))

        self.attention_a = nn.Sequential(*a_layers)
        self.attention_b = nn.Sequential(*b_layers)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        # x: [N_inst, L]
        a = self.attention_a(x)
        b = self.attention_b(x)
        gated = a * b               # [N_inst, D]
        A = self.attention_c(gated) # [N_inst, n_classes]
        return A, x


# =========================
# Main model
# =========================
class CLAMFamilySB(nn.Module):
    """
    CLAMFamilySB with robust NaN/Inf handling + optional institution-adversarial branch (GRL).

    New (optional) domain-adversarial settings:
      - domain_adapt: enable institution classifier with GRL
      - num_institutions: number of institutions (K)
      - domain_level: 'bag' | 'tile' | 'both'
            bag  -> classify from bag embedding M (pre-cov fusion) (recommended baseline)
            tile -> classify from tile hidden features h_hidden (sampled) (stronger invariance)
            both -> sum both losses (training loop decides how to weight)
      - domain_hidden: hidden size for institution head MLP

    IMPORTANT:
      - This module does NOT compute the domain loss. It only exposes logits/features.
      - Training loop should:
           model.set_domain_lambda(lam)
           dom_logits = out["domain_logits_*"]
           dom_loss = CE(dom_logits, inst_id)
    """
    def __init__(
        self,
        task_type: str,
        embed_dim: int = 1024,
        size_arg: str = "small",
        gate: bool = True,
        dropout: float = 0.25,
        n_classes: int = 2,
        k_sample: int = 8,
        instance_loss_fn: nn.Module = nn.CrossEntropyLoss(),
        subtyping: bool = False,
        # covariates
        cov_dim: int = 0,
        cov_hidden: int = 16,
        cov_use_layernorm: bool = True,
        cov_dropout: float = 0.0,
        cov_fusion: str = "concat",
        # domain-adversarial (institution)
        domain_adapt: bool = False,          
        num_institutions: int = 0,               
        domain_level: str = "bag",              
        domain_hidden: int = 256,                
        domain_dropout: float = 0.25,           
        domain_share_head: bool = True,     # NEW: share same head for bag+tile
        domain_tile_hidden: int = None,     # NEW: override hidden for tile head
        domain_tile_dropout: float = None,  # NEW: override dropout for tile head
        domain_tile_sampling: str = "random",  # NEW: random|attn_top|attn_mix
        # debug
        nan_debug: bool = False,
    ):
        super().__init__()

        assert task_type in {"classification", "survival", "regression"}, \
            "task_type must be 'classification', 'survival', or 'regression'."
        self.task_type = task_type
        self.n_classes = n_classes
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.subtyping = subtyping
        self.nan_debug = bool(nan_debug)

        # domain-adversarial config
        self.domain_adapt = bool(domain_adapt)
        self.num_institutions = int(num_institutions)
        self.domain_level = str(domain_level).lower().strip()
        if self.domain_level not in {"bag", "tile", "both"}:
            raise ValueError("domain_level must be one of: 'bag', 'tile', 'both'")
        if self.domain_adapt:
            if self.num_institutions < 2:
                raise ValueError("domain_adapt=True requires num_institutions >= 2")

        size_dict = {
            "small": [embed_dim, 512, 256],
            "big":   [embed_dim, 512, 384],
        }
        if size_arg not in size_dict:
            raise ValueError("size_arg must be 'small' or 'big'")
        in_dim, hidden_dim, attn_dim = size_dict[size_arg]

        self.pre_attn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )

        if gate:
            self.attn_mod = Attn_Net_Gated(
                L=hidden_dim,
                D=attn_dim,
                dropout=dropout,
                n_classes=1
            )
        else:
            self.attn_mod = Attn_Net(
                L=hidden_dim,
                D=attn_dim,
                dropout=dropout,
                n_classes=1
            )

        # covariates
        self.cov_dim = int(cov_dim)
        self.cov_fusion = cov_fusion
        if self.cov_dim > 0:
            cov_layers = [nn.Linear(self.cov_dim, cov_hidden)]
            if cov_use_layernorm:
                cov_layers.append(nn.LayerNorm(cov_hidden))
            cov_layers.append(nn.ReLU(inplace=True))
            if cov_dropout and cov_dropout > 0:
                cov_layers.append(nn.Dropout(p=cov_dropout))
            self.cov_mlp = nn.Sequential(*cov_layers)

            if self.cov_fusion == "concat":
                head_in_dim = hidden_dim + cov_hidden
                self.cov_to_risk = None
            elif self.cov_fusion == "cox_additive":
                head_in_dim = hidden_dim
                self.cov_to_risk = nn.Linear(cov_hidden, 1)
            else:
                raise NotImplementedError(
                    f"cov_fusion='{self.cov_fusion}' not implemented."
                )
        else:
            self.cov_mlp = None
            self.cov_to_risk = None
            head_in_dim = hidden_dim

        # task-specific head
        if self.task_type == "classification":
            self.head = nn.Linear(head_in_dim, n_classes)
            self.instance_classifiers = nn.ModuleList(
                [nn.Linear(hidden_dim, 2) for _ in range(n_classes)]
            )
        elif self.task_type == "survival":
            self.head = nn.Linear(head_in_dim, 1)
            self.instance_classifiers = None
        elif self.task_type == "regression":
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(head_in_dim, max(1, head_in_dim // 2)),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(max(1, head_in_dim // 2), 1),
            )
            self.instance_classifiers = None
            # Target normalization (z-score). The head learns in standardized
            # space; the forward un-standardizes so out["pred"] is always in raw
            # target units. Buffers are part of state_dict, so eval/inference
            # automatically inherit the train-fold stats from the checkpoint.
            self.register_buffer("reg_y_mean", torch.zeros(1))
            self.register_buffer("reg_y_std", torch.ones(1))
            # 1.0 => target was log1p-transformed before standardization; the
            # forward applies expm1 so out["pred"] is always in raw target units.
            self.register_buffer("reg_y_log1p", torch.zeros(1))

        # ---- domain-adversarial head (institution classifier)
        self.domain_share_head = bool(domain_share_head)
        self.domain_tile_sampling = str(domain_tile_sampling).lower().strip()
        if self.domain_tile_sampling not in {"random", "attn_top", "attn_mix"}:
            raise ValueError("domain_tile_sampling must be one of: random|attn_top|attn_mix")

        if self.domain_adapt:
            self.grl = GRL(lambd=0.0)

            def make_dom_head(hidden_out: int, drop_p: float):
                layers = [nn.Linear(hidden_dim, hidden_out), nn.ReLU(inplace=True)]
                if drop_p and drop_p > 0:
                    layers.append(nn.Dropout(p=float(drop_p)))
                layers.append(nn.Linear(hidden_out, self.num_institutions))
                return nn.Sequential(*layers)

            self.domain_head_bag = make_dom_head(domain_hidden, float(domain_dropout))

            if self.domain_share_head:
                self.domain_head_tile = self.domain_head_bag
            else:
                th = int(domain_tile_hidden) if domain_tile_hidden is not None else int(domain_hidden)
                td = float(domain_tile_dropout) if domain_tile_dropout is not None else float(domain_dropout)
                self.domain_head_tile = make_dom_head(th, td)
        else:
            self.grl = None
            self.domain_head_bag = None
            self.domain_head_tile = None
            

    # -------------------------
    # NaN/Inf utilities
    # -------------------------
    def _sanitize(self, x: torch.Tensor, name: str):
        """
        Replace NaN/Inf with 0.0 (in a non-inplace safe way).
        If nan_debug=True, print a one-line summary when non-finite values exist.
        """
        if x is None:
            return None
        if not torch.is_tensor(x):
            return x

        bad = ~torch.isfinite(x)
        if bad.any():
            if self.nan_debug:
                n_bad = int(bad.sum().item())
                n_tot = x.numel()
                x_tmp = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                mn = float(x_tmp.min().detach().cpu().item()) if x_tmp.numel() else 0.0
                mx = float(x_tmp.max().detach().cpu().item()) if x_tmp.numel() else 0.0
                print(f"[SANITIZE] {name}: {n_bad}/{n_tot} non-finite -> 0 (min={mn:.3g}, max={mx:.3g})")
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x

    def set_domain_lambda(self, lambd: float):
        """
        Set GRL lambda. Call this from your training loop each epoch/step.
        No-op if domain_adapt=False.
        """
        if self.domain_adapt and self.grl is not None:
            self.grl.set_lambda(lambd)

    def set_target_normalization(self, mean: float, std: float, log1p: bool = False):
        """
        Set the regression target z-score buffers (train-fold stats). When
        log1p=True the stats are computed in log1p space and forward applies
        expm1, so the head learns in standardized-log space while out["pred"]
        stays in raw target units. No-op for non-regression heads.
        """
        if self.task_type != "regression":
            return
        std = float(std)
        if not np.isfinite(std) or std <= 1e-8:
            std = 1.0
        dev = self.reg_y_mean.device
        self.reg_y_mean = torch.tensor([float(mean)], device=dev)
        self.reg_y_std = torch.tensor([std], device=dev)
        self.reg_y_log1p = torch.tensor([1.0 if log1p else 0.0], device=dev)

    @staticmethod
    def _pos_targets(k, device):
        return torch.full((k,), 1, device=device, dtype=torch.long)

    @staticmethod
    def _neg_targets(k, device):
        return torch.full((k,), 0, device=device, dtype=torch.long)

    def _inst_eval_inclass(self, A_norm, h_hidden, classifier, k):
        device = h_hidden.device
        _, N = A_norm.shape
        k = min(k, N)
        if k < 1:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

        top_idx = torch.topk(A_norm, k, dim=1)[1].squeeze(0)
        bot_idx = torch.topk(-A_norm, k, dim=1)[1].squeeze(0)

        top_feats = h_hidden[top_idx]
        bot_feats = h_hidden[bot_idx]

        pos_t = self._pos_targets(k, device)
        neg_t = self._neg_targets(k, device)

        feats   = torch.cat([top_feats, bot_feats], dim=0)
        targets = torch.cat([pos_t, neg_t], dim=0)

        logits = classifier(feats)
        preds = torch.argmax(logits, dim=1)
        inst_loss = self.instance_loss_fn(logits, targets)
        return inst_loss, preds, targets

    def _inst_eval_outclass(self, A_norm, h_hidden, classifier, k):
        device = h_hidden.device
        _, N = A_norm.shape
        k = min(k, N)
        if k < 1:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

        top_idx = torch.topk(A_norm, k, dim=1)[1].squeeze(0)
        top_feats = h_hidden[top_idx]

        neg_t = self._neg_targets(k, device)
        logits = classifier(top_feats)
        preds = torch.argmax(logits, dim=1)
        inst_loss = self.instance_loss_fn(logits, neg_t)
        return inst_loss, preds, neg_t

    def forward(
        self,
        h_instances: torch.Tensor,
        label: torch.Tensor = None,
        cov: torch.Tensor = None,
        instance_eval: bool = False,
        return_domain: bool = False,  # ✅ NEW: return domain logits/features when domain_adapt=True
        domain_tile_k: int = 256,      # ✅ NEW: sample K tiles for tile-level domain logits
    ):
        # --- sanitize inputs early
        h_instances = self._sanitize(h_instances, "h_instances")
        if cov is not None:
            cov = self._sanitize(cov, "cov")

        # 0) Force bag to be [N_inst, D] even if N_inst == 1
        if h_instances.dim() == 1:
            h_instances = h_instances.unsqueeze(0)  # [1, D]
        assert h_instances.dim() == 2, f"h_instances must be [N, D], got {tuple(h_instances.shape)}"

        # enforce float32 (prevents float64 covariates etc.)
        h_instances = h_instances.float()
        if cov is not None and torch.is_tensor(cov):
            cov = cov.float()

        # 1) Instance -> hidden
        h_hidden = self.pre_attn(h_instances)      # [N, H]
        h_hidden = self._sanitize(h_hidden, "h_hidden")
        if h_hidden.dim() == 1:
            h_hidden = h_hidden.unsqueeze(0)

        # 2) Attention
        A_raw_inst, _ = self.attn_mod(h_hidden)    # [N, 1]
        A_raw_inst = self._sanitize(A_raw_inst, "A_raw_inst")
        if A_raw_inst.dim() == 1:
            A_raw_inst = A_raw_inst.unsqueeze(1)

        # make [1, N]
        A_raw = A_raw_inst.transpose(1, 0)         # [1, N]

        # clamp logits before softmax (rare overflow protection)
        A_raw = torch.clamp(A_raw, min=-50.0, max=50.0)

        A_norm = F.softmax(A_raw, dim=1)           # [1, N]
        A_norm = self._sanitize(A_norm, "A_norm")

        # 3) Bag embedding (pre-cov)
        M = torch.mm(A_norm, h_hidden)             # [1, H]
        M = self._sanitize(M, "M")

        # 4) Covariates
        M_fused = M
        cov_emb = None

        if self.cov_mlp is not None and self.cov_dim > 0 and cov is not None:
            if cov.dim() == 1:
                cov = cov.unsqueeze(0)             # [1, C]
            cov = self._sanitize(cov, "cov(unsqueezed)")

            cov_emb = self.cov_mlp(cov)            # [1, cov_hidden]
            cov_emb = self._sanitize(cov_emb, "cov_emb")

            if self.cov_fusion == "concat":
                M_fused = torch.cat([M, cov_emb], dim=1)  # [1, H + cov_hidden]
            elif self.cov_fusion == "cox_additive":
                M_fused = M
            else:
                raise NotImplementedError(
                    f"cov_fusion='{self.cov_fusion}' not implemented."
                )

        M_fused = self._sanitize(M_fused, "M_fused")

        # 5) Task head(s)
        out = {
            "task_type": self.task_type,
            "bag_embed": M_fused,
            "attn_raw":  A_raw,
            "attn_norm": A_norm,
            "instance_eval": None
        }

        # ---- Optional domain-adversarial outputs (institution logits)
        # We expose logits but do not compute CE loss here.
        if return_domain and self.domain_adapt:
            # Always expose underlying features so you can debug UMAP / probes
            out["domain_feats_bag"] = M.detach()         # [1, H]
            out["domain_feats_tile"] = None              # (optional, can be large; we provide sampled logits instead)

            # Bag-level logits (recommended baseline)
            if self.domain_level in ("bag", "both"):
                xb = self.grl(M)                         # [1, H]
                logits_bag = self.domain_head_bag(xb)        # [1, K]
                logits_bag = self._sanitize(logits_bag, "domain_logits_bag")
                out["domain_logits_bag"] = logits_bag
            else:
                out["domain_logits_bag"] = None

            # Tile-level logits (sample tiles for stability)
            if self.domain_level in ("tile", "both"):
                N = h_hidden.size(0)
                k = int(domain_tile_k) if domain_tile_k is not None else 0
                k = max(0, min(k, N))

                if k == 0:
                    out["domain_logits_tile"] = None
                    out["domain_tile_idx"] = None
                else:
                    if k == N:
                        idx = torch.arange(N, device=h_hidden.device)
                    else:
                        if self.domain_tile_sampling == "random":
                            idx = torch.randperm(N, device=h_hidden.device)[:k]

                        elif self.domain_tile_sampling == "attn_top":
                            # use attention mass to pick top tiles
                            a = A_norm.view(-1)  # [N]
                            idx = torch.topk(a, k=k, dim=0)[1]

                        elif self.domain_tile_sampling == "attn_mix":
                            # half top-attn, half random (helps avoid overfitting to only salient regions)
                            k_top = k // 2
                            k_rand = k - k_top
                            a = A_norm.view(-1)
                            idx_top = torch.topk(a, k=k_top, dim=0)[1] if k_top > 0 else torch.empty(0, dtype=torch.long, device=h_hidden.device)

                            # sample random from remaining
                            if k_rand > 0:
                                mask = torch.ones(N, device=h_hidden.device, dtype=torch.bool)
                                mask[idx_top] = False
                                pool = torch.where(mask)[0]
                                if pool.numel() <= k_rand:
                                    idx_rand = pool
                                else:
                                    idx_rand = pool[torch.randperm(pool.numel(), device=h_hidden.device)[:k_rand]]
                            else:
                                idx_rand = torch.empty(0, dtype=torch.long, device=h_hidden.device)

                            idx = torch.cat([idx_top, idx_rand], dim=0)

                        else:
                            idx = torch.randperm(N, device=h_hidden.device)[:k]

                    Hs = h_hidden[idx]                     # [k, H]
                    xt = self.grl(Hs)
                    logits_tile = self.domain_head_tile(xt)  # [k, K]
                    logits_tile = self._sanitize(logits_tile, "domain_logits_tile")
                    out["domain_logits_tile"] = logits_tile
                    out["domain_tile_idx"] = idx.detach()

        if self.task_type == "classification":
            logits = self.head(M_fused)                  # [1, n_classes]
            logits = self._sanitize(logits, "logits")
            probs  = F.softmax(logits, dim=1)            # [1, n_classes]
            probs  = self._sanitize(probs, "probs")
            pred_label = torch.topk(logits, 1, dim=1)[1] # [1, 1]

            out.update({
                "logits": logits,
                "probs": probs,
                "pred_label": pred_label,
            })

            if instance_eval:
                if label is None:
                    raise ValueError("label is required for instance_eval in classification mode")

                label_flat = label.view(-1).long()
                onehot = F.one_hot(label_flat, num_classes=self.n_classes).squeeze(0)

                total_inst_loss = 0.0
                all_preds = []
                all_targets = []

                for class_idx, clf in enumerate(self.instance_classifiers):
                    if onehot[class_idx].item() == 1:
                        inst_loss, preds, targets = self._inst_eval_inclass(
                            A_norm, h_hidden, clf, self.k_sample
                        )
                    else:
                        if self.subtyping:
                            inst_loss, preds, targets = self._inst_eval_outclass(
                                A_norm, h_hidden, clf, self.k_sample
                            )
                        else:
                            continue

                    total_inst_loss = total_inst_loss + inst_loss
                    all_preds.append(preds.detach().cpu().numpy())
                    all_targets.append(targets.detach().cpu().numpy())

                if self.subtyping and len(self.instance_classifiers) > 0:
                    total_inst_loss = total_inst_loss / len(self.instance_classifiers)

                out["instance_eval"] = {
                    "instance_loss": total_inst_loss,
                    "inst_preds": np.concatenate(all_preds) if len(all_preds) else np.array([]),
                    "inst_targets": np.concatenate(all_targets) if len(all_targets) else np.array([]),
                }

        elif self.task_type == "survival":
            if self.cov_fusion == "cox_additive" and (cov_emb is not None) and (self.cov_to_risk is not None):
                risk_bag = self.head(M)                # [1, 1]
                risk_cov = self.cov_to_risk(cov_emb)   # [1, 1]
                risk = (risk_bag + risk_cov).view(-1)  # [1]
            else:
                risk = self.head(M_fused).view(-1)     # [1]

            risk = self._sanitize(risk, "risk")
            out.update({"risk": risk})

        elif self.task_type == "regression":
            z = self.head(M_fused).view(-1)            # [1] standardized (log) space
            z = self._sanitize(z, "pred_z")
            t = z * self.reg_y_std + self.reg_y_mean   # un-standardize
            if float(self.reg_y_log1p.item()) > 0.5:   # un-transform log1p -> raw
                t = torch.expm1(t)
            pred = self._sanitize(t, "pred")           # raw target units
            # pred_z is what the training loss is computed against (so the
            # trainable weights operate at unit scale); pred is for reporting.
            out.update({"pred": pred, "pred_z": z})

        return out