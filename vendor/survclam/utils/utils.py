import math
import collections
from itertools import islice

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (
    DataLoader,
    Sampler,
    WeightedRandomSampler,
    RandomSampler,
    SequentialSampler,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


########################################
# Samplers
########################################

class SubsetSequentialSampler(Sampler):
    """
    Samples elements sequentially from a given list of indices, without replacement.

    Args
    ----
    indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def seed_torch(seed: int = 1):
    """
    Set random seed for reproducibility across torch, numpy, and random.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


########################################
# Collate functions (UPDATED)
#
# We support three task styles:
#  - classification: label is a class index
#  - survival: label is a (time, event) tensor
#  - regression: label is a float target
#
# We ALSO support optional covariates per bag.
#
# ✅ NEW: optional domain/institution ID (inst_id) per bag, for domain-adversarial training.
#
# Supported dataset item shapes:
#
# Classification:
#   (feats, label)
#   (feats, cov, label)
#   (feats, label, inst_id)
#   (feats, cov, label, inst_id)
#
# Survival:
#   (feats, surv)
#   (feats, cov, surv)
#   (feats, surv, inst_id)
#   (feats, cov, surv, inst_id)
#
# Regression:
#   (feats, target)
#   (feats, cov, target)
#   (feats, target, inst_id)
#   (feats, cov, target, inst_id)
#
# Where:
#   feats     : Tensor [N_inst, D]
#   cov       : Tensor [C]
#   label     : LongTensor scalar
#   surv      : Tensor [2] (time, event)
#   target    : float or Tensor scalar
#   inst_id   : int (0..K-1) or Tensor scalar
#
# Return shapes:
#   - classification -> (bag_feats, [cov?], label, [inst_id?])
#   - survival       -> (bag_feats, [cov?], surv,  [inst_id?])
#   - regression     -> (bag_feats, [cov?], target,[inst_id?])
#
# IMPORTANT:
#   - We keep batch_size=1 behavior (1 bag per step) as before.
#   - For safety, we always return inst_id as torch.long scalar when present.
########################################

def _stack_bag_feats(batch, feat_index=0):
    """
    Default behavior: concatenate per-instance features across B items,
    where each item[feat_index] is [N_i, D] -> result [sum_i N_i, D].

    BUT if batch_size==1, just return that directly (faster).
    """
    if len(batch) == 1:
        return batch[0][feat_index]
    return torch.cat([item[feat_index] for item in batch], dim=0)


def _stack_vector_or_scalar(batch, getter, dtype=torch.float32):
    """
    Helper to stack covariates / targets / survival tensors while
    gracefully handling batch_size==1.

    getter(item) should return a tensor-like object.
    """
    if len(batch) == 1:
        single = getter(batch[0])
        return torch.as_tensor(single, dtype=dtype)
    return torch.stack(
        [torch.as_tensor(getter(item), dtype=dtype) for item in batch],
        dim=0
    )


def _has_inst_id(item_len: int) -> bool:
    """
    Convention:
      - 2: no cov, no inst_id
      - 3: either cov OR inst_id depending on task-specific position
      - 4: cov + inst_id
    We disambiguate by assuming: inst_id (if present) is ALWAYS the last element.
    """
    return item_len in (3, 4)


def _get_inst_id_from_item(item):
    inst = item[-1]
    # robust: allow python int, numpy scalar, tensor
    inst_t = torch.as_tensor(inst, dtype=torch.long)
    # keep scalar
    return inst_t.view(-1)[0]


def collate_classification(batch):
    """
    Handles dataset items shaped like:
        (feats, label)
        (feats, covariates, label)
        (feats, label, inst_id)
        (feats, covariates, label, inst_id)

    Returns:
        (bag_feats, label)
        (bag_feats, covariates, label)
        (bag_feats, label, inst_id)
        (bag_feats, covariates, label, inst_id)
    """
    n = len(batch[0])

    if n == 2:
        # (feats, label)
        bag_feats = _stack_bag_feats(batch, feat_index=0)
        labels = _stack_vector_or_scalar(batch, getter=lambda it: it[1], dtype=torch.long)
        return bag_feats, labels.squeeze(0)

    if n == 3:
        # either (feats, cov, label) OR (feats, label, inst_id)
        # disambiguate by dtype/shape: cov is float vector, label is long/int scalar
        # We enforce: inst_id is LAST if present -> so (feats, label, inst_id).
        # Therefore for n==3 we decide:
        #   - if last looks like inst_id (int-ish), treat as inst_id; else treat as label.
        # We’ll just follow the “inst_id is last” rule and assume:
        #   (feats, cov, label)  when middle is float-like vector
        #   (feats, label, inst_id) when middle is label-like scalar
        mid = batch[0][1]
        mid_t = torch.as_tensor(mid)

        if mid_t.ndim == 0:  # label-like
            bag_feats = _stack_bag_feats(batch, feat_index=0)
            labels = _stack_vector_or_scalar(batch, getter=lambda it: it[1], dtype=torch.long)
            inst_id = _stack_vector_or_scalar(batch, getter=_get_inst_id_from_item, dtype=torch.long)
            return bag_feats, labels.squeeze(0), inst_id.squeeze(0)
        else:
            bag_feats = _stack_bag_feats(batch, feat_index=0)
            covariates = _stack_vector_or_scalar(batch, getter=lambda it: it[1], dtype=torch.float32)
            labels = _stack_vector_or_scalar(batch, getter=lambda it: it[2], dtype=torch.long)
            return bag_feats, covariates.squeeze(0), labels.squeeze(0)

    if n == 4:
        # (feats, cov, label, inst_id)
        bag_feats = _stack_bag_feats(batch, feat_index=0)
        covariates = _stack_vector_or_scalar(batch, getter=lambda it: it[1], dtype=torch.float32)
        labels = _stack_vector_or_scalar(batch, getter=lambda it: it[2], dtype=torch.long)
        inst_id = _stack_vector_or_scalar(batch, getter=_get_inst_id_from_item, dtype=torch.long)
        return bag_feats, covariates.squeeze(0), labels.squeeze(0), inst_id.squeeze(0)

    raise ValueError(f"Unexpected classification item length: {n}")


def collate_survival(batch):
    """
    Handles:
        (feats, survival_tensor[time,event])
        (feats, covariates, survival_tensor[time,event])
        (feats, survival_tensor[time,event], inst_id)
        (feats, covariates, survival_tensor[time,event], inst_id)

    Returns either:
        (bag_feats, survival)
        (bag_feats, covariates, survival)
        (bag_feats, survival, inst_id)
        (bag_feats, covariates, survival, inst_id)
    """
    n = len(batch[0])

    if n == 2:
        bag_feats = _stack_bag_feats(batch, feat_index=0)
        survival = _stack_vector_or_scalar(batch, getter=lambda it: it[1], dtype=torch.float32)
        return bag_feats, survival.squeeze(0)

    if n == 3:
        # either (feats, cov, surv) OR (feats, surv, inst_id)
        mid = batch[0][1]
        mid_t = torch.as_tensor(mid)

        if mid_t.ndim == 1 and mid_t.numel() == 2:
            # (feats, surv, inst_id)
            bag_feats = _stack_bag_feats(batch, feat_index=0)
            survival = _stack_vector_or_scalar(batch, getter=lambda it: it[1], dtype=torch.float32)
            inst_id = _stack_vector_or_scalar(batch, getter=_get_inst_id_from_item, dtype=torch.long)
            return bag_feats, survival.squeeze(0), inst_id.squeeze(0)
        else:
            # (feats, cov, surv)
            bag_feats = _stack_bag_feats(batch, feat_index=0)
            covariates = _stack_vector_or_scalar(batch, getter=lambda it: it[1], dtype=torch.float32)
            survival = _stack_vector_or_scalar(batch, getter=lambda it: it[2], dtype=torch.float32)
            return bag_feats, covariates.squeeze(0), survival.squeeze(0)

    if n == 4:
        bag_feats = _stack_bag_feats(batch, feat_index=0)
        covariates = _stack_vector_or_scalar(batch, getter=lambda it: it[1], dtype=torch.float32)
        survival = _stack_vector_or_scalar(batch, getter=lambda it: it[2], dtype=torch.float32)
        inst_id = _stack_vector_or_scalar(batch, getter=_get_inst_id_from_item, dtype=torch.long)
        return bag_feats, covariates.squeeze(0), survival.squeeze(0), inst_id.squeeze(0)

    raise ValueError(f"Unexpected survival item length: {n}")


def collate_regression(batch):
    """
    Handles:
        (feats, target_float)
        (feats, covariates, target_float)
        (feats, target_float, inst_id)
        (feats, covariates, target_float, inst_id)

    Returns:
        (bag_feats, target)
        (bag_feats, covariates, target)
        (bag_feats, target, inst_id)
        (bag_feats, covariates, target, inst_id)
    """
    n = len(batch[0])

    if n == 2:
        bag_feats = _stack_bag_feats(batch, feat_index=0)
        target = _stack_vector_or_scalar(batch, getter=lambda it: it[1], dtype=torch.float32)
        return bag_feats, target.squeeze(0)

    if n == 3:
        # either (feats, cov, target) OR (feats, target, inst_id)
        mid = batch[0][1]
        mid_t = torch.as_tensor(mid)

        if mid_t.ndim == 0:
            # (feats, target, inst_id)
            bag_feats = _stack_bag_feats(batch, feat_index=0)
            target = _stack_vector_or_scalar(batch, getter=lambda it: it[1], dtype=torch.float32)
            inst_id = _stack_vector_or_scalar(batch, getter=_get_inst_id_from_item, dtype=torch.long)
            return bag_feats, target.squeeze(0), inst_id.squeeze(0)
        else:
            # (feats, cov, target)
            bag_feats = _stack_bag_feats(batch, feat_index=0)
            covariates = _stack_vector_or_scalar(batch, getter=lambda it: it[1], dtype=torch.float32)
            target = _stack_vector_or_scalar(batch, getter=lambda it: it[2], dtype=torch.float32)
            return bag_feats, covariates.squeeze(0), target.squeeze(0)

    if n == 4:
        bag_feats = _stack_bag_feats(batch, feat_index=0)
        covariates = _stack_vector_or_scalar(batch, getter=lambda it: it[1], dtype=torch.float32)
        target = _stack_vector_or_scalar(batch, getter=lambda it: it[2], dtype=torch.float32)
        inst_id = _stack_vector_or_scalar(batch, getter=_get_inst_id_from_item, dtype=torch.long)
        return bag_feats, covariates.squeeze(0), target.squeeze(0), inst_id.squeeze(0)

    raise ValueError(f"Unexpected regression item length: {n}")


########################################################
# Legacy collate fns kept for backward compat
########################################################

def collate_MIL(batch):
    """
    LEGACY survival-style collate.
    Returns (concat_feats, list_of_survival_tensors).
    """
    data = torch.cat([item[0] for item in batch], dim=0)
    survival = [item[1].clone().detach().float() for item in batch]
    return data, survival

def collate_MIL_cluster(batch):
    img = torch.cat([torch.tensor(item[0]) for item in batch], dim=0)
    cluster_ids = torch.cat([item[1] for item in batch], dim=0).type(torch.LongTensor)
    label = torch.LongTensor([item[2] for item in batch])
    return [img, cluster_ids, label]

def collate_MIL_graph(batch):
    img = batch[0][0]
    label = torch.LongTensor([item[1] for item in batch])
    return [img, label]

def collate_features(batch):
    img = torch.cat([torch.tensor(item[0]) for item in batch], dim=0)
    coords = np.vstack([item[1] for item in batch])
    return [img, coords]


########################################
# Loader builders (UPDATED)
########################################

def get_simple_loader(dataset, batch_size=1, num_workers=1, task_type='classification'):
    kwargs = {'num_workers': num_workers, 'pin_memory': (device.type == "cuda")}

    if task_type == 'classification':
        cfn = collate_classification
    elif task_type == 'survival':
        cfn = collate_survival
    elif task_type == 'regression':
        cfn = collate_regression
    else:
        cfn = None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        collate_fn=cfn,
        **kwargs
    )
    return loader


def get_split_loader(split_dataset,
                     training=False,
                     testing=False,
                     weighted=False,
                     task_type=None,
                     domain_adapt: bool = False):
    """
    Builds a MIL-style DataLoader for a (train/val/test) FamilySplit.

    ✅ NEW: domain_adapt flag is *not strictly required* because collate fns
    auto-detect inst_id if the dataset returns it, but keeping this flag lets
    you sanity check behavior / document intent.
    """
    # Decide sampler
    if weighted and task_type == 'classification':
        has_bins = (
            (getattr(split_dataset, "bag_level", "slide") == "patient" and hasattr(split_dataset, "patient_cls_ids"))
            or
            (getattr(split_dataset, "bag_level", "slide") == "slide" and hasattr(split_dataset, "slide_cls_ids"))
        )
        if has_bins:
            weights = make_weights_for_balanced_classes_split(split_dataset)
            data_sampler = WeightedRandomSampler(weights, len(weights))
            shuffle_flag = False
        else:
            # fallback if bins missing
            data_sampler = RandomSampler(split_dataset) if training else SequentialSampler(split_dataset)
            shuffle_flag = False
    else:
        if training:
            data_sampler = RandomSampler(split_dataset)
            shuffle_flag = False  # sampler overrides shuffle
        else:
            data_sampler = SequentialSampler(split_dataset)
            shuffle_flag = False

    # Decide collate_fn
    if task_type == 'classification':
        cfn = collate_classification
    elif task_type == 'survival':
        cfn = collate_survival
    elif task_type == 'regression':
        cfn = collate_regression
    else:
        # fallback to legacy behavior
        return DataLoader(
            split_dataset,
            batch_size=1,
            shuffle=training,
            collate_fn=None
        )

    # IMPORTANT: for MIL bags we keep batch_size=1
    loader = DataLoader(
        split_dataset,
        batch_size=1,
        sampler=data_sampler,
        shuffle=shuffle_flag,
        collate_fn=cfn,
        num_workers=0 if device.type != "cuda" else 4,
        pin_memory=(device.type == "cuda"),
    )
    return loader


########################################
# Optimizer / model utils
########################################

def get_optim(model, args):
    """
    Returns Adam or SGD over trainable params.

    We assume:
      args.opt in {"adam","sgd"}
      args.lr
      args.reg  (weight_decay)
    """
    params = filter(lambda p: p.requires_grad, model.parameters())
    if args.opt == "adam":
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.reg)
    elif args.opt == "sgd":
        optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.reg)
    else:
        raise NotImplementedError(f"Unsupported optimizer {args.opt}")
    return optimizer


def print_network(net):
    """
    Pretty-print the model and count parameters.
    """
    num_params = 0
    num_params_train = 0
    print(net)

    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n

    print('Total number of parameters: {}'.format(num_params))
    print('Total number of trainable parameters: {}'.format(num_params_train))


########################################
# Split utilities / misc
########################################

def generate_split(cls_ids, val_num, test_num, samples, n_splits=5,
                   seed=7, label_frac=1.0, custom_test_ids=None):
    """
    Generate stratified train/val/test indices for cross-validation.
    (unchanged)
    """
    indices = np.arange(samples).astype(int)

    if custom_test_ids is None and any(t > 0 for t in test_num):
        np.random.seed(seed)
        all_test_ids = []
        for c in range(len(test_num)):
            possible_indices = np.intersect1d(cls_ids[c], indices)
            if len(possible_indices) < test_num[c]:
                raise ValueError(
                    f"Not enough samples in class/bin {c} for requested test_num {test_num[c]}"
                )
            test_ids = np.random.choice(possible_indices, test_num[c], replace=False)
            all_test_ids.extend(test_ids)
        all_test_ids = np.array(all_test_ids)
    elif custom_test_ids is not None:
        all_test_ids = np.array(custom_test_ids)
    else:
        all_test_ids = np.array([])

    available_indices = np.setdiff1d(indices, all_test_ids)

    for _ in range(n_splits):
        all_val_ids = []
        sampled_train_ids = []

        for c in range(len(val_num)):
            possible_indices = np.intersect1d(cls_ids[c], available_indices)

            if len(possible_indices) < val_num[c]:
                raise ValueError(
                    f"Not enough samples in class/bin {c} for requested val_num {val_num[c]}"
                )

            val_ids = np.random.choice(possible_indices, val_num[c], replace=False)
            all_val_ids.extend(val_ids)

            remaining_ids = np.setdiff1d(possible_indices, val_ids)

            if label_frac == 1.0:
                sampled_train_ids.extend(remaining_ids)
            else:
                sample_num = math.ceil(len(remaining_ids) * label_frac)
                sampled_train_ids.extend(remaining_ids[:sample_num])

        yield list(sampled_train_ids), list(all_val_ids), list(all_test_ids)


def nth(iterator, n, default=None):
    if n is None:
        return collections.deque(iterator, maxlen=0)
    else:
        return next(islice(iterator, n, None), default)


def calculate_error(Y_hat, Y):
    """
    Classification error = 1 - accuracy.
    """
    error = 1.0 - Y_hat.float().eq(Y.float()).float().mean().item()
    return error


def make_weights_for_balanced_classes_split(dataset):
    bag_level = getattr(dataset, "bag_level", "slide")

    if bag_level == "patient":
        if not hasattr(dataset, "patient_cls_ids") or dataset.patient_cls_ids is None:
            raise ValueError(
                "Weighted sampling requested with bag_level='patient' but dataset.patient_cls_ids is missing."
            )
        cls_ids = dataset.patient_cls_ids
        labels_source = dataset.patient_data.get("label", None)
        if labels_source is None:
            raise ValueError("patient_data['label'] missing; cannot weight patients.")
        all_labels = np.asarray(labels_source, dtype=int)
    else:
        if not hasattr(dataset, "slide_cls_ids") or dataset.slide_cls_ids is None:
            raise ValueError(
                "Weighted sampling requested with bag_level='slide' but dataset.slide_cls_ids is missing."
            )
        cls_ids = dataset.slide_cls_ids
        all_labels = dataset.slide_data["label"].astype(int).values

    N = float(len(dataset))

    class_vals = []
    for c_list in cls_ids:
        if len(c_list) == 0:
            class_vals.append(None)
        else:
            idx0 = int(np.asarray(c_list).ravel()[0])
            class_vals.append(int(all_labels[idx0]))

    label_to_class_index = {lab: j for j, lab in enumerate(class_vals) if lab is not None}

    counts = [len(c) for c in cls_ids]
    if any(c == 0 for c in counts):
        raise ValueError(f"Found empty class in cls_ids: counts={counts}, class_vals={class_vals}")

    weight_per_class = [N / c for c in counts]

    weights = [0.0] * int(N)
    for i in range(len(dataset)):
        y = int(all_labels[i])
        j = label_to_class_index.get(y, None)
        if j is None:
            raise ValueError(f"Label {y} not found in label_to_class_index={label_to_class_index}")
        weights[i] = weight_per_class[j]

    return torch.DoubleTensor(weights)


def initialize_weights(module):
    """
    Xavier-normal init for Linear layers and sane defaults for BatchNorm1d.
    Call this on a model or submodule.
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)