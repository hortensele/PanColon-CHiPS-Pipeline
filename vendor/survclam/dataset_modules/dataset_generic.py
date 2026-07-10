import os
import re
import math
import numpy as np
import pandas as pd
from scipy import stats
import torch
from torch.utils.data import Dataset, Subset

from utils.utils import generate_split, nth


############################################################
# Utility: resolve PT id from a row
############################################################
def _resolve_pt_id_from_row(row: pd.Series, pt_id_col: str) -> str:
    """
    Return the identifier that corresponds to the .pt filename stem.

    Example: if pt files are pt_files/<ID>.pt then pt_id_col should be the column
    in the master CSV that contains <ID>.
    """
    if pt_id_col not in row.index:
        raise KeyError(
            f"pt_id_col='{pt_id_col}' missing from row. "
            f"Available columns: {list(row.index)}"
        )
    return str(row[pt_id_col]).strip()


def _EMPTY_COV() -> torch.Tensor:
    """
    When return_institution=True and has_covariates=False, we return a 4-tuple
    (feats, cov_empty, y, inst_id) to avoid ambiguity with 3-tuples.

    Training code should treat covariates with numel()==0 as None.
    """
    return torch.empty((0,), dtype=torch.float32)


############################################################
# Utility: save_splits
############################################################
def save_splits(split_datasets, column_keys, filename, boolean_style=False):
    """
    Write which items were used for train/val(/test).

    Behavior:
      - bag_level='slide'   -> list slide_ids (from slide_data['slide_id'])
      - bag_level='patient' -> list UNIQUE '<case_id>_BAG' (derived from case_id)

    Works with:
      - FamilySplit
      - torch.utils.data.Subset(FamilySplit | Generic_WSIFamilyDataset)
    """
    def _extract_bag_ids(dset):
        # Unwrap Subset -> (base_ds, idxs)
        if isinstance(dset, Subset):
            base = dset.dataset
            idxs = list(dset.indices)
            bag_level = getattr(base, "bag_level", "slide")

            # IMPORTANT: patient Subset indices refer to patient_data, not slide_data
            if bag_level == "patient":
                if not hasattr(base, "patient_data"):
                    raise AttributeError("Underlying dataset lacks patient_data for patient bags")
                if "case_id" not in base.patient_data:
                    raise KeyError("patient_data must contain 'case_id'")
                case_ids = np.asarray(base.patient_data["case_id"]).astype(str)[idxs].tolist()
                bag_ids = sorted({f"{cid}_BAG" for cid in case_ids})
                return bag_ids

            # slide-level subset -> indices refer to slide_data
            if not hasattr(base, "slide_data"):
                raise AttributeError("Underlying dataset lacks slide_data")
            slide_df = base.slide_data.iloc[idxs].reset_index(drop=True)

        else:
            # FamilySplit or Generic_WSIFamilyDataset
            if not hasattr(dset, "slide_data"):
                raise AttributeError("Dataset lacks slide_data")
            slide_df = dset.slide_data.reset_index(drop=True)
            bag_level = getattr(dset, "bag_level", "slide")

        if bag_level == "patient":
            if "case_id" not in slide_df.columns:
                raise KeyError("case_id column required for patient-level bags")
            bag_ids = sorted({f"{cid}_BAG" for cid in slide_df["case_id"].astype(str).tolist()})
        else:
            if "slide_id" not in slide_df.columns:
                raise KeyError("slide_id column required for slide-level bags")
            bag_ids = slide_df["slide_id"].astype(str).tolist()

        return bag_ids

    # Keep only non-empty splits
    pairs = []
    for i, dset in enumerate(split_datasets):
        if dset is None:
            continue
        try:
            ids = _extract_bag_ids(dset)
        except Exception:
            ids = []
        if len(ids) > 0:
            pairs.append((column_keys[i], ids))

    if len(pairs) == 0:
        pd.DataFrame(columns=column_keys).to_csv(filename, index=False)
        print(f"[save_splits] Wrote empty split file: {filename}")
        return

    if not boolean_style:
        max_len = max(len(ids) for _, ids in pairs)
        data = {}
        for key, ids in pairs:
            col = ids + [np.nan] * (max_len - len(ids))
            data[key] = col
        df = pd.DataFrame(data)
    else:
        all_ids = []
        for _, ids in pairs:
            all_ids.extend(ids)
        uniq = sorted(set(all_ids))
        df = pd.DataFrame(False, index=uniq, columns=[k for k, _ in pairs])
        for key, ids in pairs:
            df.loc[ids, key] = True
        df = df.reset_index().rename(columns={"index": "bag_id"})

    df.to_csv(filename, index=False)
    print(f"[save_splits] Wrote {filename}")


############################################################
# Core unified dataset
############################################################
class Generic_WSIFamilyDataset(Dataset):
    """
    Unified dataset for classification / survival / regression.

    Key toggles:
      - patient_strat: if True, CV splits are drawn at patient level; else slide level.
      - bag_level:
            'slide'   -> each __getitem__ = one slide bag
            'patient' -> each __getitem__ = concatenated slides for a patient
      - pt_id_col:
            Column in master CSV that matches the .pt filename stem.
            (This decouples pt filenames from 'slide_id' which may be something else.)
      - return_institution:
            If True, __getitem__ returns institution_id as an extra field:
              • no covariates:  (feats, cov_empty, y, inst_id)   ✅ FIXED (was ambiguous 3-tuple)
              • with covariates:(feats, cov_vec, y, inst_id)
    """

    def __init__(
        self,
        csv_path,
        task_type='classification',      # 'classification' | 'survival' | 'regression'
        label_col='label',
        label_dict=None,
        ignore=None,
        time_col='time',
        event_col='event',
        target_col='target',
        covariate_cols=None,
        filter_dict=None,
        shuffle=False,
        seed=7,
        print_info=True,
        patient_strat=False,
        patient_voting='max',
        bag_level='slide',
        reg_bins=4,
        pt_id_col="slide_id",
        institution_col="institution",
        return_institution=False,
    ):
        super().__init__()

        assert task_type in ['classification', 'survival', 'regression'], \
            f"task_type must be one of 'classification','survival','regression', got {task_type}"
        assert bag_level in ['slide', 'patient'], \
            f"bag_level must be 'slide' or 'patient', got {bag_level}"

        self.task_type = task_type
        self.label_col = label_col
        self.label_dict = label_dict if label_dict is not None else {}
        self.ignore = ignore if ignore is not None else []

        self.time_col = time_col
        self.event_col = event_col
        self.target_col = target_col

        self.covariate_cols = covariate_cols if covariate_cols is not None else []
        self.has_covariates = len(self.covariate_cols) > 0

        self.patient_strat = patient_strat
        self.patient_voting = patient_voting
        self.bag_level = bag_level
        self.reg_bins = reg_bins
        self.seed = seed

        self.pt_id_col = pt_id_col
        self.institution_col = institution_col
        self.return_institution = bool(return_institution)

        self.split_gen = None

        # Will be populated if institution_col exists
        self.institution2id = None
        self.id2institution = None
        self.num_institutions = 0

        if self.task_type == 'classification':
            assert len(self.label_dict) > 0, "Must supply label_dict for classification!"
            self.num_classes = len(set(self.label_dict.values()))
        else:
            self.num_classes = None

        self.train_ids, self.val_ids, self.test_ids = ([], [], [])
        self.train_pat_ids, self.val_pat_ids, self.test_pat_ids = ([], [], [])
        self.data_dir = None  # set externally

        df = pd.read_csv(csv_path)
        df = self._filter_df(df, filter_dict if filter_dict is not None else {})

        # ✅ Ensure pt_id_col exists in master CSV
        if self.pt_id_col not in df.columns:
            raise KeyError(
                f"pt_id_col='{self.pt_id_col}' not found in clinical CSV. "
                f"Available columns: {list(df.columns)}"
            )

        # ✅ Institution encoding (optional but recommended)
        if self.institution_col in df.columns:
            df[self.institution_col] = df[self.institution_col].astype(str).fillna("UNKNOWN")
            insts = sorted(df[self.institution_col].unique().tolist())
            self.institution2id = {k: i for i, k in enumerate(insts)}
            self.id2institution = {i: k for k, i in self.institution2id.items()}
            self.num_institutions = len(self.institution2id)
            df["institution_id"] = df[self.institution_col].map(self.institution2id).astype(int)
        else:
            # If user wants institution but column missing, fail early
            if self.return_institution:
                raise KeyError(
                    f"return_institution=True but institution_col='{self.institution_col}' not found in CSV. "
                    f"Available columns: {list(df.columns)}"
                )

        if self.task_type == 'classification':
            df = self._prep_classification_df(df)

        if self.task_type == 'survival':
            if self.time_col not in df.columns or self.event_col not in df.columns:
                raise ValueError("Survival mode requires time_col and event_col in the CSV.")
            df[self.time_col] = df[self.time_col].astype(float)
            df[self.event_col] = df[self.event_col].astype(int)

        if self.task_type == 'regression':
            if self.target_col not in df.columns:
                raise ValueError(
                    f"Regression mode requires target_col '{self.target_col}' in the CSV. "
                    f"Available columns: {list(df.columns)}"
                )
            df[self.target_col] = pd.to_numeric(df[self.target_col], errors='coerce')
            if df[self.target_col].isna().all():
                raise ValueError(f"All values in '{self.target_col}' are NaN after coercion.")

        if shuffle:
            np.random.seed(seed)
            df = df.sample(frac=1).reset_index(drop=True)

        self.slide_data = df.reset_index(drop=True)

        self._build_patient_level_metadata()
        self._build_cls_ids()

        if print_info:
            self.summarize()

    ############################################################
    # Internal helpers
    ############################################################
    def _filter_df(self, df, filter_dict):
        if len(filter_dict) == 0:
            return df
        mask = np.full(len(df), True, dtype=bool)
        for key, val in filter_dict.items():
            mask &= df[key].isin(val)
        return df[mask].reset_index(drop=True)

    def _prep_classification_df(self, df):
        if self.label_col != 'label':
            df['label'] = df[self.label_col].copy()

        mask_ignore = df['label'].isin(self.ignore)
        df = df[~mask_ignore].reset_index(drop=True)

        for i in df.index:
            raw_key = df.loc[i, 'label']
            if raw_key not in self.label_dict:
                raise KeyError(f"Label '{raw_key}' not found in label_dict keys={list(self.label_dict.keys())}")
            df.at[i, 'label'] = self.label_dict[raw_key]

        df['label'] = df['label'].astype(int)
        return df

    def _build_patient_level_metadata(self):
        if "case_id" not in self.slide_data.columns:
            raise KeyError("CSV must contain 'case_id' for patient-level operations")

        grouped = self.slide_data.groupby('case_id').first().reset_index()
        patient_data = {'case_id': grouped['case_id'].values}

        if self.task_type == 'classification':
            patient_labels = []
            for case_id in patient_data['case_id']:
                idxs = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
                slide_labels = self.slide_data.loc[idxs, 'label'].values.astype(int)
                if self.patient_voting == 'max':
                    voted = slide_labels.max()
                elif self.patient_voting == 'maj':
                    voted = stats.mode(slide_labels, keepdims=True)[0]
                else:
                    raise NotImplementedError("patient_voting must be 'max' or 'maj'")
                patient_labels.append(int(voted))
            patient_data['label'] = np.array(patient_labels, dtype=int)

        elif self.task_type == 'survival':
            patient_data['time']  = grouped[self.time_col].values.astype(float)
            patient_data['event'] = grouped[self.event_col].values.astype(int)

        else:
            patient_data['target'] = grouped[self.target_col].astype(float).values

        if self.has_covariates:
            for cov_col in self.covariate_cols:
                if cov_col not in grouped.columns:
                    raise ValueError(f"Covariate column {cov_col} not found in CSV.")
                patient_data[cov_col] = grouped[cov_col].values

        # ✅ Patient-level institution_id (first per case, consistent with grouped)
        if "institution_id" in grouped.columns:
            patient_data["institution_id"] = grouped["institution_id"].astype(int).values

        self.patient_data = patient_data

        self.case_to_slide_idxs = {
            str(case_id): self.slide_data.index[self.slide_data['case_id'] == case_id].tolist()
            for case_id in self.patient_data['case_id']
        }

    def _build_cls_ids(self):
        if self.task_type == 'classification':
            num_classes = self.num_classes

            self.patient_cls_ids = [[] for _ in range(num_classes)]
            for i, lab in enumerate(self.patient_data['label']):
                self.patient_cls_ids[int(lab)].append(i)

            self.slide_cls_ids = [[] for _ in range(num_classes)]
            slide_labels = self.slide_data['label'].values.astype(int)
            for c in range(num_classes):
                self.slide_cls_ids[c] = np.where(slide_labels == c)[0]

        elif self.task_type == 'survival':
            self.patient_cls_ids = [[], []]
            for i, e in enumerate(self.patient_data['event']):
                self.patient_cls_ids[int(e)].append(i)

            events_slide = self.slide_data[self.event_col].astype(int).values
            self.slide_cls_ids = [[], []]
            for ev in [0, 1]:
                self.slide_cls_ids[ev] = np.where(events_slide == ev)[0]

            self.num_classes = 2
            self.label_dict = {0: 'censored', 1: 'event'}

        else:
            all_patients = list(range(len(self.patient_data['case_id'])))
            self.patient_cls_ids = [all_patients]
            self.slide_cls_ids = [np.arange(len(self.slide_data)).astype(int)]
            self.num_classes = 1
            self.label_dict = {0: 'all'}

    def filter_patients_with_missing_embeddings(self, pt_id_col: str = None, verbose: bool = True):
        """
        Remove patients (case_id) for which NONE of their slides have an embedding .pt file.

        Assumes embeddings are per-slide. In patient bag mode, we concatenate slide pt files.
        """
        if self.data_dir is None:
            raise RuntimeError("data_dir must be set before filtering missing embeddings.")

        eff_pt_id_col = pt_id_col if pt_id_col is not None else self.pt_id_col

        if eff_pt_id_col not in self.slide_data.columns:
            raise KeyError(
                f"pt_id_col='{eff_pt_id_col}' not found in slide_data columns: {list(self.slide_data.columns)}"
            )

        # helper: check existence of a single slide pt
        def _pt_exists(row):
            base_dir = self._resolve_backend_dir(row)
            sid = str(row[eff_pt_id_col])

            p1 = os.path.join(base_dir, f"{sid}.pt")
            if os.path.isfile(p1):
                return True
            p2 = os.path.join(base_dir, "pt_files", f"{sid}.pt")
            return os.path.isfile(p2)

        keep_case_ids = []
        dropped_case_ids = []

        for case_id, df_case in self.slide_data.groupby("case_id"):
            any_ok = False
            for _, row in df_case.iterrows():
                if _pt_exists(row):
                    any_ok = True
                    break
            if any_ok:
                keep_case_ids.append(case_id)
            else:
                dropped_case_ids.append(case_id)

        before_slides = len(self.slide_data)
        before_pats = len(self.patient_data["case_id"])

        # filter slide_data
        self.slide_data = self.slide_data[self.slide_data["case_id"].isin(keep_case_ids)].reset_index(drop=True)

        # rebuild everything dependent on slide_data
        self._build_patient_level_metadata()
        self._build_cls_ids()

        after_slides = len(self.slide_data)
        after_pats = len(self.patient_data["case_id"])

        if verbose:
            print(
                f"[filter_missing_embeddings] Dropped patients with 0 slide pt files: "
                f"{len(dropped_case_ids)} | kept patients: {after_pats}/{before_pats} | "
                f"slides: {after_slides}/{before_slides}"
            )
            if len(dropped_case_ids) > 0:
                print(f"[filter_missing_embeddings] Example dropped case_ids: {dropped_case_ids[:10]}")

    ############################################################
    # API
    ############################################################
    def __len__(self):
        return len(self.patient_data['case_id']) if self.bag_level == 'patient' else len(self.slide_data)

    def summarize(self):
        print(f"task_type: {self.task_type}")
        print(f"bag_level: {self.bag_level}")
        print(f"pt_id_col: {self.pt_id_col}")
        print(f"return_institution: {self.return_institution}")
        if self.institution2id is not None:
            print(f"institution_col: {self.institution_col}")
            print(f"num_institutions: {self.num_institutions}")
        print(f"Total slides: {len(self.slide_data)}")
        print(f"Unique patients: {len(np.unique(self.slide_data['case_id']))}")

    ############################################################
    # Splitting logic (unchanged from your snippet beyond earlier fixes)
    ############################################################
    def create_splits(
        self,
        k=3,
        val_num=None,
        test_num=None,
        label_frac=1.0,
        custom_test_ids=None,
    ):
        """
        Build self.split_gen that yields splits over stratification units
        (patients if self.patient_strat else slides).

        Features:
          • Fixed test set: sampled once per stratum (or provided via custom_test_ids),
            then reused for all folds.
          • Regression bins: stores per-unit bin IDs on self so you can audit them.
          • Safe when test_frac=0 (no test set created).
        """

        # -----------------------------
        # Helper: build regression bins
        # -----------------------------
        def _make_regression_bins(num_bins: int):
            # fetch targets + index universe depending on stratification unit
            if self.patient_strat:
                # patient-level targets (prefer precomputed; otherwise derive)
                if 'target' in self.patient_data:
                    targets_all = np.asarray(self.patient_data['target'], dtype=float)
                else:
                    targets_all = []
                    for case_id in self.patient_data['case_id']:
                        rows = self.slide_data[self.slide_data['case_id'] == case_id]
                        targets_all.append(float(rows[self.target_col].iloc[0]))
                    targets_all = np.asarray(targets_all, dtype=float)

                idx_universe = np.arange(len(self.patient_data['case_id']))
                samples = len(idx_universe)
                bin_level = 'patient'
            else:
                targets_all = np.asarray(self.slide_data[self.target_col].values, dtype=float)
                idx_universe = np.arange(len(self.slide_data))
                samples = len(idx_universe)
                bin_level = 'slide'

            # drop NaNs
            valid_mask = np.isfinite(targets_all)
            idx_universe = idx_universe[valid_mask]
            targets = targets_all[valid_mask]

            if targets.size == 0:
                bin_cls_ids = [idx_universe]
                base_count = np.array([len(idx_universe)], dtype=int)
                reg_bin_ids = np.full(samples, -1, dtype=int)
                reg_bin_ids[valid_mask] = 0
                return bin_cls_ids, base_count, samples, reg_bin_ids, bin_level

            unique_vals = np.unique(targets)
            eff_bins = int(max(1, min(num_bins, len(unique_vals))))
            if eff_bins <= 1:
                bin_cls_ids = [idx_universe]
                base_count = np.array([len(idx_universe)], dtype=int)
                reg_bin_ids = np.full(samples, -1, dtype=int)
                reg_bin_ids[valid_mask] = 0
                return bin_cls_ids, base_count, samples, reg_bin_ids, bin_level

            q = np.linspace(0.0, 1.0, eff_bins + 1)
            q_edges = np.quantile(targets, q)
            eps = 1e-8
            for i in range(1, len(q_edges)):
                if q_edges[i] <= q_edges[i - 1]:
                    q_edges[i] = q_edges[i - 1] + eps

            cutpoints = q_edges[1:-1]
            bin_ids_valid = np.digitize(targets, cutpoints, right=False)

            bin_cls_ids = []
            for b in range(eff_bins):
                sel = idx_universe[bin_ids_valid == b]
                bin_cls_ids.append(sel)

            base_count = np.array([len(x) for x in bin_cls_ids], dtype=int)
            if np.count_nonzero(base_count) <= 1:
                bin_cls_ids = [idx_universe]
                base_count = np.array([len(idx_universe)], dtype=int)
                bin_ids_valid = np.zeros_like(bin_ids_valid, dtype=int)
                eff_bins = 1

            reg_bin_ids = np.full(samples, -1, dtype=int)
            reg_bin_ids[idx_universe] = bin_ids_valid

            self.last_regression_base_count = base_count.copy()
            return bin_cls_ids, base_count, samples, reg_bin_ids, bin_level

        # -----------------------------
        # Pick strata (cls_ids) + counts
        # -----------------------------
        if self.task_type == 'regression':
            cls_ids, base_count, samples, reg_bin_ids, bin_level = _make_regression_bins(self.reg_bins)

            self.regression_bin_ids   = reg_bin_ids
            self.regression_bin_level = bin_level

            # convenient maps
            self.reg_bin_of_slide = {int(i): int(b) for i, b in enumerate(
                self.regression_bin_ids if bin_level == 'slide'
                else np.full(len(self.slide_data), -1, dtype=int)
            )}
            if bin_level == 'patient':
                case_ids_arr = np.asarray(self.patient_data['case_id']).astype(str)
                self.reg_bin_of_patient = {case_ids_arr[i]: int(reg_bin_ids[i]) for i in range(len(case_ids_arr))}
            else:
                self.reg_bin_of_patient = {}

            # auto val/test counts if not provided
            if val_num is None or test_num is None:
                default_val_frac = 0.1
                default_test_frac = 0.1
                val_arr  = np.round(base_count * default_val_frac).astype(int)
                test_arr = np.round(base_count * default_test_frac).astype(int)

                val_arr  = np.where((base_count >= 2) & (val_arr  == 0), 1, val_arr)
                test_arr = np.where((base_count >= 2) & (test_arr == 0), 1, test_arr)

                val_num  = val_arr
                test_num = test_arr
            else:
                val_num  = np.asarray(val_num,  dtype=int)
                test_num = np.asarray(test_num, dtype=int)
        else:
            if self.patient_strat:
                cls_ids = self.patient_cls_ids
                samples = len(self.patient_data['case_id'])
            else:
                cls_ids = self.slide_cls_ids
                samples = len(self.slide_data)
            val_num  = np.asarray(val_num,  dtype=int)
            test_num = np.asarray(test_num, dtype=int)

        # -----------------------------
        # Handle no-test-set case safely
        # -----------------------------
        force_test_from_custom = (custom_test_ids is not None) and (len(custom_test_ids) > 0)
        no_test = (
            (not force_test_from_custom) and (
                test_num is None
                or np.all(np.asarray(test_num) == 0)
                or (isinstance(test_num, (float, int)) and float(test_num) == 0.0)
            )
        )

        if no_test:
            print("[INFO] test_frac/test_num = 0 → No test set will be created.")
            fixed_test_ids = np.array([], dtype=int)
            cls_ids_minus_test = cls_ids
            self.fixed_test_ids_unit = ('patient' if self.patient_strat else 'slide')
        else:
            # --- Fix the test set once (per stratum) ---
            def _sample_per_stratum(strata_lists, take_counts, seed):
                rng = np.random.default_rng(seed)
                chosen = []
                for ids, n_take in zip(strata_lists, take_counts):
                    ids = np.asarray(ids, dtype=int)
                    n_take = int(n_take) if n_take is not None else 0
                    n_take = max(0, min(n_take, ids.size))
                    if n_take > 0 and ids.size > 0:
                        pick = rng.choice(ids, size=n_take, replace=False)
                        chosen.append(pick)
                if len(chosen) == 0:
                    return np.array([], dtype=int)
                return np.concatenate(chosen, axis=0)

            if force_test_from_custom:
                fixed_test_ids = np.array(custom_test_ids, dtype=int)
            else:
                fixed_test_ids = _sample_per_stratum(cls_ids, test_num, seed=self.seed)

            self.fixed_test_ids_unit = ('patient' if self.patient_strat else 'slide')

            cls_ids_minus_test = []
            for ids in cls_ids:
                ids = np.asarray(ids, dtype=int)
                if ids.size == 0:
                    cls_ids_minus_test.append(ids)
                else:
                    keep_mask = ~np.isin(ids, fixed_test_ids)
                    cls_ids_minus_test.append(ids[keep_mask])

        self.fixed_test_ids = fixed_test_ids

        # -----------------------------
        # Hand off to generator
        # -----------------------------
        remaining_per_stratum = np.array([len(a) for a in cls_ids_minus_test], dtype=int)
        val_num = np.asarray(val_num, dtype=int)
        val_num = np.minimum(val_num, remaining_per_stratum)

        settings = dict(
            n_splits=k,
            val_num=val_num,
            test_num=None,
            label_frac=label_frac,
            seed=self.seed,
            custom_test_ids=fixed_test_ids,
            cls_ids=cls_ids_minus_test,
            samples=samples,
        )
        self.split_gen = generate_split(**settings)

        # -----------------------------
        # Debug info
        # -----------------------------
        if self.task_type == 'regression':
            bc = getattr(self, "last_regression_base_count", None)
            print(f"[DEBUG] Regression bins created: {len(cls_ids)} | counts={bc.tolist() if bc is not None else 'NA'}")
            if fixed_test_ids.size > 0:
                per_bin_test = {}
                for b_idx, ids in enumerate(cls_ids):
                    ids = np.asarray(ids, dtype=int)
                    per_bin_test[b_idx] = int(np.isin(fixed_test_ids, ids).sum())
                print(f"[DEBUG] Fixed test size: {fixed_test_ids.size} (per-bin approx after removal: {per_bin_test})")
        else:
            if fixed_test_ids.size > 0:
                print(f"[DEBUG] Fixed test size: {fixed_test_ids.size}")

    def set_splits(self, start_from=None):
        if not hasattr(self, "split_gen") or self.split_gen is None:
            raise RuntimeError("set_splits() called before create_splits(). Call create_splits(...) first.")

        ids_tuple = nth(self.split_gen, start_from) if start_from is not None else next(self.split_gen)
        train_ids_raw, val_ids_raw, test_ids_raw = ids_tuple

        if self.patient_strat:
            self.train_pat_ids = list(train_ids_raw)
            self.val_pat_ids   = list(val_ids_raw)
            self.test_pat_ids  = list(test_ids_raw)

            self.train_ids = self._expand_patient_ids_to_slides(self.train_pat_ids)
            self.val_ids   = self._expand_patient_ids_to_slides(self.val_pat_ids)
            self.test_ids  = self._expand_patient_ids_to_slides(self.test_pat_ids)
        else:
            self.train_ids = list(train_ids_raw)
            self.val_ids   = list(val_ids_raw)
            self.test_ids  = list(test_ids_raw)
            self.train_pat_ids, self.val_pat_ids, self.test_pat_ids = ([], [], [])

    def _expand_patient_ids_to_slides(self, pat_id_list):
        slide_indices = []
        for pat_idx in pat_id_list:
            case_id = self.patient_data['case_id'][pat_idx]
            case_slide_idxs = self.slide_data.index[self.slide_data['case_id'] == case_id].tolist()
            slide_indices.extend(case_slide_idxs)
        return slide_indices

    ############################################################
    # Split recreation from CSV (supports slide_id, case_id, <case>_BAG)
    ############################################################
    def get_split_from_df(self, all_splits, split_key='train'):
        split_col = all_splits[split_key].dropna().reset_index(drop=True)
        if len(split_col) == 0:
            return None

        case_set = set(self.slide_data["case_id"].astype(str).tolist()) if "case_id" in self.slide_data.columns else set()

        wanted_slide_idxs = []
        for bag_id in split_col.tolist():
            s = str(bag_id).strip()

            if "slide_id" in self.slide_data.columns:
                m = np.where(self.slide_data["slide_id"].astype(str) == s)[0]
                if len(m) > 0:
                    wanted_slide_idxs.extend(m.tolist())
                    continue

            if s.endswith("_BAG") and "case_id" in self.slide_data.columns:
                case_id = s[:-4]
                m2 = np.where(self.slide_data["case_id"].astype(str) == case_id)[0]
                if len(m2) > 0:
                    wanted_slide_idxs.extend(m2.tolist())
                    continue

            if s in case_set:
                m3 = np.where(self.slide_data["case_id"].astype(str) == s)[0]
                if len(m3) > 0:
                    wanted_slide_idxs.extend(m3.tolist())
                    continue

        wanted_slide_idxs = sorted(set(wanted_slide_idxs))
        df_slice = self.slide_data.iloc[wanted_slide_idxs].reset_index(drop=True)

        return FamilySplit(
            df_slice,
            data_dir=self.data_dir,
            task_type=self.task_type,
            time_col=self.time_col,
            event_col=self.event_col,
            target_col=self.target_col,
            covariate_cols=self.covariate_cols,
            bag_level=self.bag_level,
            pt_id_col=self.pt_id_col,
            institution_col=self.institution_col,
            institution2id=self.institution2id,
            return_institution=self.return_institution,
        )

    def get_merged_split_from_df(self, all_splits, split_keys=('train',)):
        merged_ids = []
        for split_key in split_keys:
            split_col = all_splits[split_key].dropna().reset_index(drop=True).tolist()
            merged_ids.extend(split_col)

        if len(merged_ids) == 0:
            return None

        case_set = set(self.slide_data["case_id"].astype(str).tolist()) if "case_id" in self.slide_data.columns else set()

        wanted_slide_idxs = []
        for bag_id in merged_ids:
            s = str(bag_id).strip()

            if "slide_id" in self.slide_data.columns:
                m = np.where(self.slide_data["slide_id"].astype(str) == s)[0]
                if len(m) > 0:
                    wanted_slide_idxs.extend(m.tolist())
                    continue

            if s.endswith("_BAG") and "case_id" in self.slide_data.columns:
                case_id = s[:-4]
                m2 = np.where(self.slide_data["case_id"].astype(str) == case_id)[0]
                if len(m2) > 0:
                    wanted_slide_idxs.extend(m2.tolist())
                    continue

            if s in case_set:
                m3 = np.where(self.slide_data["case_id"].astype(str) == s)[0]
                if len(m3) > 0:
                    wanted_slide_idxs.extend(m3.tolist())
                    continue

        wanted_slide_idxs = sorted(set(wanted_slide_idxs))
        df_slice = self.slide_data.iloc[wanted_slide_idxs].reset_index(drop=True)

        return FamilySplit(
            df_slice,
            data_dir=self.data_dir,
            task_type=self.task_type,
            time_col=self.time_col,
            event_col=self.event_col,
            target_col=self.target_col,
            covariate_cols=self.covariate_cols,
            bag_level=self.bag_level,
            pt_id_col=self.pt_id_col,
            institution_col=self.institution_col,
            institution2id=self.institution2id,
            return_institution=self.return_institution,
        )

    ############################################################
    # Build loaders from splits
    ############################################################
    def return_splits(
        self,
        from_id: bool = False,
        csv_path: str = None,
        train_ids: list = None,
        val_ids: list = None,
        test_ids: list = None,
        seed: int = 1,
        bag_level: str = None,
    ):
        if csv_path is not None:
            split_df = pd.read_csv(csv_path)
            train_col = split_df['train'].dropna().tolist() if 'train' in split_df.columns else []
            val_col   = split_df['val'].dropna().tolist()   if 'val'   in split_df.columns else []
            test_col  = split_df['test'].dropna().tolist()  if 'test'  in split_df.columns else []
            train_ids = train_ids or train_col
            val_ids   = val_ids   or val_col
            test_ids  = test_ids  or test_col

        eff_bag_level = bag_level if bag_level is not None else self.bag_level
        assert eff_bag_level in ('slide', 'patient')

        def _resolve_patient_indices(id_list):
            """
            Returns patient_data indices.
        
            Accepts tokens that may be:
              - patient indices (0,1,2,...)                       [from_id=False, numeric]
              - case_id                                           [from_id=True]
              - case_id_BAG                                       [from_id=True]
              - slide_id (legacy) -> resolved via slide_to_case   [from_id=True]
            """
            if id_list is None or len(id_list) == 0:
                return []
        
            # normalize strings, drop empties
            toks = []
            for tok in id_list:
                s = str(tok).strip()
                if s == "" or s.lower() == "nan":
                    continue
                toks.append(s)
            if len(toks) == 0:
                return []
        
            case_ids_arr = np.asarray(self.patient_data["case_id"]).astype(str)
        
            # If NOT from_id, we expect integer patient indices.
            # But allow strings as long as they are numeric.
            if not from_id:
                if not all(re.fullmatch(r"-?\d+", t) for t in toks):
                    raise ValueError(
                        f"[return_splits] from_id=False but found non-numeric patient ids (examples): {toks[:5]}"
                    )
                return sorted(set(map(int, toks)))
        
            # from_id=True: tokens are IDs (case_id / case_id_BAG / slide_id)
            slide_to_case = None
            if "slide_id" in self.slide_data.columns and "case_id" in self.slide_data.columns:
                slide_to_case = dict(
                    zip(self.slide_data["slide_id"].astype(str),
                        self.slide_data["case_id"].astype(str))
                )
        
            resolved_case_ids = []
            case_set = set(case_ids_arr.tolist())
        
            for t in toks:
                # case_id_BAG
                if t.endswith("_BAG"):
                    resolved_case_ids.append(t[:-4])
                    continue
        
                # direct case_id
                if t in case_set:
                    resolved_case_ids.append(t)
                    continue
        
                # slide_id -> case_id (legacy support)
                if slide_to_case is not None and t in slide_to_case:
                    resolved_case_ids.append(slide_to_case[t])
                    continue
        
            resolved_case_ids = sorted(set(resolved_case_ids))
        
            out_idx = []
            for c in resolved_case_ids:
                hits = np.where(case_ids_arr == c)[0]
                out_idx.extend(hits.tolist())
        
            out_idx = sorted(set(out_idx))
        
            if len(out_idx) == 0:
                # helpful debug
                raise ValueError(
                    f"[return_splits] from_id=True but resolved 0 patients. "
                    f"Examples toks={toks[:10]} | "
                    f"case_id examples in dataset={case_ids_arr[:10].tolist()}"
                )
        
            return out_idx

            case_ids_arr = np.asarray(self.patient_data['case_id']).astype(str)

            slide_to_case = None
            if 'slide_id' in self.slide_data.columns and 'case_id' in self.slide_data.columns:
                slide_to_case = dict(
                    zip(self.slide_data['slide_id'].astype(str),
                        self.slide_data['case_id'].astype(str))
                )

            resolved_case_ids = []
            for tok in map(str, id_list):
                t = tok.strip()
                if t.endswith("_BAG"):
                    resolved_case_ids.append(t[:-4])
                    continue
                if t in case_ids_arr:
                    resolved_case_ids.append(t)
                    continue
                if slide_to_case is not None and t in slide_to_case:
                    resolved_case_ids.append(slide_to_case[t])
                    continue

            resolved_case_ids = sorted(set(resolved_case_ids))
            out_idx = []
            for c in resolved_case_ids:
                hits = np.where(case_ids_arr == c)[0]
                out_idx.extend(hits.tolist())
            return sorted(set(out_idx))

        if eff_bag_level == 'patient':
            train_pat = _resolve_patient_indices(train_ids)
            val_pat   = _resolve_patient_indices(val_ids)
            test_pat  = _resolve_patient_indices(test_ids)

            def _slice_by_pat(pat_idx_list, split_name):
                if pat_idx_list is None or len(pat_idx_list) == 0:
                    raise ValueError(
                        f"[return_splits] Patient-level: '{split_name}' resolved to 0 patients."
                    )
                case_arr = np.asarray(self.patient_data['case_id'])
                selected_cases = set(case_arr[pat_idx_list].astype(str).tolist())
                mask = self.slide_data['case_id'].astype(str).isin(selected_cases)
                df_slice = self.slide_data.loc[mask].reset_index(drop=True)
                return FamilySplit(
                    df_slice,
                    data_dir=self.data_dir,
                    task_type=self.task_type,
                    time_col=self.time_col,
                    event_col=self.event_col,
                    target_col=self.target_col,
                    covariate_cols=self.covariate_cols,
                    bag_level='patient',
                    pt_id_col=self.pt_id_col,
                    institution_col=self.institution_col,
                    institution2id=self.institution2id,
                    return_institution=self.return_institution,
                )

            train_split = _slice_by_pat(train_pat, 'train')
            val_split   = _slice_by_pat(val_pat,   'val')
            test_split  = _slice_by_pat(test_pat,  'test') if (test_pat is not None and len(test_pat) > 0) else None
            return train_split, val_split, test_split

        # -----------------------------
        # SLIDE-LEVEL SPLITS
        # -----------------------------
        def _resolve_slide_indices(id_list):
            """
            Resolve split IDs into slide_data row indices.

            Accepts tokens that may be:
              - slide_id (legacy)
              - pt_id_col values (self.pt_id_col)
              - case_id
              - case_id_BAG
            """
            if id_list is None or len(id_list) == 0:
                return []

            slide_df = self.slide_data.reset_index(drop=True)

            # maps for fast lookup
            # 1) pt_id_col -> indices
            pt_map = {}
            if self.pt_id_col in slide_df.columns:
                for i, v in enumerate(slide_df[self.pt_id_col].astype(str).tolist()):
                    pt_map.setdefault(v, []).append(i)

            # 2) slide_id -> indices (legacy support)
            slideid_map = {}
            if "slide_id" in slide_df.columns:
                for i, v in enumerate(slide_df["slide_id"].astype(str).tolist()):
                    slideid_map.setdefault(v, []).append(i)

            # 3) case_id -> indices
            case_map = {}
            if "case_id" in slide_df.columns:
                for i, v in enumerate(slide_df["case_id"].astype(str).tolist()):
                    case_map.setdefault(v, []).append(i)

            out = []
            for tok in map(str, id_list):
                t = tok.strip()
                if t == "" or t.lower() == "nan":
                    continue

                # case_id_BAG
                if t.endswith("_BAG"):
                    cid = t[:-4]
                    if cid in case_map:
                        out.extend(case_map[cid])
                    continue

                # exact case_id
                if t in case_map:
                    out.extend(case_map[t])
                    continue

                # exact pt_id_col (preferred)
                if t in pt_map:
                    out.extend(pt_map[t])
                    continue

                # exact slide_id (legacy)
                if t in slideid_map:
                    out.extend(slideid_map[t])
                    continue

            return sorted(set(out))

        def _slice_by_slides(slide_idx_list, split_name):
            if slide_idx_list is None or len(slide_idx_list) == 0:
                raise ValueError(f"[return_splits] Slide-level: '{split_name}' resolved to 0 slides.")

            df_slice = self.slide_data.iloc[slide_idx_list].reset_index(drop=True)
            return FamilySplit(
                df_slice,
                data_dir=self.data_dir,
                task_type=self.task_type,
                time_col=self.time_col,
                event_col=self.event_col,
                target_col=self.target_col,
                covariate_cols=self.covariate_cols,
                bag_level="slide",
                pt_id_col=self.pt_id_col,
                institution_col=self.institution_col,
                institution2id=self.institution2id,
                return_institution=self.return_institution,
            )

        train_slides = _resolve_slide_indices(train_ids)
        val_slides   = _resolve_slide_indices(val_ids)
        test_slides  = _resolve_slide_indices(test_ids) if (test_ids is not None and len(test_ids) > 0) else []

        # An external-inference split may populate only one set (e.g. all 'test').
        # Return None for empty sets instead of raising, mirroring 'test' below;
        # _build_requested_split_df skips None splits and only uses which_sets.
        train_split = _slice_by_slides(train_slides, "train") if len(train_slides) > 0 else None
        val_split   = _slice_by_slides(val_slides,   "val")   if len(val_slides)   > 0 else None
        test_split  = _slice_by_slides(test_slides,  "test")  if len(test_slides)  > 0 else None
        return train_split, val_split, test_split

    ############################################################
    # __getitem__
    ############################################################
    def __getitem__(self, idx):
        if self.bag_level == 'patient':
            return self._build_patient_bag(idx)
        else:
            row = self.slide_data.iloc[idx]
            return self._build_item_from_row(row)

    def _build_patient_bag(self, pat_idx):
        case_id = self.patient_data['case_id'][pat_idx]
        slide_indices = self.case_to_slide_idxs[str(case_id)]

        feats_list = []
        missing = 0

        for sidx in slide_indices:
            row_s = self.slide_data.iloc[sidx]

            pt_id = _resolve_pt_id_from_row(row_s, self.pt_id_col)

            try:
                feats_s = _load_pt(self._resolve_backend_dir(row_s), pt_id)
            except FileNotFoundError:
                missing += 1
                continue

            if feats_s is None or feats_s.numel() == 0:
                missing += 1
                continue

            feats_list.append(feats_s)

        if len(feats_list) == 0:
            raise RuntimeError(
                f"[Patient bag] All slide embeddings missing for case_id={case_id} "
                f"(n_slides={len(slide_indices)})"
            )

        feats = torch.cat(feats_list, dim=0)

        if self.task_type == 'classification':
            y = torch.tensor(int(self.patient_data['label'][pat_idx]), dtype=torch.long)
        elif self.task_type == 'survival':
            y = torch.tensor([
                float(self.patient_data['time'][pat_idx]),
                float(self.patient_data['event'][pat_idx])
            ], dtype=torch.float32)
        else:
            y = torch.tensor(float(self.patient_data['target'][pat_idx]), dtype=torch.float32)

        inst_id = None
        if self.return_institution:
            if "institution_id" not in self.patient_data:
                raise KeyError("return_institution=True but patient_data lacks 'institution_id'")
            inst_id = torch.tensor(int(self.patient_data["institution_id"][pat_idx]), dtype=torch.long)

        if self.has_covariates:
            cov_list = [float(self.patient_data[cov][pat_idx]) for cov in self.covariate_cols]
            cov_vec = torch.tensor(cov_list, dtype=torch.float32)
            if self.return_institution:
                return feats, cov_vec, y, inst_id
            return feats, cov_vec, y
        else:
            if self.return_institution:
                # ✅ FIX: always 4-tuple when return_institution=True (prevents survival scalar bug)
                return feats, _EMPTY_COV(), y, inst_id
            return feats, y

    def _build_item_from_row(self, row):
        pt_id = _resolve_pt_id_from_row(row, self.pt_id_col)
        feats = _load_pt(self._resolve_backend_dir(row), pt_id)

        if self.task_type == 'classification':
            y = torch.tensor(int(row['label']), dtype=torch.long)
        elif self.task_type == 'survival':
            y = torch.tensor([float(row[self.time_col]), float(row[self.event_col])], dtype=torch.float32)
        else:
            y = torch.tensor(float(row[self.target_col]), dtype=torch.float32)

        inst_id = None
        if self.return_institution:
            if "institution_id" not in row.index:
                raise KeyError("return_institution=True but slide_data lacks 'institution_id'")
            inst_id = torch.tensor(int(row["institution_id"]), dtype=torch.long)

        if self.has_covariates:
            cov_list = [float(row[c]) for c in self.covariate_cols]
            cov_vec = torch.tensor(cov_list, dtype=torch.float32)
            if self.return_institution:
                return feats, cov_vec, y, inst_id
            return feats, cov_vec, y
        else:
            if self.return_institution:
                # ✅ FIX: always 4-tuple when return_institution=True (prevents survival scalar bug)
                return feats, _EMPTY_COV(), y, inst_id
            return feats, y

    def _resolve_backend_dir(self, row):
        if isinstance(self.data_dir, dict):
            if 'source' not in row:
                raise ValueError("Row missing 'source' column while data_dir is a dict.")
            return self.data_dir[row['source']]
        return self.data_dir


############################################################
# Split dataset used for loaders
############################################################
class FamilySplit(Dataset):
    """
    Train/val/test split dataset (slice of slide_data) that respects bag_level.

    If return_institution=True, __getitem__ returns institution_id as an extra field:
      • no covariates:  (feats, cov_empty, y, inst_id)   ✅ FIXED (was ambiguous 3-tuple)
      • with covariates:(feats, cov_vec, y, inst_id)

    IMPORTANT:
      - For stable institution IDs across folds/splits, pass institution2id from the parent dataset.
    """

    def __init__(
        self,
        slide_data: pd.DataFrame,
        data_dir,
        task_type,
        time_col,
        event_col,
        target_col,
        covariate_cols,
        bag_level='slide',
        pt_id_col="slide_id",
        institution_col="institution",
        institution2id=None,
        return_institution=False,
    ):
        super().__init__()
        assert bag_level in ['slide', 'patient']
        self.slide_data = slide_data.reset_index(drop=True)
        self.data_dir = data_dir
        self.task_type = task_type
        self.time_col = time_col
        self.event_col = event_col
        self.target_col = target_col
        self.covariate_cols = covariate_cols
        self.bag_level = bag_level
        self.pt_id_col = pt_id_col

        self.institution_col = institution_col
        self.return_institution = bool(return_institution)

        # Build / apply institution mapping (stable if provided)
        self.institution2id = institution2id
        if self.institution_col in self.slide_data.columns:
            self.slide_data[self.institution_col] = self.slide_data[self.institution_col].astype(str).fillna("UNKNOWN")
            if self.institution2id is None:
                insts = sorted(self.slide_data[self.institution_col].unique().tolist())
                self.institution2id = {k: i for i, k in enumerate(insts)}
            self.slide_data["institution_id"] = self.slide_data[self.institution_col].map(self.institution2id).astype(int)
        else:
            if self.return_institution:
                raise KeyError(
                    f"return_institution=True but institution_col='{self.institution_col}' missing in split slide_data."
                )

        grouped = self.slide_data.groupby('case_id').first().reset_index()
        self.patient_data = {'case_id': grouped['case_id'].values}

        if self.task_type == 'classification' and 'label' in self.slide_data.columns:
            patient_labels = []
            for case_id in self.patient_data['case_id']:
                idxs = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
                slide_labels = self.slide_data.loc[idxs, 'label'].values.astype(int)
                voted = slide_labels.max()
                patient_labels.append(int(voted))
            self.patient_data['label'] = np.array(patient_labels, dtype=int)

        elif self.task_type == 'survival':
            self.patient_data['time']  = grouped[self.time_col].astype(float).values
            self.patient_data['event'] = grouped[self.event_col].astype(int).values
        else:
            self.patient_data['target'] = grouped[self.target_col].astype(float).values

        if len(self.covariate_cols) > 0:
            for cov_col in self.covariate_cols:
                if cov_col not in grouped.columns:
                    raise ValueError(f"Covariate column {cov_col} missing for patient-level bag.")
                self.patient_data[cov_col] = grouped[cov_col].values

        # ✅ patient-level institution_id (first per case)
        if "institution_id" in grouped.columns:
            self.patient_data["institution_id"] = grouped["institution_id"].astype(int).values

        self.case_to_slide_idxs = {
            str(case_id): self.slide_data.index[self.slide_data['case_id'] == case_id].tolist()
            for case_id in self.patient_data['case_id']
        }

    def __len__(self):
        return len(self.patient_data['case_id']) if self.bag_level == 'patient' else len(self.slide_data)

    def __getitem__(self, idx):
        if self.bag_level == 'patient':
            return self._build_patient_bag(idx)
        else:
            row = self.slide_data.iloc[idx]
            return self._build_slide_bag(row)

    def _build_patient_bag(self, pat_idx):
        case_id = str(self.patient_data['case_id'][pat_idx])
        slide_indices = self.case_to_slide_idxs[case_id]

        feats_list = []
        missing = 0

        for sidx in slide_indices:
            row_s = self.slide_data.iloc[sidx]
            pt_id = _resolve_pt_id_from_row(row_s, self.pt_id_col)

            try:
                feats_s = _load_pt(self._resolve_backend_dir(row_s), pt_id)
            except FileNotFoundError:
                missing += 1
                continue

            if feats_s is None or feats_s.numel() == 0:
                missing += 1
                continue

            feats_list.append(feats_s)

        if len(feats_list) == 0:
            raise RuntimeError(
                f"[Patient bag / split] All slide embeddings missing for case_id={case_id} "
                f"(n_slides={len(slide_indices)})"
            )

        feats = torch.cat(feats_list, dim=0)

        if self.task_type == 'classification':
            y_torch = torch.tensor(int(self.patient_data['label'][pat_idx]), dtype=torch.long)
        elif self.task_type == 'survival':
            y_torch = torch.tensor(
                [float(self.patient_data['time'][pat_idx]),
                 float(self.patient_data['event'][pat_idx])],
                dtype=torch.float32
            )
        else:
            y_torch = torch.tensor(float(self.patient_data['target'][pat_idx]), dtype=torch.float32)

        inst_id = None
        if self.return_institution:
            if "institution_id" not in self.patient_data:
                raise KeyError("return_institution=True but patient_data lacks 'institution_id'")
            inst_id = torch.tensor(int(self.patient_data["institution_id"][pat_idx]), dtype=torch.long)

        if len(self.covariate_cols) > 0:
            cov_list = [float(self.patient_data[cov][pat_idx]) for cov in self.covariate_cols]
            cov_vec = torch.tensor(cov_list, dtype=torch.float32)
            if self.return_institution:
                return feats, cov_vec, y_torch, inst_id
            return feats, cov_vec, y_torch
        else:
            if self.return_institution:
                # ✅ FIX: always 4-tuple when return_institution=True (prevents survival scalar bug)
                return feats, _EMPTY_COV(), y_torch, inst_id
            return feats, y_torch

    def _build_slide_bag(self, row):
        pt_id = _resolve_pt_id_from_row(row, self.pt_id_col)
        feats = _load_pt(self._resolve_backend_dir(row), pt_id)

        if self.task_type == 'classification':
            y_torch = torch.tensor(int(row['label']), dtype=torch.long)
        elif self.task_type == 'survival':
            y_torch = torch.tensor(
                [float(row[self.time_col]), float(row[self.event_col])],
                dtype=torch.float32
            )
        else:
            y_torch = torch.tensor(float(row[self.target_col]), dtype=torch.float32)

        inst_id = None
        if self.return_institution:
            if "institution_id" not in row.index:
                raise KeyError("return_institution=True but slide_data lacks 'institution_id'")
            inst_id = torch.tensor(int(row["institution_id"]), dtype=torch.long)

        if len(self.covariate_cols) > 0:
            cov_list = [float(row[c]) for c in self.covariate_cols]
            cov_vec = torch.tensor(cov_list, dtype=torch.float32)
            if self.return_institution:
                return feats, cov_vec, y_torch, inst_id
            return feats, cov_vec, y_torch
        else:
            if self.return_institution:
                # ✅ FIX: always 4-tuple when return_institution=True (prevents survival scalar bug)
                return feats, _EMPTY_COV(), y_torch, inst_id
            return feats, y_torch

    def _resolve_backend_dir(self, row):
        if isinstance(self.data_dir, dict):
            if 'source' not in row:
                raise ValueError("Row missing 'source' column while data_dir is a dict.")
            return self.data_dir[row['source']]
        return self.data_dir


############################################################
# Feature loading helper
############################################################
def _load_pt(base_dir, slide_id):
    """
    Load patch embeddings for a slide (slide_id here means "pt_id").

    Supported layouts:
      (A) base_dir/<pt_id>.pt
      (B) base_dir/pt_files/<pt_id>.pt
    """
    if base_dir is None:
        return torch.tensor([], dtype=torch.float32)

    p1 = os.path.join(base_dir, f"{slide_id}.pt")
    if os.path.isfile(p1):
        feats = torch.load(p1)
        if not torch.is_tensor(feats):
            feats = torch.tensor(feats, dtype=torch.float32)
        return feats

    p2 = os.path.join(base_dir, "pt_files", f"{slide_id}.pt")
    if os.path.isfile(p2):
        feats = torch.load(p2)
        if not torch.is_tensor(feats):
            feats = torch.tensor(feats, dtype=torch.float32)
        return feats

    raise FileNotFoundError(
        f"Could not find embeddings for pt_id='{slide_id}'. Tried:\n"
        f"  - {p1}\n"
        f"  - {p2}"
    )
