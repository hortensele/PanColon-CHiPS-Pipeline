"""Configuration loading and path resolution.

A single YAML file (see config/pipeline.yaml) drives the whole pipeline. This
module loads it, resolves relative paths against the repo root, and exposes a
small dot-accessible wrapper so step modules can read `cfg.paths.work_dir`.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required to read the config. `pip install pyyaml` "
        "(it ships in both stage environments)."
    ) from exc


class DotDict(dict):
    """dict whose keys are also accessible as attributes (recursively)."""

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
            self[name] = value
        return value

    def __setattr__(self, name, value):
        self[name] = value


# Config keys that hold filesystem paths and should be resolved to absolute
# paths (relative entries are taken relative to repo_root).
_PATH_KEYS = {
    ("paths", "wsi_dir"),
    ("paths", "work_dir"),
    ("paths", "clinical_csv"),
    ("tools", "deeppath_root"),
    ("tools", "hpl_root"),
    ("tools", "survclam_root"),
    ("weights", "root"),
    ("weights", "hpl_checkpoint"),
    ("weights", "hpl_reference_h5"),
    ("weights", "hpl_artifact_reference_h5"),
    ("weights", "hpl_folds_pickle"),
    ("weights", "survclam_runs_root"),
}


def _find_repo_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    # repo root = parent of this package directory
    return Path(__file__).resolve().parent.parent


def load_config(config_path: str) -> DotDict:
    """Load a pipeline config, resolve paths, and validate required fields."""
    config_path = os.path.abspath(os.path.expanduser(config_path))
    if not os.path.isfile(config_path):
        raise SystemExit(f"Config not found: {config_path}")
    with open(config_path) as fh:
        raw = yaml.safe_load(fh) or {}

    cfg = DotDict(raw)
    repo_root = _find_repo_root(raw.get("repo_root") or "")
    cfg["repo_root"] = str(repo_root)
    cfg["_config_path"] = config_path

    for section, key in _PATH_KEYS:
        sec = cfg.get(section)
        if not isinstance(sec, dict):
            continue
        val = sec.get(key)
        if not val:
            continue
        p = Path(os.path.expanduser(str(val)))
        if not p.is_absolute():
            p = repo_root / p
        sec[key] = str(p)

    return cfg


def require(cfg: DotDict, dotted: str):
    """Fetch cfg['a']['b'] via 'a.b', raising a clear error if missing/empty."""
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise SystemExit(f"Config is missing required key: {dotted}")
        node = node[part]
    if node in ("", None, "CHANGE_ME"):
        raise SystemExit(f"Config key '{dotted}' must be set (currently {node!r}).")
    return node
