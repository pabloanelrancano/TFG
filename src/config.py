# src/config.py
# Pablo Anel Rancano - TFG HAR
"""Load config.yaml and provide CLI override helpers."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _parse_yaml_simple(path: Path) -> Dict[str, Any]:
    """Read a YAML file and return it as a dict."""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Defaults (used if config.yaml is missing)
_DEFAULTS: Dict[str, Any] = {
    "dataset_path": "../DataSets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset",
    "processed_dir": "data/processed",
    "results_dir": "results",
    "evaluation": {
        "n_splits": 5,
        "random_state": 42,
        "scoring": "accuracy",
    },
    "models": ["random_forest", "logistic_regression", "linear_svm", "knn"],
    "pipelines": ["baseline_561", "interpretable", "tsfresh"],
    "interpretable": {
        "include_frequency": True,
        "sampling_rate": 50,
    },
    "tsfresh": {
        "extraction_settings": "efficient",
        "nan_threshold": 0.5,
        "impute_remaining": True,
    },
    "tsfeatures_r": {
        "rscript_path": "Rscript",
    },
}

_config: Optional[Dict[str, Any]] = None


def _find_config_yaml() -> Optional[Path]:
    """Look for config.yaml in CWD or in the project root."""
    candidates = [
        Path.cwd() / "config.yaml",
        Path(__file__).resolve().parent.parent / "config.yaml",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load config from YAML, falling back to defaults."""
    cfg = dict(_DEFAULTS)

    yaml_path = config_path or _find_config_yaml()
    if yaml_path and yaml_path.is_file():
        try:
            file_cfg = _parse_yaml_simple(yaml_path)
            if file_cfg:
                _deep_merge(cfg, file_cfg)
        except Exception as exc:
            print(f"[config] WARNING: could not parse {yaml_path}: {exc}", file=sys.stderr)

    return cfg


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base in-place."""
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val


def get_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Return the global config (loaded once, cached)."""
    global _config
    if _config is None:
        _config = load_config(config_path)
    return _config


def reset_config() -> None:
    """Force reload on next get_config() call."""
    global _config
    _config = None


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the CLI flags shared by all entry points."""
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Override dataset root path (folder with train/ and test/).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Override results output directory.",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Override processed features cache directory.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=None,
        help="Number of GroupKFold splits.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: auto-detect).",
    )
    return parser


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Overwrite config values with whatever was passed via CLI flags."""
    if getattr(args, "dataset_path", None):
        cfg["dataset_path"] = args.dataset_path
    if getattr(args, "results_dir", None):
        cfg["results_dir"] = args.results_dir
    if getattr(args, "processed_dir", None):
        cfg["processed_dir"] = args.processed_dir
    if getattr(args, "n_splits", None):
        cfg["evaluation"]["n_splits"] = args.n_splits
    if getattr(args, "config", None):
        reset_config()
        cfg.update(load_config(Path(args.config)))
    return cfg


def dataset_path(cfg: Optional[Dict[str, Any]] = None) -> Path:
    cfg = cfg or get_config()
    return Path(cfg["dataset_path"])


def results_dir(cfg: Optional[Dict[str, Any]] = None) -> Path:
    cfg = cfg or get_config()
    return Path(cfg["results_dir"])


def processed_dir(cfg: Optional[Dict[str, Any]] = None) -> Path:
    cfg = cfg or get_config()
    return Path(cfg["processed_dir"])
