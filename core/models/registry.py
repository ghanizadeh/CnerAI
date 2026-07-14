"""
core/models/registry.py
Loads available models + default params from config/models.yaml.

Public API
----------
get_model_names(task_type)         -> list[str]
get_default_params(task_type, name) -> dict
get_model_instance(task_type, name, params) -> sklearn estimator
"""

from __future__ import annotations
import importlib
from pathlib import Path
import yaml


_REGISTRY: dict | None = None


def _load_registry() -> dict:
    global _REGISTRY
    if _REGISTRY is None:
        yaml_path = Path(__file__).parent.parent.parent / "config" / "models.yaml"
        with open(yaml_path, "r") as f:
            _REGISTRY = yaml.safe_load(f)
    return _REGISTRY


def get_model_names(task_type: str) -> list[str]:
    """Return list of model names for 'classification' or 'regression'."""
    reg = _load_registry()
    return list(reg.get(task_type, {}).keys())


def get_default_params(task_type: str, model_name: str) -> dict:
    """Return default hyperparameters for a given model."""
    reg = _load_registry()
    return dict(reg[task_type][model_name].get("default_params", {}))


def get_param_grid(task_type: str, model_name: str) -> dict:
    """Return hyperparameter search grid for GridSearchCV."""
    reg = _load_registry()
    return dict(reg[task_type][model_name].get("param_grid", {}))


def get_model_instance(
    task_type: str,
    model_name: str,
    params: dict | None = None,
) -> object:
    """
    Instantiate and return an sklearn estimator.

    Parameters
    ----------
    params : if None, default_params from YAML are used
    """
    reg = _load_registry()
    class_path: str = reg[task_type][model_name]["class"]
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)

    effective_params = get_default_params(task_type, model_name)
    if params:
        effective_params.update(params)

    return cls(**effective_params)
