"""Lightweight pure-Python Hydra compatibility layer for environments without pip."""

from __future__ import annotations
import functools
import importlib
import os
import sys
from typing import Any, Callable, Optional
from omegaconf import DictConfig, load_yaml_file, _parse_scalar


class _HydraUtils:
    @staticmethod
    def instantiate(config: Optional[DictConfig], *args: Any, **kwargs: Any) -> Any:
        if not config:
            return None
        target = config.get("_target_") if isinstance(config, dict) else getattr(config, "_target_", None)
        if not target:
            return None
        module_path, attr_name = target.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls_or_fn = getattr(module, attr_name)
        call_kwargs = {k: v for k, v in config.items() if k != "_target_"}
        call_kwargs.update(kwargs)
        return cls_or_fn(*args, **call_kwargs)


utils = _HydraUtils()


def compose_config(config_path: str = "cfg", config_name: str = "config", overrides: Optional[list[str]] = None) -> DictConfig:
    """Load Hydra YAML configuration and apply defaults + CLI overrides."""
    base_dir = os.path.abspath(config_path)
    main_file = os.path.join(base_dir, f"{config_name}.yaml")
    cfg = load_yaml_file(main_file)

    defaults = cfg.get("defaults", [])
    if isinstance(defaults, list):
        for entry in defaults:
            if isinstance(entry, dict):
                for group, name in entry.items():
                    if group.startswith("override ") or name is None:
                        continue
                    group_key = group.split("@")[-1]
                    group_dir = group.split("@")[0]
                    sub_path = os.path.join(base_dir, group_dir, f"{name}.yaml")
                    if os.path.exists(sub_path):
                        cfg[group_key] = load_yaml_file(sub_path)

    for arg in (overrides or []):
        if "=" in arg and not arg.startswith("-"):
            key_path, raw_val = arg.split("=", 1)
            val = _parse_scalar(raw_val)
            parts = key_path.split(".")
            if len(parts) == 1 and os.path.exists(os.path.join(base_dir, parts[0], f"{val}.yaml")):
                cfg[parts[0]] = load_yaml_file(os.path.join(base_dir, parts[0], f"{val}.yaml"))
            else:
                cur = cfg
                for p in parts[:-1]:
                    if p not in cur or not isinstance(cur[p], dict):
                        cur[p] = DictConfig()
                    cur = cur[p]
                cur[parts[-1]] = val

    return DictConfig(cfg)


def main(
    version_base: Optional[str] = None,
    config_path: str = "cfg",
    config_name: str = "config",
) -> Callable[[Callable[[DictConfig], Any]], Callable[[], Any]]:
    """Decorator matching @hydra.main(version_base=None, config_path=..., config_name=...)."""

    def decorator(fn: Callable[[DictConfig], Any]) -> Callable[[], Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if args:
                return fn(*args, **kwargs)
            caller_dir = os.path.dirname(os.path.abspath(sys.modules[fn.__module__].__file__ or os.getcwd()))
            resolved_cfg_path = os.path.join(caller_dir, config_path) if not os.path.isabs(config_path) else config_path
            cli_overrides = [a for a in sys.argv[1:] if "=" in a]
            cfg = compose_config(config_path=resolved_cfg_path, config_name=config_name, overrides=cli_overrides)
            return fn(cfg)

        return wrapper

    return decorator


__all__ = ["main", "utils", "compose_config", "DictConfig"]
