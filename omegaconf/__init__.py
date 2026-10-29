"""Lightweight pure-Python OmegaConf compatibility layer for environments without pip."""

from __future__ import annotations
import os
import re
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional


class DictConfig(dict):
    """Dictionary with attribute-style access and OmegaConf-compatible helpers."""

    def __init__(self, initial: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__()
        data = dict(initial or {})
        data.update(kwargs)
        for k, v in data.items():
            self[k] = self._wrap(v)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, DictConfig):
            return value
        if isinstance(value, dict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        return value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(f"'DictConfig' object has no attribute '{name}'") from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = self._wrap(value)

    def get(self, key: str, default: Any = None) -> Any:
        return super().get(key, default)


def _parse_scalar(raw: str) -> Any:
    val = raw.strip()
    if not val or val.lower() in ("null", "none", "~"):
        return None
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    # Handle ${oc.env:VAR,default}
    env_match = re.match(r"^\$\{oc\.env:([^,}]+)(?:,([^}]*))?\}$", val)
    if env_match:
        var_name = env_match.group(1).strip()
        fallback = env_match.group(2)
        env_val = os.environ.get(var_name)
        if env_val is not None:
            return env_val
        return _parse_scalar(fallback) if fallback is not None else None
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    try:
        if "." in val or "e" in val.lower():
            return float(val)
        return int(val)
    except ValueError:
        return val


def _strip_inline_comment(line: str) -> str:
    in_single = False
    in_double = False
    for idx, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            if idx == 0 or line[idx - 1].isspace():
                return line[:idx]
    return line


def load_yaml_file(filepath: str) -> DictConfig:
    """Minimal pure-Python YAML loader supporting Hydra config files."""
    if not os.path.exists(filepath):
        return DictConfig()
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    root: Dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    for raw_line in lines:
        cleaned = _strip_inline_comment(raw_line).rstrip()
        if not cleaned.strip():
            continue
        indent = len(cleaned) - len(cleaned.lstrip(" "))
        stripped = cleaned.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if stripped.startswith("- "):
            item_str = stripped[2:].strip()
            if isinstance(parent, dict):
                # Convert last empty dict entry to list if needed
                pass
            if isinstance(parent, list):
                if ":" in item_str:
                    k, v = item_str.split(":", 1)
                    parent.append({k.strip(): _parse_scalar(v)})
                else:
                    parent.append(_parse_scalar(item_str))
            continue

        if ":" in stripped:
            key, val_part = stripped.split(":", 1)
            key = key.strip()
            val_part = val_part.strip()
            if val_part == "":
                # Peek ahead to see if next non-empty line is a list item
                new_container: Any = {}
                parent[key] = new_container
                stack.append((indent, new_container))
            else:
                parent[key] = _parse_scalar(val_part)

    # Fix defaults list if parsed as dict
    if "defaults" in root and isinstance(root["defaults"], dict) and not root["defaults"]:
        defaults_list = []
        in_defaults = False
        for raw_line in lines:
            cleaned = _strip_inline_comment(raw_line).rstrip()
            if not cleaned.strip():
                continue
            s = cleaned.strip()
            if s.startswith("defaults:"):
                in_defaults = True
                continue
            if in_defaults:
                if s.startswith("- "):
                    item = s[2:].strip()
                    if ":" in item:
                        k, v = item.split(":", 1)
                        defaults_list.append({k.strip(): _parse_scalar(v)})
                    else:
                        defaults_list.append(_parse_scalar(item))
                elif not cleaned.startswith(" "):
                    in_defaults = False
        root["defaults"] = defaults_list

    return DictConfig(root)


class OmegaConf:
    """Helper class matching OmegaConf static methods."""

    @staticmethod
    def create(obj: Optional[Mapping[str, Any]] = None) -> DictConfig:
        return DictConfig(obj)

    @staticmethod
    def load(filepath: str) -> DictConfig:
        return load_yaml_file(filepath)

    @staticmethod
    def to_container(cfg: Any, resolve: bool = True) -> Any:
        if isinstance(cfg, DictConfig):
            return {k: OmegaConf.to_container(v, resolve=resolve) for k, v in cfg.items()}
        if isinstance(cfg, list):
            return [OmegaConf.to_container(v, resolve=resolve) for v in cfg]
        return cfg


__all__ = ["DictConfig", "OmegaConf", "load_yaml_file"]
