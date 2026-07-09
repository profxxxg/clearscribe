"""User-saved enhancement presets, persisted to ~/.clearscribe/presets.json."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

from clearscribe.enhance import PRESETS, EnhanceSettings


def _config_dir() -> Path:
    return Path(os.environ.get("CLEARSCRIBE_CONFIG_DIR",
                               str(Path.home() / ".clearscribe")))


def _presets_file() -> Path:
    return _config_dir() / "presets.json"


def load_user_presets() -> dict[str, EnhanceSettings]:
    """Load saved presets. Unknown/removed fields are ignored (forward compat)."""
    try:
        data = json.loads(_presets_file().read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    valid = {f.name for f in dataclasses.fields(EnhanceSettings)}
    presets: dict[str, EnhanceSettings] = {}
    for name, vals in data.items():
        if isinstance(vals, dict):
            presets[name] = EnhanceSettings(
                **{k: v for k, v in vals.items() if k in valid})
    return presets


def save_user_preset(name: str, settings: EnhanceSettings) -> None:
    name = name.strip()
    if not name:
        raise ValueError("Preset name cannot be empty.")
    if name in PRESETS:
        raise ValueError(f"'{name}' is a built-in preset — pick another name.")
    try:
        data = json.loads(_presets_file().read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    data[name] = dataclasses.asdict(settings)
    _config_dir().mkdir(parents=True, exist_ok=True)
    _presets_file().write_text(json.dumps(data, indent=2), encoding="utf-8")


def delete_user_preset(name: str) -> bool:
    presets_path = _presets_file()
    try:
        data = json.loads(presets_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    if name not in data:
        return False
    del data[name]
    presets_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return True


def all_presets() -> dict[str, EnhanceSettings]:
    """Built-in presets plus user presets (user names can't shadow built-ins)."""
    return {**load_user_presets(), **PRESETS}
