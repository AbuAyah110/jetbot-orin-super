from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_robot_config(path: str | Path | None = None) -> Dict[str, Any]:
    if path is None:
        # repo_root/config/robot.yaml relative to this file: src/jetbot_control/config_loader.py
        repo_root = Path(__file__).resolve().parents[2]
        path = repo_root / 'config' / 'robot.yaml'
    path = Path(path)
    with path.open('r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError('robot config must be a mapping')
    return data
