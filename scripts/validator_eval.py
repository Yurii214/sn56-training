"""Mirror of the validator's evaluation-set length rule.

The validator evaluates with ``sequence_len = 4096`` and drops (does not truncate) rows longer
than that; when the submitted model's ``max_position_embeddings`` is below ``2 * 4096`` it
first shrinks ``sequence_len`` to ``ceil(max_position_embeddings / 2)``
(G.O.D ``validator/evaluation/common.py``, ``_load_and_update_evaluation_config``). The dev
split used for checkpoint and soup selection follows the same rule so that it only contains
rows the grader can score.
"""
import json
import math
import os
from typing import Optional

VALIDATOR_SEQUENCE_LEN = 4096


def model_max_positions(model_path: str) -> Optional[int]:
    try:
        with open(os.path.join(model_path, "config.json")) as f:
            cfg = json.load(f)
    except Exception:
        return None
    for key in ("max_position_embeddings", "n_positions", "max_sequence_length"):
        v = cfg.get(key)
        if isinstance(v, int) and v > 0:
            return v
    tc = cfg.get("text_config")
    if isinstance(tc, dict):
        v = tc.get("max_position_embeddings")
        if isinstance(v, int) and v > 0:
            return v
    return None


def eval_sequence_len(model_path: Optional[str], base: int = VALIDATOR_SEQUENCE_LEN) -> int:
    max_pos = model_max_positions(model_path) if model_path else None
    if max_pos and max_pos < 2 * base:
        return int(math.ceil(max_pos / 2))
    return int(base)


def dev_sequence_len(model_path: Optional[str], train_max_length: int) -> int:
    """Dev rows longer than this are dropped: the grader's cut, but never above the training
    window (dev is packed/padded with the same window as training)."""
    cut = eval_sequence_len(model_path)
    if train_max_length and int(train_max_length) > 0:
        return min(int(train_max_length), cut)
    return cut
