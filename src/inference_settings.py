"""Explicit inference settings shared by the functional check and benchmark."""

from copy import deepcopy


def normalize_attention_config(config):
    """Remove an inactive Qwen2 window value without changing active attention.

    Transformers 4.51.3 warns on sliding_window alone, while Qwen2 attention
    applies the window only when use_sliding_window is also true. Preserve
    genuinely enabled sliding-window configurations.
    """
    changes = {}
    if (getattr(config, "model_type", None) == "qwen2"
            and getattr(config, "use_sliding_window", None) is False
            and getattr(config, "sliding_window", None) is not None):
        changes["sliding_window"] = {"checkpoint": config.sliding_window, "runtime": None,
                                     "reason": "window disabled by use_sliding_window=false"}
        config.sliding_window = None
    return changes


def greedy_config(original, max_new_tokens):
    """Copy model defaults; remove inactive sampling controls for greedy runs."""
    config = deepcopy(original)
    config.do_sample = False
    config.temperature = None
    config.top_p = None
    config.top_k = None
    config.max_new_tokens = max_new_tokens
    return config
