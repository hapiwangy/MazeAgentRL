<<<<<<< HEAD
SPARSE_REWARD_VALUES = {
    "key": 20.0,
    "exit": 50.0,
}

DENSE_REWARD_CONFIG = {
    "step_penalty": -0.1,
    "revisit_penalty": -0.05,
    "progress_scale": 0.25,
    "pre_key_key_weight": 1.0,
    "pre_key_exit_weight": 0.0,
    "post_key_key_weight": 0.0,
    "post_key_exit_weight": 1.0,
}

LLM_REWARD_RANGE_CONFIG = {
    "step_min": -0.05,
    "step_max": 0.5,
    "total_budget_ratio": 0.49,

    # LLM-only mode tends to under-train because its shaping signal is tiny
    # compared to sparse/dense milestone rewards. These values are applied
    # ONLY when reward_mode == "llm".
    "llm_only_scale": 5.0,
    "llm_only_budget_scale": 5.0,
    "llm_only_key_bonus": 5.0,
    "llm_only_exit_bonus": 20.0,
    "llm_only_exit_without_key_penalty": -2.0,
}

REWARD_MODE_COMPONENTS = {
    "sparse": ("sparse",),
    "dense": ("dense",),
    "llm": ("llm",),
    "sparse_dense": ("sparse", "dense"),
    "dense_llm": ("dense", "llm"),
    "sparse_llm": ("sparse", "llm"),
    "sparse_dense_llm": ("sparse", "dense", "llm"),
}

DEFAULT_REWARD_MODE = "sparse_dense_llm"


def get_reward_mode_choices():
    """Return supported reward-mode names for CLI validation."""
    return tuple(REWARD_MODE_COMPONENTS.keys())


def build_reward_components(sparse_reward, dense_reward, llm_reward):
    """Create a named reward-component mapping for the current transition."""
    return {
        "sparse": float(sparse_reward),
        "dense": float(dense_reward),
        "llm": float(llm_reward),
    }


def combine_rewards(reward_mode, reward_components):
    """Combine reward components according to the selected reward mode."""
    component_names = REWARD_MODE_COMPONENTS[reward_mode]
    return sum(reward_components[name] for name in component_names)


def reward_mode_uses_llm(reward_mode):
    """Return whether the selected reward mode depends on the LLM component."""
    return "llm" in REWARD_MODE_COMPONENTS[reward_mode]
=======
SPARSE_REWARD_VALUES = {
    "key": 20.0,
    "exit": 50.0,
}

DENSE_REWARD_CONFIG = {
    "step_penalty": -0.1,
    "revisit_penalty": -0.05,
    "progress_scale": 0.25,
    "pre_key_key_weight": 1.0,
    "pre_key_exit_weight": 0.0,
    "post_key_key_weight": 0.0,
    "post_key_exit_weight": 1.0,
}

LLM_REWARD_RANGE_CONFIG = {
    "step_min": -0.05,
    "step_max": 0.5,
    "total_budget_ratio": 0.49,

    # LLM-only mode tends to under-train because its shaping signal is tiny
    # compared to sparse/dense milestone rewards. These values are applied
    # ONLY when reward_mode == "llm".
    "llm_only_scale": 5.0,
    "llm_only_budget_scale": 5.0,
    "llm_only_key_bonus": 5.0,
    "llm_only_exit_bonus": 20.0,
    "llm_only_exit_without_key_penalty": -2.0,
}

REWARD_MODE_COMPONENTS = {
    "sparse": ("sparse",),
    "dense": ("dense",),
    "llm": ("llm",),
    "sparse_dense": ("sparse", "dense"),
    "dense_llm": ("dense", "llm"),
    "sparse_llm": ("sparse", "llm"),
    "sparse_dense_llm": ("sparse", "dense", "llm"),
}

DEFAULT_REWARD_MODE = "sparse_dense_llm"


def get_reward_mode_choices():
    """Return supported reward-mode names for CLI validation."""
    return tuple(REWARD_MODE_COMPONENTS.keys())


def build_reward_components(sparse_reward, dense_reward, llm_reward):
    """Create a named reward-component mapping for the current transition."""
    return {
        "sparse": float(sparse_reward),
        "dense": float(dense_reward),
        "llm": float(llm_reward),
    }


def combine_rewards(reward_mode, reward_components):
    """Combine reward components according to the selected reward mode."""
    component_names = REWARD_MODE_COMPONENTS[reward_mode]
    return sum(reward_components[name] for name in component_names)


def reward_mode_uses_llm(reward_mode):
    """Return whether the selected reward mode depends on the LLM component."""
    return "llm" in REWARD_MODE_COMPONENTS[reward_mode]
>>>>>>> 782edc09766074deaca156230cc233e6bcb4b88a
