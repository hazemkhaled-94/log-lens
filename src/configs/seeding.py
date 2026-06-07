"""Global random-seed control for reproducible runs."""

from configs.settings import RANDOM_STATE


def seed_everything(seed: int | None = None) -> int:
    """Seed Python, NumPy, and Torch RNGs for reproducibility.

    Args:
        seed: Seed to apply; falls back to ``RANDOM_STATE`` from the env.

    Returns:
        The seed that was applied.
    """
    resolved = RANDOM_STATE if seed is None else seed
    # Lazy import: transformers.set_seed seeds random, numpy and torch.
    from transformers import set_seed

    set_seed(resolved)
    return resolved
