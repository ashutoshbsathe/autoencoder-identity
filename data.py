"""Synthetic data distributions and train/val/test splitting."""

import numpy as np

from config import DataConfig


def gaussian(cfg: DataConfig) -> np.ndarray:
    """Samples N(mean, std**2 I) of shape (n_samples, d_in)."""
    rng = np.random.default_rng(cfg.seed)
    x = cfg.mean + cfg.std * rng.standard_normal((cfg.n_samples, cfg.d_in))
    return x.astype(np.float32)


def sample(cfg: DataConfig) -> np.ndarray:
    """Dispatches to the configured distribution."""
    if cfg.dist == 'gaussian':
        return gaussian(cfg)
    raise NotImplementedError(f'distribution {cfg.dist!r} not implemented')


def split(
    x: np.ndarray, splits: tuple[float, float, float], seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shuffles and partitions x into train/val/test by fraction."""
    rng = np.random.default_rng(seed)
    x = x[rng.permutation(len(x))]
    n_train = int(splits[0] * len(x))
    n_val = int(splits[1] * len(x))
    return tuple(np.split(x, [n_train, n_train + n_val]))
