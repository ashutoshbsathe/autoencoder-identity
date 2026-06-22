"""Synthetic data distributions and train/val/test splitting."""

import numpy as np

from config import DataConfig


def _orthonormal(rng, n: int, k: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((n, k)))
    return q


def _embed(manifold: np.ndarray, cfg: DataConfig, rng) -> np.ndarray:
    m = manifold.shape[1]
    if cfg.d_in < m:
        raise ValueError(f'{cfg.dist} needs d_in >= {m}')
    x = manifold @ _orthonormal(rng, cfg.d_in, m).T
    x = x + cfg.noise_std * rng.standard_normal(x.shape)
    return x.astype(np.float32)


def gaussian(cfg: DataConfig) -> np.ndarray:
    """Isotropic N(mean, std**2 I)."""
    rng = np.random.default_rng(cfg.seed)
    x = cfg.mean + cfg.std * rng.standard_normal((cfg.n_samples, cfg.d_in))
    return x.astype(np.float32)


def low_rank_gaussian(cfg: DataConfig) -> np.ndarray:
    """Rank-intrinsic_dim Gaussian signal plus isotropic noise."""
    rng = np.random.default_rng(cfg.seed)
    basis = _orthonormal(rng, cfg.d_in, cfg.intrinsic_dim)
    z = rng.standard_normal((cfg.n_samples, cfg.intrinsic_dim))
    noise = cfg.noise_std * rng.standard_normal((cfg.n_samples, cfg.d_in))
    return (z @ basis.T + noise).astype(np.float32)


def anisotropic_gaussian(cfg: DataConfig) -> np.ndarray:
    """Full-rank Gaussian with a geometrically decaying eigenspectrum."""
    rng = np.random.default_rng(cfg.seed)
    eigvals = cfg.decay ** np.arange(cfg.d_in)
    q = _orthonormal(rng, cfg.d_in, cfg.d_in)
    z = rng.standard_normal((cfg.n_samples, cfg.d_in))
    return ((z * np.sqrt(eigvals)) @ q.T).astype(np.float32)


def uniform(cfg: DataConfig) -> np.ndarray:
    """Uniform with variance std**2, centered at mean."""
    rng = np.random.default_rng(cfg.seed)
    half = cfg.std * np.sqrt(3.0)
    shape = (cfg.n_samples, cfg.d_in)
    x = rng.uniform(cfg.mean - half, cfg.mean + half, shape)
    return x.astype(np.float32)


def laplace(cfg: DataConfig) -> np.ndarray:
    """Laplace with variance std**2, centered at mean."""
    rng = np.random.default_rng(cfg.seed)
    b = cfg.std / np.sqrt(2.0)
    x = rng.laplace(cfg.mean, b, (cfg.n_samples, cfg.d_in))
    return x.astype(np.float32)


def mixture(cfg: DataConfig) -> np.ndarray:
    """Mixture of n_components isotropic Gaussian blobs."""
    rng = np.random.default_rng(cfg.seed)
    means = cfg.std * rng.standard_normal((cfg.n_components, cfg.d_in))
    comp = rng.integers(0, cfg.n_components, cfg.n_samples)
    noise = cfg.std * rng.standard_normal((cfg.n_samples, cfg.d_in))
    return (means[comp] + noise).astype(np.float32)


def swiss_roll(cfg: DataConfig) -> np.ndarray:
    """2-D swiss-roll manifold embedded in d_in dims with noise."""
    rng = np.random.default_rng(cfg.seed)
    t = 1.5 * np.pi * (1 + 2 * rng.random(cfg.n_samples))
    h = 21 * rng.random(cfg.n_samples)
    roll = np.stack([t * np.cos(t), h, t * np.sin(t)], axis=1)
    return _embed(roll, cfg, rng)


def s_curve(cfg: DataConfig) -> np.ndarray:
    """2-D s-curve manifold embedded in d_in dims with noise."""
    rng = np.random.default_rng(cfg.seed)
    t = 3 * np.pi * (rng.random(cfg.n_samples) - 0.5)
    h = 2 * rng.random(cfg.n_samples)
    curve = np.stack([np.sin(t), h, np.sign(t) * (np.cos(t) - 1)], axis=1)
    return _embed(curve, cfg, rng)


_SAMPLERS = {
    'gaussian': gaussian,
    'low_rank_gaussian': low_rank_gaussian,
    'anisotropic_gaussian': anisotropic_gaussian,
    'uniform': uniform,
    'laplace': laplace,
    'mixture': mixture,
    'swiss_roll': swiss_roll,
    's_curve': s_curve,
}


def sample(cfg: DataConfig) -> np.ndarray:
    """Dispatches to the configured distribution."""
    return _SAMPLERS[cfg.dist](cfg)


def split(
    x: np.ndarray, splits: tuple[float, float, float], seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shuffles and partitions x into train/val/test by fraction."""
    rng = np.random.default_rng(seed)
    x = x[rng.permutation(len(x))]
    n_train = int(splits[0] * len(x))
    n_val = int(splits[1] * len(x))
    return tuple(np.split(x, [n_train, n_train + n_val]))


def standardize(
    train: np.ndarray, val: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardizes every split using train-split statistics."""
    mean = train.mean(axis=0)
    std = train.std(axis=0) + 1e-8

    def scale(s):
        return ((s - mean) / std).astype(np.float32)

    return scale(train), scale(val), scale(test)


def prepare(
    cfg: DataConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Samples, splits, and optionally standardizes (on train stats)."""
    train, val, test = split(sample(cfg), cfg.splits, cfg.seed)
    if cfg.standardize:
        train, val, test = standardize(train, val, test)
    return train, val, test
