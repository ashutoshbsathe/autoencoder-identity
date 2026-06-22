"""Train the autoencoder, checkpointing weights at every eval.

Examples:
    uv run python train.py
    uv run python train.py --model.activation relu --model.d-mid 64
    uv run python train.py --config configs/default.yaml --train.lr 5e-3
"""

import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

import checkpoint
import config
import data
import model

ACTIVATIONS = {
    'identity': lambda x: x,
    'relu': jax.nn.relu,
    'leaky_relu': jax.nn.leaky_relu,
}
OPTIMIZERS = {'adam': optax.adam, 'sgd': optax.sgd}


def _recon(x, x_hat, kind):
    if kind == 'gaussian_nll':
        return 0.5 * jnp.mean(jnp.sum((x - x_hat) ** 2, axis=-1))
    return jnp.mean((x - x_hat) ** 2)


def _kl(mu, logvar):
    per = 0.5 * jnp.sum(jnp.exp(logvar) + mu**2 - 1 - logvar, axis=-1)
    return jnp.mean(per)


def make_loss(mcfg, activation):
    def loss_fn(params, x, key):
        if mcfg.kind == 'vae':
            x_hat, mu, logvar = model.vae_forward(params, x, key, activation)
            recon = _recon(x, x_hat, mcfg.recon)
            kl = _kl(mu, logvar)
            return recon + mcfg.beta * kl, {'recon': recon, 'kl': kl}
        x_hat = model.reconstruct(params, x, activation)
        loss = jnp.mean((x - x_hat) ** 2)
        return loss, {'recon': loss, 'kl': jnp.zeros(())}

    return loss_fn


def train(cfg: config.Config) -> model.Params:
    train_x, val_x, _ = data.prepare(cfg.data)
    val_batch = jnp.asarray(val_x)

    activation = ACTIVATIONS[cfg.model.activation]
    loss_fn = make_loss(cfg.model, activation)
    init_key, train_key = jax.random.split(jax.random.key(cfg.train.seed))
    params = model.init_params(init_key, cfg.model)
    optimizer = OPTIMIZERS[cfg.train.optimizer](cfg.train.lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, batch, key):
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, _), grads = grad_fn(params, batch, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    run_path = os.path.join(cfg.train.run_dir, time.strftime('%Y%m%d-%H%M%S'))
    os.makedirs(run_path, exist_ok=True)
    config.dump(cfg, os.path.join(run_path, 'config.yaml'))
    manager = checkpoint.make_manager(os.path.join(run_path, 'checkpoints'))
    metrics_path = os.path.join(run_path, 'metrics.jsonl')

    def record(at_step, params, train_loss, key):
        loss, aux = loss_fn(params, val_batch, key)
        vl, rc, kl = float(loss), float(aux['recon']), float(aux['kl'])
        metrics = {
            'train_loss': train_loss,
            'val_loss': vl,
            'val_recon': rc,
            'val_kl': kl,
        }
        checkpoint.save(manager, metrics_path, at_step, params, metrics)
        print(
            f'step {at_step:>6}  train {train_loss:.4f}  '
            f'val {vl:.4f}  recon {rc:.4f}  kl {kl:.4f}'
        )

    rng = np.random.default_rng(cfg.train.seed)
    batch_size = min(cfg.train.batch_size, len(train_x))
    for i in range(cfg.train.steps):
        idx = rng.choice(len(train_x), batch_size, replace=False)
        batch = jnp.asarray(train_x[idx])
        train_key, step_key = jax.random.split(train_key)
        params, opt_state, loss = step(params, opt_state, batch, step_key)
        if i % cfg.train.eval_every == 0:
            train_key, eval_key = jax.random.split(train_key)
            record(i, params, float(loss), eval_key)

    train_key, eval_key = jax.random.split(train_key)
    record(cfg.train.steps, params, float(loss), eval_key)
    manager.wait_until_finished()
    print(f'done -> {run_path}')
    return params


def main():
    argv = sys.argv[1:]
    base = config.Config()
    if '--config' in argv:
        i = argv.index('--config')
        base = config.load(argv[i + 1])
        argv = argv[:i] + argv[i + 2 :]
    train(tyro.cli(config.Config, default=base, args=argv))


if __name__ == '__main__':
    main()
