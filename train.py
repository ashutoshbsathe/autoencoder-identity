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


def mse(params, x, activation):
    x_hat = model.reconstruct(params, x, activation)
    return jnp.mean((x - x_hat) ** 2)


def train(cfg: config.Config) -> model.Params:
    x = data.sample(cfg.data)
    train_x, val_x, _ = data.split(x, cfg.data.splits, cfg.data.seed)
    val_batch = jnp.asarray(val_x)

    activation = ACTIVATIONS[cfg.model.activation]
    params = model.init_params(jax.random.key(cfg.train.seed), cfg.model)
    optimizer = OPTIMIZERS[cfg.train.optimizer](cfg.train.lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, batch):
        loss, grads = jax.value_and_grad(mse)(params, batch, activation)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    run_path = os.path.join(cfg.train.run_dir, time.strftime('%Y%m%d-%H%M%S'))
    os.makedirs(run_path, exist_ok=True)
    config.dump(cfg, os.path.join(run_path, 'config.yaml'))
    manager = checkpoint.make_manager(os.path.join(run_path, 'checkpoints'))
    metrics_path = os.path.join(run_path, 'metrics.jsonl')

    def record(at_step, params, train_loss):
        val_loss = float(mse(params, val_batch, activation))
        metrics = {'train_loss': train_loss, 'val_loss': val_loss}
        checkpoint.save(manager, metrics_path, at_step, params, metrics)
        print(f'step {at_step:>6}  train {train_loss:.4f}  val {val_loss:.4f}')

    rng = np.random.default_rng(cfg.train.seed)
    batch_size = min(cfg.train.batch_size, len(train_x))
    for i in range(cfg.train.steps):
        idx = rng.choice(len(train_x), batch_size, replace=False)
        batch = jnp.asarray(train_x[idx])
        params, opt_state, loss = step(params, opt_state, batch)
        if i % cfg.train.eval_every == 0:
            record(i, params, float(loss))

    record(cfg.train.steps, params, float(loss))
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
