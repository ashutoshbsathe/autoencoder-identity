"""Print the learned encoder/decoder weights from a run's checkpoint.

Usage:
    uv run python inspect_weights.py [run_dir] [--step N]
"""

import glob
import os
import sys

import jax
import numpy as np

import checkpoint
import config
import model

np.set_printoptions(precision=3, suppress=True)


def latest_run(run_dir):
    pattern = os.path.join(run_dir, '**', 'config.yaml')
    runs = glob.glob(pattern, recursive=True)
    if not runs:
        raise FileNotFoundError(f'no runs under {run_dir!r}')
    return os.path.dirname(max(runs, key=os.path.getmtime))


def show(name, layers):
    for i, layer in enumerate(layers):
        w, b = np.asarray(layer.w), np.asarray(layer.b)
        print(f'\n{name}[{i}].w  {w.shape}\n{w}')
        print(f'{name}[{i}].b  {b.shape}\n{b}')


def main():
    args = sys.argv[1:]
    want_step = None
    if '--step' in args:
        i = args.index('--step')
        want_step = int(args[i + 1])
        args = args[:i] + args[i + 2 :]
    run = args[0] if args else latest_run('runs')
    cfg = config.load(os.path.join(run, 'config.yaml'))
    template = model.init_params(jax.random.key(0), cfg.model)
    ckpt_dir = os.path.join(run, 'checkpoints')
    step, params = checkpoint.load(ckpt_dir, template, want_step)

    print(f'run {run}  (checkpoint step {step})')
    show('encoder', params.encoder)
    show('decoder', params.decoder)

    if cfg.model.kind == 'ae' and len(params.encoder) == 1:
        we = np.asarray(params.encoder[0].w)
        wd = np.asarray(params.decoder[0].w)
        print(f'\nencoder.w @ decoder.w  (≈ I if identity learned)\n{we @ wd}')


if __name__ == '__main__':
    main()
