"""Print the learned encoder/decoder weights from a run's checkpoint.

Usage:
    uv run python inspect_weights.py [run_dir] [--step N] [--norms]
    uv run python inspect_weights.py [run_dir] --series   # norms over all steps
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


def norm(a):
    return float(np.linalg.norm(np.asarray(a)))


def _det(a):
    a = np.asarray(a)
    if a.shape[0] != a.shape[1]:
        return float('nan')
    return float(np.linalg.det(a))


def show_norms(params, cfg):
    groups = (('encoder', params.encoder), ('decoder', params.decoder))
    for name, layers in groups:
        for i, layer in enumerate(layers):
            wn, bn = norm(layer.w), norm(layer.b)
            print(f'{name}[{i}]  ||w|| {wn:.4f}   ||b|| {bn:.4f}')
    if cfg.model.kind == 'ae' and len(params.encoder) == 1:
        we = np.asarray(params.encoder[0].w)
        wd = np.asarray(params.decoder[0].w)
        be = np.asarray(params.encoder[0].b)
        bd = np.asarray(params.decoder[0].b)
        comp = we @ wd
        gap = norm(comp - np.eye(len(comp)))
        print(f'effective bias  ||b_e@W_d + b_d||  {norm(be @ wd + bd):.4f}')
        print(f'composition     ||W_e@W_d||        {norm(comp):.4f}')
        print(f'identity gap    ||W_e@W_d - I||    {gap:.4f}')
        if we.shape[0] == we.shape[1]:
            de, dd = _det(we), _det(wd)
            print(f'det(W_e)        {de:+.4f}')
            print(f'det(W_d)        {dd:+.4f}    (product {de * dd:+.4f})')


def _total(arrs):
    return float(np.sqrt(sum(norm(a) ** 2 for a in arrs)))


def show_series(traj, cfg):
    single = cfg.model.kind == 'ae' and len(traj[0][1].encoder) == 1
    if single:
        cols = [
            'step',
            '||We||',
            '||be||',
            '||Wd||',
            '||bd||',
            'det(We)',
            'det(Wd)',
            '||eff_b||',
            '||WeWd-I||',
        ]
    else:
        cols = ['step', '||W||', '||b||']
    print(''.join(f'{c:>10}' for c in cols))
    for step, p in traj:
        if single:
            we = np.asarray(p.encoder[0].w)
            wd = np.asarray(p.decoder[0].w)
            be = np.asarray(p.encoder[0].b)
            bd = np.asarray(p.decoder[0].b)
            comp = we @ wd
            vals = [
                norm(we),
                norm(be),
                norm(wd),
                norm(bd),
                _det(we),
                _det(wd),
                norm(be @ wd + bd),
                norm(comp - np.eye(len(comp))),
            ]
        else:
            layers = p.encoder + p.decoder
            vals = [
                _total([layer.w for layer in layers]),
                _total([layer.b for layer in layers]),
            ]
        print(f'{step:>10}' + ''.join(f'{v:>10.4f}' for v in vals))


def main():
    args = sys.argv[1:]
    series = '--series' in args
    norms = '--norms' in args
    args = [a for a in args if a not in ('--series', '--norms')]
    want_step = None
    if '--step' in args:
        i = args.index('--step')
        want_step = int(args[i + 1])
        args = args[:i] + args[i + 2 :]
    run = args[0] if args else latest_run('runs')
    cfg = config.load(os.path.join(run, 'config.yaml'))
    template = model.init_params(jax.random.key(0), cfg.model)
    ckpt_dir = os.path.join(run, 'checkpoints')

    if series:
        traj = checkpoint.load_trajectory(ckpt_dir, template)
        print(f'run {run}  ({len(traj)} checkpoints)')
        show_series(traj, cfg)
        return

    step, params = checkpoint.load(ckpt_dir, template, want_step)
    print(f'run {run}  (checkpoint step {step})')
    if norms:
        show_norms(params, cfg)
        return
    show('encoder', params.encoder)
    show('decoder', params.decoder)
    if cfg.model.kind == 'ae' and len(params.encoder) == 1:
        we = np.asarray(params.encoder[0].w)
        wd = np.asarray(params.decoder[0].w)
        print(f'\nencoder.w @ decoder.w  (≈ I if identity learned)\n{we @ wd}')


if __name__ == '__main__':
    main()
