"""Orbax checkpointing of weight snapshots, plus a metrics log."""

import json
import os

import orbax.checkpoint as ocp


def make_manager(checkpoint_dir: str) -> ocp.CheckpointManager:
    """A manager that keeps every snapshot, for trajectory analysis."""
    options = ocp.CheckpointManagerOptions(max_to_keep=None)
    path = os.path.abspath(checkpoint_dir)
    return ocp.CheckpointManager(path, options=options)


def save(manager, metrics_path, step, params, metrics):
    manager.save(step, args=ocp.args.StandardSave(params))
    with open(metrics_path, 'a') as f:
        f.write(json.dumps({'step': step, **metrics}) + '\n')


def load_trajectory(checkpoint_dir, template):
    """Restores every snapshot into `template`'s structure, in step order."""
    manager = make_manager(checkpoint_dir)
    return [
        (step, manager.restore(step, args=ocp.args.StandardRestore(template)))
        for step in manager.all_steps()
    ]
