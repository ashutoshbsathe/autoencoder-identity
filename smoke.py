"""Confirm JAX sees CUDA and ops run on the GPU."""

import jax
import jax.numpy as jnp

print(f'jax {jax.__version__}')
print(f'default backend: {jax.default_backend()}')
print(f'devices: {jax.devices()}')

x = jnp.ones((1024, 1024))
y = (x @ x).block_until_ready()
print(f'matmul landed on: {y.devices()}')

assert jax.default_backend() == 'gpu', 'CUDA not active: JAX fell back to CPU'
print('CUDA + JAX works!')
