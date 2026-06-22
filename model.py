"""Autoencoder: an optionally-2-layer MLP encoder/decoder in pure JAX."""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from config import ModelConfig


class Layer(NamedTuple):
    w: jax.Array
    b: jax.Array


class Params(NamedTuple):
    encoder: list[Layer]
    decoder: list[Layer]


def _dims(cfg: ModelConfig) -> tuple[list, list]:
    d_enc = 2 * cfg.d_hidden if cfg.kind == 'vae' else cfg.d_hidden
    if cfg.d_mid is None:
        return [(cfg.d_in, d_enc)], [(cfg.d_hidden, cfg.d_in)]
    return (
        [(cfg.d_in, cfg.d_mid), (cfg.d_mid, d_enc)],
        [(cfg.d_hidden, cfg.d_mid), (cfg.d_mid, cfg.d_in)],
    )


def _init_layer(key: jax.Array, d_in: int, d_out: int) -> Layer:
    w = jax.random.normal(key, (d_in, d_out)) * (d_in**-0.5)
    return Layer(w=w, b=jnp.zeros(d_out))


def init_params(key: jax.Array, cfg: ModelConfig) -> Params:
    enc_dims, dec_dims = _dims(cfg)
    dims = enc_dims + dec_dims
    keys = jax.random.split(key, len(dims))
    layers = [_init_layer(k, i, o) for k, (i, o) in zip(keys, dims)]
    n = len(enc_dims)
    return Params(encoder=layers[:n], decoder=layers[n:])


def _apply(
    layers: list[Layer], x: jax.Array, activation: Callable
) -> jax.Array:
    *hidden, last = layers
    for layer in hidden:
        x = activation(x @ layer.w + layer.b)
    return x @ last.w + last.b


def encode(params: Params, x: jax.Array, activation: Callable) -> jax.Array:
    return _apply(params.encoder, x, activation)


def decode(params: Params, z: jax.Array, activation: Callable) -> jax.Array:
    return _apply(params.decoder, z, activation)


def reconstruct(
    params: Params, x: jax.Array, activation: Callable
) -> jax.Array:
    return decode(params, encode(params, x, activation), activation)


def reparameterize(
    key: jax.Array, mu: jax.Array, logvar: jax.Array
) -> jax.Array:
    return mu + jnp.exp(0.5 * logvar) * jax.random.normal(key, mu.shape)


def vae_forward(
    params: Params, x: jax.Array, key: jax.Array, activation: Callable
) -> tuple[jax.Array, jax.Array, jax.Array]:
    mu, logvar = jnp.split(encode(params, x, activation), 2, axis=-1)
    z = reparameterize(key, mu, logvar)
    return decode(params, z, activation), mu, logvar
