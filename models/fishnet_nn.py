#@title network stuff in Jax  <font color='lightblue'>[run me]</font>
from tqdm import tqdm
import math

from typing import Any, Callable, Sequence, Optional
Array = Any

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import jraph



def fill_triangular(x):
    m = x.shape[0] # should be n * (n+1) / 2
    # solve for n
    n = int(math.sqrt((0.25 + 2 * m)) - 0.5)
    idx = int(m - (n**2 - m))
    x_tail = x[idx:]

    return jnp.concatenate([x_tail, jnp.flip(x, [0])], 0).reshape(n, n)


def fill_diagonal(a, val):
  assert a.ndim >= 2
  i, j = jnp.diag_indices(min(a.shape[-2:]))
  return a.at[..., i, j].set(val)


def construct_fisher_matrix_single(outputs):
    Q = fill_triangular(outputs)
    middle = jnp.diag(jnp.triu(Q) - nn.softplus(jnp.triu(Q)))
    padding = jnp.zeros(Q.shape)
    L = Q - fill_diagonal(padding, middle)

    return jnp.einsum('...ij,...jk->...ik', L, jnp.transpose(L, (1, 0)))




# custom aggregation function for padded inputs
def fishnets_aggregation(
                 n_p: int,
                 data: jnp.ndarray,
                 segment_ids: jnp.ndarray,
                 num_segments: Optional[int] = None,
                 indices_are_sorted: bool = False,
                 unique_indices: bool = False):
  """Returns mean for each segment.
  Args:
    n_p: size of weighted bottleneck
    n_data: the number of data we want to take the mean of
    data: the values which are averaged segment-wise.
    segment_ids: indices for the segments.
    num_segments: total number of segments.
    indices_are_sorted: whether ``segment_ids`` is known to be sorted.
    unique_indices: whether ``segment_ids`` is known to be free of duplicates.
  """
  #print("data", data.shape)
  score = data[..., :n_p]
  fisher = data[..., n_p:]
    
  score = jraph.segment_sum(
      score,
      segment_ids,
      num_segments,
      indices_are_sorted=indices_are_sorted,
      unique_indices=unique_indices)

  # construct fisher matrix
  # should construct matrix before doing sum but let's see how this works
  fisher = jraph.segment_sum(
      fisher,
      segment_ids,
      num_segments,
      indices_are_sorted=indices_are_sorted,
      unique_indices=unique_indices)

  fisher = jax.vmap(construct_fisher_matrix_single)(fisher)
  fisher += jnp.eye(n_p) # add prior
  mle = jnp.einsum('...jk,...k->...j', jnp.linalg.inv(fisher), score)

  return mle

