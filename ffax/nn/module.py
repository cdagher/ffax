from typing import Tuple, Callable, Any, Optional
from jaxtyping import Array, Float, PyTree

import jax
from jax import numpy as jnp

import equinox as eqx

import optax


class Module(eqx.Module):

    _layer: eqx.Module
    _activation: Callable[[Any], Any]
    _goodness_fn: Callable[[Any], float]
    _opt_state: PyTree

    _theta: float = eqx.field(static=True)

    def __init__(
            self,
            layer: eqx.Module,
            activation: Optional[Callable[[Any], Any]] = None,
            goodness_fn: Optional[Callable[[Any], float]] = None,
            theta: float = 2.0
        ):
        """
        Args:
            layer: The layer to be wrapped.
            activation: The activation function to be applied after the layer. By default, it is the identity function.
            goodness_fn: The goodness function to be applied after the layer. By default, it is the mean squared activation.
        """
        super().__init__()
        
        self._layer = layer
        
        self._theta = theta
        
        self._activation = activation or (lambda x: jax.nn.relu(x)) # default to relu
        self._goodness_fn = goodness_fn or (lambda x: jnp.mean(jnp.square(x), axis=-1)) # default to mean squared activation

        self._opt_state = None

    @property
    def theta(self) -> float:
        return self._theta

    @eqx.filter_jit
    def forward(self, x: Array) -> Array:
        direction = x / (jnp.linalg.norm(x, keepdims=True) + 1e-4)
        return self._activation(jax.vmap(self._layer)(direction))

    # @eqx.filter_jit
    def goodness(self, x: Array) -> float:
        return self._goodness_fn(self.forward(x))

    # Forward pass
    @eqx.filter_jit
    def _loss_fn(
        self,
        layer: eqx.Module,
        x_p: Array,
        x_n: Array
    ):
        """
        Perform a forward pass on self.

        Args:
            layer: The Equinox module that is held in self (used for gradient calculation)
            x_p: The positive input data.
            x_n: The negative input data.

        Returns:        
            The goodness score and the output data.
        """

        dir_p = x_p / (jnp.linalg.norm(x_p) + 1e-4)
        dir_n = x_n / (jnp.linalg.norm(x_n) + 1e-4)

        g_pos = self._goodness_fn(
            self._activation(
                jax.vmap(layer)(dir_p)
            )
        )
        g_neg = self._goodness_fn(
            self._activation(
                jax.vmap(layer)(dir_n)
            )
        )

        # probability of layer goodness
        loss = jnp.log(
            1 + jnp.exp(jnp.array([
                -g_pos + self._theta, # pushes pos samples to values larger than self._theta
                 g_neg - self._theta  # pushes neg samples to values smaller than self._theta
            ])
        ))
        loss = jnp.mean(loss)

        return loss

    @eqx.filter_jit
    def _updates(
            self,
            x_p: Array,
            x_n: Array,
            opt: optax.GradientTransformation,
            opt_state: PyTree
        ) -> Tuple[Tuple[float, float], PyTree]:

        # Perform a forward pass, get the goodness and the output, as well as the gradient
        loss, grad = eqx.filter_value_and_grad(
            self._loss_fn,
            has_aux=False
        )(
            self._layer,
            x_p,
            x_n
        )
        
        updates, opt_state = opt.update(
            grad, opt_state, eqx.filter(self._layer, eqx.is_array)
        )

        g_pos = self.forward(x_p)
        g_neg = self.forward(x_n)

        return loss, (g_pos, g_neg), updates, opt_state

    def train_step(
            self,
            x_p: Array,
            x_n: Array,
            opt: optax.GradientTransformation,
            batch: bool = False,
            batch_index: Optional[int] = None
            ) -> None:
        """
        Perform a forward pass on the given data `x` and update the weights of the layer based on the goodness of the output.

        Args:
            x: The input data.
            positive: Whether the data `x` is a positive sample.
            opt: The optimizer to use
            batch: Whether to treat `x` as a batch of data.
            batch_index: The batch index of `x`.

        Returns:
            Tuple[g, y]: The goodness and output of the Module
            self: The updated Module
        """
        if batch and batch_index is None:
            raise ValueError("batch_index must be specified when batch is True")

        if self._opt_state is None:
            self = eqx.tree_at(lambda m: m._opt_state, self, opt.init(eqx.filter(self._layer, eqx.is_array)))

        # Add batch dimension if necessary. This simplifies handling
        if not batch:
            x_p = jnp.expand_dims(x_p, axis=0)
            x_n = jnp.expand_dims(x_n, axis=0)

        loss, (g_pos, g_neg), updates, opt_state = self._updates(
            x_p,
            x_n,
            opt,
            self._opt_state
        )

        self = eqx.tree_at(lambda m: m._opt_state, self, opt_state)

        self = eqx.tree_at(lambda m: m._layer, self, eqx.apply_updates(self._layer, updates))

        return (loss, (g_pos, g_neg)), self
