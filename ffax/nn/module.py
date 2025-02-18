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
            theta: float = 1.0
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
        
        self._activation = activation or (lambda x: x) # default to identity (linear activation)
        self._goodness_fn = goodness_fn or (lambda x: jnp.mean(jnp.square(x))) # default to mean squared activation

        self._opt_state = None

    @property
    def theta(self) -> float:
        return self._theta

    @eqx.filter_jit
    def forward(self, x: Array) -> Array:
        return self._activation(self._layer(x))

    # @eqx.filter_jit
    def goodness(self, x: Array) -> float:
        return self._goodness_fn(self._activation(self._layer(x)))

    # Forward pass
    @eqx.filter_jit
    def _forward_pass(self, layer: eqx.Module, x: Array):
        """
        Perform a forward pass on self.

        Args:
            layer: The Equinox module that is held in self (used for gradient calculation)
            x: The input data

        Returns:        
            The goodness score and the output data.
        """
        x = layer(x)
        x = self._activation(x)
        
        g = self._goodness_fn(x)

        # probability of goodness
        p_g = jax.nn.sigmoid(g - self._theta)

        return p_g, (g, x)

    @eqx.filter_jit
    def _updates(
            self,
            x: Array,
            opt: optax.GradientTransformation,
            opt_state: PyTree
        ) -> Tuple[Tuple[float, float], PyTree]:

        # Perform a forward pass, get the goodness and the output, as well as the gradient
        (out), grad = eqx.filter_value_and_grad(self._forward_pass, has_aux=True)(self._layer, x)
        p_g, (g, y) = out

        updates, opt_state = opt.update(
            grad, opt_state, eqx.filter(self._layer, eqx.is_array)
        )

        return p_g, (g, y), updates, opt_state

    def train_step(
            self,
            x: Array,
            positive: bool,
            opt: optax.GradientTransformation
            ) -> None:
        """
        Perform a forward pass on the given data `x` and update the weights of the layer based on the goodness of the output.

        Args:
            x: The input data.
            positive: Whether the data `x` is a positive sample.
            opt: The optimizer to use

        Returns:
            Tuple[g, y]: The goodness and output of the Module
            self: The updated Module
        """
        if self._opt_state is None:
            self = eqx.tree_at(lambda m: m._opt_state, self, opt.init(eqx.filter(self._layer, eqx.is_array)))

        p_g, out, updates, opt_state = self._updates(x, opt, self._opt_state)
        g, y = out

        # we want to increase goodness in a positive pass,
        # but SGD methods will try to minimize it. Simple
        # solution is to flip the updates
        if positive:
            updates = jax.tree_util.tree_map(lambda u: -u if isinstance(u, jnp.ndarray) else u, updates)

        self = eqx.tree_at(lambda m: m._opt_state, self, opt_state)

        self = eqx.tree_at(lambda m: m._layer, self, eqx.apply_updates(self._layer, updates))

        return (p_g, g, y), self
