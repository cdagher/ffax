from typing import List, Dict, Tuple, Callable, Any, Optional
from jaxtyping import Array, Float, PyTree

import numpy as np

import jax
from jax import numpy as jnp
from jax import random as jr

import equinox as eqx

import optax

import random

from .module import Module


TRAIN_EPSILON = 1e-5


class Network:

    _layers: List[Module]               # The layers of the network
    _g_scores: Dict[int, float]      # The goodness scores of the network's layers
    _opt: optax.GradientTransformation  # The optimizer used to train the network

    _trainable_layers: List[int]        # The indices of the layers that are trainable

    def __init__(
            self,
            layers: List[Module],
            opt: optax.GradientTransformation
        ):
        """
        Args:
            layers: The layers of the network.
            opt: The optimizer used to train the network.
        """

        self._layers = layers
        self._g_scores = {
            i: 0.0 for i, _ in enumerate(layers)
        }
        self._opt = opt

        self._trainable_layers = list(range(len(layers)))

    @property
    def layers(self) -> List[Module]:
        """
        Returns:
            The layers of the network.
        """
        return self._layers

    @property
    def trainable_layers(self) -> List[int]:
        """
        Returns:
            The indices of the layers that are trainable.
        """
        return self._trainable_layers

    @property
    def g_scores(self) -> Dict[int, float]:
        """
        Returns:
            The goodness scores of the network's layers.
        """
        return self._g_scores

    @property
    def goodness(self) -> float:
        """
        Returns:
            The "global" goodness score of the network.
        """
        return np.sum(np.array(list(self._g_scores.values())))
    
    def __str__(self):
        s = "Network(\n"
        s += f"goodness: {self.goodness},\n"
        s += f"trainable layers: {self._trainable_layers},\n"
        s += f"optimizer: {self._opt},\n"
        for i, layer in enumerate(self._layers):
            s += f"Layer {i}: {layer},\n"
        s += ")"
        return s


    @eqx.filter_jit
    def forward(self, x: Array) -> Array:
        """
        Perform a forward pass on self.

        Args:
            x: The input data

        Returns:        
            The output data.
        """
        for layer in self._layers:
            self._g_scores[layer] = layer.goodness(x)
            x = layer.forward(x)
        
        return x

    def train_layer(
            self,
            i: int,
            x: Array,
            positive: bool
        ) -> Tuple[float, Array]:
        """
        Train the i-th layer of the network.

        Args:
            i: The index of the layer to be trained.
            x: The input data.
            positive: Whether `x` is a positive or negative example.

        Returns:
            The goodness score and the output of the layer.
        """
        # Get the layer to be trained
        layer = self._layers[i]
        
        # Perform a forward pass, get the goodness and the output
        out, layer = layer.train_step(x, positive, self._opt)
        p_g, g, y = out

        self._layers[i] = layer
        self._g_scores[i] = p_g

        # If the layer fully trained, remove it from the list of trainable layers
        if positive and np.abs(layer.theta - p_g) < TRAIN_EPSILON:
            self._trainable_layers.remove(i)
        elif not positive and np.abs(p_g - layer.theta) < TRAIN_EPSILON:
            self._trainable_layers.remove(i)

        return p_g, g, y
    
    def train_sequential(
            self,
            x: Array,
            positive: bool
        ) -> Tuple[float, Array]:
        """
        Train the network sequentially.
        Args:
            x: The input data.
            positive: Whether `x` is a positive or negative example.
        Returns:
            The goodness score and the output of the network.
        """

        # Train the layers sequentially
        for i in self._trainable_layers:
            while i in self._trainable_layers:
                p_g, g, y = self.train_layer(i, x, positive)
            # normalize the output
            y = y / np.linalg.norm(y)
            x = y

            # Update the goodness score of the layer
            self._g_scores[self._layers[i]] = p_g

        return p_g, y
