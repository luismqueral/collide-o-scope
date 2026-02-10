"""
random.py - Seeded randomness for reproducible synthesis

Wraps Python's random module with seed management. Every script that
uses randomness should create an RNG through this module so that
sessions can be reproduced from a seed.

Usage:
    from utils.random import create_rng

    rng = create_rng(seed=12345)       # reproducible
    rng = create_rng(seed=None)        # random each run, but seed is logged

    value = rng.uniform(0.2, 0.4)
    items = rng.sample(my_list, 3)
    choice = rng.choice(my_list)
    value = rng.from_range([0.2, 0.4]) # convenience for config ranges
"""

import random
import time


def create_rng(seed=None):
    """
    Create a seeded random number generator.

    If seed is None, generates one from current time so it's different
    each run but still recordable in the manifest for reproduction.

    Args:
        seed: Integer seed for reproducibility, or None for random

    Returns:
        An RNG instance with a .seed attribute recording the seed used
    """
    if seed is None:
        seed = int(time.time() * 1000) % (2**31)

    rng = random.Random(seed)

    # store the seed so it can be recorded in manifests
    rng.seed_value = seed

    # convenience: resolve a config value that might be a range or a fixed value
    # e.g., config["similarity"] could be [0.2, 0.4] or just 0.3
    def from_range(value):
        """
        Resolve a config value that's either a [min, max] range or a fixed value.

        If it's a list/tuple of two numbers, pick randomly within that range.
        If it's a single number, return it as-is.
        """
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return rng.uniform(value[0], value[1])
        return value

    def int_from_range(value):
        """
        Same as from_range but returns an integer.
        Useful for things like num_videos, duration.
        """
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return rng.randint(value[0], value[1])
        return int(value)

    rng.from_range = from_range
    rng.int_from_range = int_from_range

    return rng
