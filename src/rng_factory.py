"""
Get seeds from a master seed, for reproducibility, using numpy SeedSequence. Cleaner way to ensure no adjacent seed dependence, required for rliable style multi-seed reproducability.
"""

import numpy as np


class SeededRNG:
    """
    RNG class that creates a seed sequence once, and use next_rng or next_seed to advance the seedsequence ensuring seed independence.
    """

    def __init__(self, master_seed: int):
        self.master_seed = master_seed
        self.seed_sequence = np.random.SeedSequence(master_seed)

    def next_rng(self) -> np.random.Generator:
        """
        Return a SeedSequence seeded numpy random number generator.
        Every call to next_rng is guaranteed to return a different seeded (non-adjacent) RNG
        """
        new_seed = self.seed_sequence.spawn(1)[0]
        return np.random.default_rng(new_seed)

    def next_seed(self) -> int:
        """
        Return a brand new seed, guaranteed to be non-adjacent to any seeds called so far.
        """
        return int(self.seed_sequence.spawn(1)[0].generate_state(1)[0])


if __name__ == "__main__":
    rng = SeededRNG(42)
    for _ in range(10):
        print(rng.next_seed())
