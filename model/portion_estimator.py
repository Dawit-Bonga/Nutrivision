"""Portion size estimation helpers.

TODO:
- Load a depth estimation model such as MiDaS.
- Infer scale from a reference object heuristic.
- Estimate serving size in grams or volume units.
"""


class PortionEstimator:
    """Estimate portion size from a single image."""

    def estimate(self, image):
        raise NotImplementedError("Implement portion estimation.")
