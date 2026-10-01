"""Optional terminal target-distance reward for a future VLN pilot."""

from __future__ import annotations

import math


def bounded_terminal_progress(start_m: float, final_m: float, weight: float) -> tuple[float, bool]:
    """Return a once-only bonus and whether a distance was invalid.

    Weight is restricted to [0, 1], keeping a failure's progress bonus below
    the current success floor of 2. A zero weight disables the feature.
    """
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError(f"progress weight must be finite and in [0, 1], got {weight}")
    if weight == 0:
        return 0.0, False
    if not all(math.isfinite(d) and d >= 0 for d in (start_m, final_m)):
        return 0.0, True
    progress = (start_m - final_m) / max(start_m, 3.0)
    return weight * max(-1.0, min(1.0, progress)), False
