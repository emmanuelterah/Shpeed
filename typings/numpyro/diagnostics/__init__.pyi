"""NumPyro MCMC diagnostics."""

from typing import Any

def summary(
    samples: Any, prob: float = 0.9, group_by_chain: bool = True
) -> dict[str, dict[str, Any]]: ...
