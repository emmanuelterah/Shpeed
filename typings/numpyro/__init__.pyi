"""Type stubs for the parts of NumPyro that Shpeed uses (``f1sim.adapters.models.pace_hyper``).

NumPyro ships without complete type information. Values that are JAX arrays or NumPyro
objects are typed ``Any``; extend these stubs when the pace model uses more of NumPyro.
"""

from typing import Any

def sample(name: str, fn: Any, obs: Any = None) -> Any: ...
def plate(
    name: str, size: int, subsample_size: int | None = None, dim: int | None = None
) -> Any: ...
def factor(name: str, log_factor: Any) -> None: ...
