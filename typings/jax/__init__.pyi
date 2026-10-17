"""Type stubs for the parts of JAX that Shpeed uses (``f1sim.adapters.models.pace_hyper``).

JAX ships only partial type information, so strict pyright reports its members as
``Unknown``. Everything Shpeed touches from JAX is a JAX array or a JAX transform, typed
``Any`` here, matching the NumPyro stubs. Extend these stubs when the pace model uses more
of JAX.
"""

from typing import Any

def __getattr__(name: str) -> Any: ...
