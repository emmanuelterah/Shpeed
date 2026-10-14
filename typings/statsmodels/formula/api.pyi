"""Formula-based model constructors; fitted results are typed ``Any``."""

from typing import Any

import pandas as pd

def mixedlm(
    formula: str,
    data: pd.DataFrame,
    re_formula: str | None = None,
    vc_formula: dict[str, str] | None = None,
    subset: Any = None,
    use_sparse: bool = False,
    missing: str = "none",
    *args: Any,
    **kwargs: Any,
) -> Any: ...
def ols(
    formula: str,
    data: pd.DataFrame,
    subset: Any = None,
    drop_cols: Any = None,
    *args: Any,
    **kwargs: Any,
) -> Any: ...
