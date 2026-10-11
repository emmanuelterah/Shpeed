"""Domain error hierarchy (P0-15).

Code catches these specific errors rather than using a bare ``except``.
"""


class ShpeedError(Exception):
    """Base class for every error Shpeed raises on purpose."""


class DataError(ShpeedError):
    """Source data is missing, malformed or could not be fetched."""


class ValidationError(DataError):
    """Data failed a schema or plausibility check, so its transaction was rolled back."""


class StoreBusyError(ShpeedError):
    """Another process holds the data store's single-writer lock."""


class LeakageError(ShpeedError):
    """A read went past the prediction cutoff. This is always a bug."""


class InvariantError(ShpeedError):
    """A domain invariant was violated, for example probabilities that do not sum to 1."""


class BudgetExceededError(ShpeedError):
    """A run exceeded its time budget."""
