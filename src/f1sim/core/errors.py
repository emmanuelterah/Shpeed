"""Domain error hierarchy (P0-15). Catch specific errors; never use a bare ``except``."""


class ShpeedError(Exception):
    """Base class for every error Shpeed raises on purpose."""


class DataError(ShpeedError):
    """Source data is missing, malformed or could not be fetched."""


class ValidationError(DataError):
    """Data failed a schema or plausibility check; the transaction is rolled back."""


class LeakageError(ShpeedError):
    """A read went past the prediction cutoff. Always a bug."""


class InvariantError(ShpeedError):
    """A domain invariant was violated (e.g. probabilities do not sum to 1)."""


class BudgetExceededError(ShpeedError):
    """A run exceeded its time budget."""
