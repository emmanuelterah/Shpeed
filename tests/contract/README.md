# Contract tests

One suite per port. Every implementation (including test doubles such as `InMemoryStore`) is added to the suite's parameter list and must pass it unchanged. This is how Liskov substitution is enforced.
