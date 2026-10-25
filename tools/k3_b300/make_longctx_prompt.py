#!/usr/bin/env python3
"""The accept suite's long-context prompt: ~30K tokens of design notes, then a coding task (stdout)."""
FILLER = ("Design note: the service keeps a bounded cache of parsed configuration documents keyed by path and version. "
          "Entries expire after a configurable TTL; a background sweeper evicts the least recently used entries when the "
          "size limit is exceeded, and every mutation is recorded in an append-only audit log with a monotonic sequence number.\n")
TASK = ("\n\nBased on the design notes above, write a complete Python module implementing this cache service: a ConfigCache "
        "class with get/put/invalidate, TTL expiry, LRU eviction with a size limit, the audit log with monotonic sequence "
        "numbers, thread safety, full docstrings and a pytest test suite. Return code only.\n")
print(FILLER * int(30000 * 4.3 / len(FILLER)) + TASK, end="")
