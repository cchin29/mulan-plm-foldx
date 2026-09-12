"""Evaluation split protocols.

    from mulan import splits
    splits.list_protocols()                    # the leakage-controlled ones
    splits.get_protocol("cath").folds
    splits.comparability_warning("bycomplex")  # what to say beside the number
"""

from .protocols import (
    PROTOCOLS,
    Protocol,
    comparability_warning,
    frontier_comparable,
    get_protocol,
    list_protocols,
)

__all__ = ["PROTOCOLS", "Protocol", "comparability_warning", "frontier_comparable",
           "get_protocol", "list_protocols"]
