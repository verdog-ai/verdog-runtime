"""The Verdog workflow runtime.

Byte-identical for every project, which is why it is one installed package
rather
than a copy inside each: one shared runtime keeps execution types identical
across
projects and lets subroutines interoperate directly.
"""

from verdog_runtime.cancellation import (
    CancellationToken as CancellationToken,
)
from verdog_runtime.cancellation import (
    ExecutionCancelled as ExecutionCancelled,
)

__all__ = ["CancellationToken", "ExecutionCancelled"]
