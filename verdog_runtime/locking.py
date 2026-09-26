"""Cross-platform advisory file locks shared with runtime clients."""

from verdog_runtime._file_lock import locked_file as locked_file

__all__ = ["locked_file"]
