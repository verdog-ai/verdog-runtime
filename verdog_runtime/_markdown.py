"""Markdown formatting shared by invocation reports."""

from collections.abc import Iterable
from html import escape

from tabulate import tabulate


def format_cell(value: object) -> str:
    try:
        rendered = str(value)
    except Exception:
        rendered = f"[unprintable {type(value).__name__}]"
    return (
        escape(rendered, quote=False)
        .replace("|", "&#124;")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", "<br>")
    )


def format_table(rows: Iterable[Iterable[object]], headers: tuple[str, ...]) -> str:
    return tabulate(
        (tuple(format_cell(cell) for cell in row) for row in rows),
        headers=headers,
        tablefmt="pipe",
        disable_numparse=True,
    )
