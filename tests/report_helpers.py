import re
from pathlib import Path
from urllib.parse import unquote


def table_rows(path: Path, heading: str | None = None) -> list[list[str]]:
    document = path.read_text("utf-8")
    if heading is not None:
        document = document.split(f"## {heading}\n", 1)[1]
    section = document.split("\n## ", 1)[0]
    return [
        [cell.strip() for cell in line.split("|")[1:-1]]
        for line in section.splitlines()
        if line.startswith("|")
    ][2:]


def call_reports(path: Path) -> list[Path]:
    document = path.read_text("utf-8")
    if "## Calls\n" not in document:
        return []
    section = document.split("## Calls\n", 1)[1].split("\n## ", 1)[0]
    reports: list[Path] = []
    for target in re.findall(r"^- \[.*\]\((.*)\)$", section, re.MULTILINE):
        relative = Path(unquote(target))
        assert not relative.is_absolute()
        report = (path.parent / relative).resolve()
        assert report.is_file()
        reports.append(report)
    return reports
