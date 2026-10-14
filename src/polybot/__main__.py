"""`python -m polybot` runs the paper-trading loop, `python -m polybot report` prints the report."""

import logging
import sys

from polybot.config import load_settings
from polybot.engine.loop import run
from polybot.report import build_report, format_report
from polybot.storage.journal import Journal


def main(argv: list[str]) -> int:
    settings = load_settings()
    if argv[:1] == ["report"]:
        print(format_report(build_report(Journal(settings.db_path))), end="")
        return 0
    if argv:
        print(f"usage: python -m polybot [report]  (unknown arguments: {argv})", file=sys.stderr)
        return 2

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    run(settings)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
