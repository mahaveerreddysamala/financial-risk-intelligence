"""Execute the Streamlit dashboard headlessly for CI smoke validation."""
from __future__ import annotations

import os
from pathlib import Path

from streamlit.testing.v1 import AppTest


def main() -> None:
    os.environ.setdefault("FINANCIAL_DASHBOARD_ROWS", "5000")
    app = Path(__file__).parents[1] / "dashboards" / "financial_risk_app.py"
    test = AppTest.from_file(str(app), default_timeout=120).run()
    if test.exception:
        raise AssertionError(f"Dashboard raised exceptions: {test.exception}")
    titles = [element.value for element in test.title]
    if "Financial Risk Intelligence" not in titles:
        raise AssertionError(f"Expected dashboard title, found: {titles}")
    if len(test.metric) < 5:
        raise AssertionError("Expected at least five executive metrics")
    print(f"Dashboard smoke passed with {len(test.metric)} rendered metrics")


if __name__ == "__main__":
    main()
