#!/usr/bin/env python3
"""Launch the GeoResQ Interactive Visual Triage & All-Weather Studio."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from georesq.dashboard.server import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
