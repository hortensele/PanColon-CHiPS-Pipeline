#!/usr/bin/env python3
"""PanColon-CHiPS-Pipeline entry point.

Run `python pancolon_pipeline.py --help` or `python pancolon_pipeline.py list`.
"""
import sys
from pathlib import Path

# Make the bundled package importable when run from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pancolon.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
