"""Compatibility shim for the canonical CoSearch QA reward implementation.

The implementation lives in ``CoAgenticRetriever/rewards``.  Keep this module so
legacy VERL paths and existing configs continue to import the same symbols.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rewards.search_qa_f1_with_format_penalty import *  # noqa: F401,F403
