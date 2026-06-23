"""Compatibility shim for canonical reranker UID grouping functions."""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rewards.uid_group_functions import *  # noqa: F401,F403
