"""Reserved extension point for extra signal scorers."""

from __future__ import annotations


class ExtraScorerStub:
    def score(self, *_args, **_kwargs):
        return None
