from __future__ import annotations

from typing import List

try:
    import Levenshtein  # type: ignore
except Exception:  # noqa: BLE001
    Levenshtein = None  # type: ignore


def token_accuracy(y_true: List[str], y_pred: List[str], pad_token: str = "<pad>") -> float:
    if not y_true:
        return 0.0
    n = min(len(y_true), len(y_pred))
    correct = 0
    total = 0
    for i in range(n):
        if y_true[i] == pad_token:
            continue
        total += 1
        if y_true[i] == y_pred[i]:
            correct += 1
    return float(correct) / max(total, 1)


def levenshtein_similarity(a: str, b: str) -> float:
    """Return normalized similarity 1 - dist/maxlen. Falls back to DP if Levenshtein unavailable."""
    if a == b:
        return 1.0
    if len(a) == 0 or len(b) == 0:
        return 0.0
    if Levenshtein is not None:
        dist = Levenshtein.distance(a, b)  # type: ignore
    else:
        dist = _levenshtein_fallback(a, b)
    return 1.0 - (float(dist) / float(max(len(a), len(b))))


def _levenshtein_fallback(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,      # deletion
                dp[j - 1] + 1,  # insertion
                prev + cost,    # substitution
            )
            prev = temp
    return dp[m]


