"""Responses refusal parsing shared by transport validation and normalization.

Keep this independent of adapters and provider discovery: providers import the
transport while their registry is still being assembled.
"""

from typing import Any


CONTENT_POLICY_ERROR_CODES = frozenset({"cyber_policy", "content_policy_violation", "content_filter"})


class ResponsesPolicyError(RuntimeError):
    """Keep the provider's structured refusal available to the normal error classifier."""

    def __init__(self, code: str, message: Any) -> None:
        detail = str(message).strip() if message else ""
        super().__init__(f"{code}: {detail}" if detail else code)
        self.body = {"code": code, "message": message}


def raise_responses_policy_error(response: Any) -> None:
    """A failed policy response is terminal even when it contains no output items."""
    status = str(getattr(response, "status", "") or "").strip().lower()
    error = getattr(response, "error", None)
    code = error.get("code") if isinstance(error, dict) else getattr(error, "code", None)
    code = str(code or "").strip().lower()
    if status == "failed" and code in CONTENT_POLICY_ERROR_CODES:
        message = error.get("message") if isinstance(error, dict) else getattr(error, "message", None)
        raise ResponsesPolicyError(code, message)
