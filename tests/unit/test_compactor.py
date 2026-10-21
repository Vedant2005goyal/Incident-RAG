"""Unit tests for the LLMLingua-2 compactor module.

Validates that:
- Output token count ≤ budget + 5 %.
- All ``<TRACE_ID:*>`` tags present in output.
- Empty input returns empty string.
- Input already within budget is returned unchanged.
- Multiple tag types are preserved.

Runs without the model (it can't load in tests), so earlier words count as more important.
"""

from __future__ import annotations

import pytest

from src.compactor import TAG, LLMLingua2Compactor, word_scores


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


@pytest.fixture()
def compactor() -> LLMLingua2Compactor:
    """Return a compactor instance using the extractive fallback (no model).

    We pass a non-existent model name to guarantee that ``_load_model``
    fails and the compactor uses the extractive fallback path.
    """
    return LLMLingua2Compactor(model_name="__nonexistent_model_for_tests__")


LONG_CHUNKS = [
    (
        "The payments service lost connectivity to <IP_ADDR:192.168.1.100> "
        "at 09:57 UTC. Error log shows <TRACE_ID:abc123def456> with status "
        "code 500. The database primary <IP_ADDR:10.0.0.5> was unreachable. "
        "Multiple retries failed over a three-minute window causing cascading "
        "failures across the gateway and auth services."
    ),
    (
        "Runbook procedure: When <ERROR_CODE:ERR_CONN_RESET> occurs trigger "
        "manual failover to db-replica-1 via ops-console. Verify health with "
        "systemctl status postgresql and confirm replication lag below two seconds. "
        "Cross-reference with <TRACE_ID:trace789xyz> in the distributed tracing "
        "dashboard before closing the incident ticket."
    ),
    (
        "Slack discussion thread regarding outage on 2026-04-06. User A reports "
        "seeing connection refused errors in payments service logs. User B confirms "
        "<HASH:d8c05cb> commit deployed thirty minutes prior introduced a config "
        "regression. Path <PATH:/var/log/payments/app.log> shows the exact failure "
        "timestamp. Resolution: revert commit and restart the service pod."
    ),
]


# -----------------------------------------------------------------------
# Tests: output token count within budget
# -----------------------------------------------------------------------


class TestTokenBudget:
    """Verify the compactor respects the token budget."""

    def test_output_within_budget(self, compactor: LLMLingua2Compactor) -> None:
        """Compressed output must have ≤ budget + 5 % tokens."""
        budget = 30
        result = compactor.compress(LONG_CHUNKS, budget)
        token_count = len(result.split())
        max_allowed = int(budget * 1.05) + 1  # allow rounding
        assert token_count <= max_allowed, (
            f"Token count {token_count} exceeds budget {budget} + 5 % "
            f"(max {max_allowed})"
        )

    def test_tight_budget(self, compactor: LLMLingua2Compactor) -> None:
        """Even a very tight budget should not exceed the ceiling."""
        budget = 10
        result = compactor.compress(LONG_CHUNKS, budget)
        token_count = len(result.split())
        max_allowed = int(budget * 1.05) + 1
        assert token_count <= max_allowed

    def test_generous_budget_returns_all_content(
        self, compactor: LLMLingua2Compactor
    ) -> None:
        """When budget exceeds input length, all content is preserved."""
        budget = 5000
        result = compactor.compress(LONG_CHUNKS, budget)
        full = "\n\n".join(LONG_CHUNKS)
        # All words from the original should appear (order-independent)
        original_words = set(full.split())
        result_words = set(result.split())
        assert original_words == result_words


# -----------------------------------------------------------------------
# Tests: technical-ID tag preservation
# -----------------------------------------------------------------------


class TestTagPreservation:
    """Verify that <TAG:value> technical identifiers survive compression."""

    def test_trace_ids_preserved(self, compactor: LLMLingua2Compactor) -> None:
        """All <TRACE_ID:*> tags must appear in the compressed output."""
        budget = 30
        result = compactor.compress(LONG_CHUNKS, budget)
        assert "<TRACE_ID:abc123def456>" in result
        assert "<TRACE_ID:trace789xyz>" in result

    def test_all_tag_types_preserved(
        self, compactor: LLMLingua2Compactor
    ) -> None:
        """IP, HASH, PATH, and ERROR_CODE tags must also survive."""
        budget = 40
        result = compactor.compress(LONG_CHUNKS, budget)

        expected_tags = [
            "<IP_ADDR:192.168.1.100>",
            "<IP_ADDR:10.0.0.5>",
            "<ERROR_CODE:ERR_CONN_RESET>",
            "<HASH:d8c05cb>",
            "<PATH:/var/log/payments/app.log>",
            "<TRACE_ID:abc123def456>",
            "<TRACE_ID:trace789xyz>",
        ]
        for tag in expected_tags:
            assert tag in result, f"Missing tag in output: {tag}"


# -----------------------------------------------------------------------
# Tests: edge cases
# -----------------------------------------------------------------------


class TestEdgeCases:
    """Edge-case coverage for the compactor."""

    def test_empty_chunks(self, compactor: LLMLingua2Compactor) -> None:
        """Empty chunk list returns empty string."""
        assert compactor.compress([], budget_tokens=100) == ""

    def test_single_empty_chunk(self, compactor: LLMLingua2Compactor) -> None:
        """A list containing one empty string returns empty string."""
        assert compactor.compress([""], budget_tokens=100) == ""

    def test_no_tags_still_compresses(
        self, compactor: LLMLingua2Compactor
    ) -> None:
        """Chunks without any tags can still be compressed."""
        chunks = ["This is a plain sentence without any special identifiers."]
        result = compactor.compress(chunks, budget_tokens=5)
        assert len(result.split()) <= 6  # budget + tolerance


# -----------------------------------------------------------------------
# Tests: internal helpers
# -----------------------------------------------------------------------


class TestHelpers:
    """Validate internal helper functions."""

    def test_tag_regex_matches(self) -> None:
        """The TAG regex matches all expected tag formats."""
        for tag in ["<IP_ADDR:192.168.1.1>", "<TRACE_ID:abc123>", "<HASH:d8c05cb>",
                    "<PATH:/var/log/app.log>", "<ERROR_CODE:ERR_CONN_RESET>"]:
            assert TAG.search(tag), f"TAG did not match {tag}"

    def test_word_scores_take_the_best_piece_of_each_word(self) -> None:
        """Each word gets the highest score of its sub-word pieces (the model path)."""
        words = ["replica", "lag", "grew"]
        offsets = [[0, 0], [0, 3], [3, 7], [8, 11], [12, 16], [0, 0]]  # [CLS] rep lica lag grew [SEP]
        probabilities = [0.0, 0.2, 0.9, 0.4, 0.7, 0.0]

        assert word_scores(words, offsets, probabilities) == [0.9, 0.4, 0.7]
