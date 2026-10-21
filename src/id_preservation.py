"""After compression: put back any <TAG:value> technical ID (IP, trace ID, error code...) the compressor dropped."""

import re
from dataclasses import dataclass, field

TAG = re.compile(r"<(?:IP_ADDR|TRACE_ID|HASH|PATH|ERROR_CODE):[^>]+>")


@dataclass
class PreservationResult:
    text: str  # the compressed text, with every original tag present
    reinjected_tags: list[str] = field(default_factory=list)
    total_reinjections: int = 0


class TechnicalIDPreservationLayer:
    def verify_and_reinject(self, original_chunks: list[str], compressed_text: str) -> PreservationResult:
        """Find the tags of the original text that are missing from the compressed text, and add them back."""
        if not compressed_text:
            return PreservationResult(text=compressed_text)
        original_tags = list(dict.fromkeys(tag for chunk in original_chunks for tag in TAG.findall(chunk)))
        missing = [tag for tag in original_tags if tag not in compressed_text]
        if not missing:
            return PreservationResult(text=compressed_text)

        # Put each missing tag after the sentence that mentions it most
        sentences = re.split(r"(?<=[.!?\n])\s+", compressed_text)
        for tag in missing:
            best = _find_best_sentence_index(sentences, tag)
            sentences[best] = sentences[best].rstrip() + " " + tag
        text = " ".join(s for s in sentences if s.strip())
        return PreservationResult(text=text, reinjected_tags=missing, total_reinjections=len(missing))


def _find_best_sentence_index(sentences: list[str], tag: str) -> int:
    """The sentence sharing the most words with the tag (e.g. "trace", "abc123"); the last one if none does."""
    match = re.match(r"<(\w+):(.+)>", tag)
    if not match:
        return len(sentences) - 1
    keywords = set(match.group(1).lower().split("_")) | set(re.split(r"[.:/_\-]", match.group(2).lower()))
    keywords.discard("")

    best, best_score = len(sentences) - 1, 0
    for index, sentence in enumerate(sentences):
        score = sum(1 for keyword in keywords if keyword in sentence.lower())
        if score > best_score:
            best, best_score = index, score
    return best
