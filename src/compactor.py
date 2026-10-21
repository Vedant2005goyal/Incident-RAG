"""Shrink a chunk to a word budget with LLMLingua-2, never dropping a <TAG:value> technical ID."""

import logging
import re

logger = logging.getLogger(__name__)

TAG = re.compile(r"<(?:IP_ADDR|TRACE_ID|HASH|PATH|ERROR_CODE):[^>]+>")


class LLMLingua2Compactor:
    """Keeps the `budget_tokens` most important words, in their original order.

    Importance comes from the LLMLingua-2 model, a BERT classifier trained to tell which words a text can
    lose. Without the model (it failed to load, or in tests) earlier words count as more important.
    Words holding a technical-ID tag are always kept.
    """

    def __init__(self, model_name: str) -> None:
        try:
            from transformers import AutoModelForTokenClassification, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name).eval()
        except Exception:
            logger.warning("compression model %s unavailable: keeping the first words instead", model_name)
            self.model = None

    def compress(self, chunks: list[str], budget_tokens: int) -> str:
        words = "\n\n".join(chunks).split()
        if len(words) <= budget_tokens:
            return " ".join(words)

        importance = self._importance(words) if self.model else [-index for index in range(len(words))]
        tags = {i for i, word in enumerate(words) if TAG.search(word)}
        others = sorted((i for i in range(len(words)) if i not in tags), key=lambda i: importance[i], reverse=True)
        keep = tags | set(others[: max(0, budget_tokens - len(tags))])
        return " ".join(words[i] for i in sorted(keep))

    def _importance(self, words: list[str]) -> list[float]:
        """The model's probability of keeping each word."""
        import torch

        inputs = self.tokenizer(" ".join(words), return_tensors="pt", truncation=True, max_length=512,
                                return_offsets_mapping=True)
        offsets = inputs.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            keep_probability = torch.softmax(self.model(**inputs).logits[0], dim=-1)[:, 1].tolist()  # label 1 = keep
        return word_scores(words, offsets, keep_probability)


def word_scores(words: list[str], offsets: list[list[int]], probabilities: list[float]) -> list[float]:
    """Turn the model's per-piece scores into per-word scores.

    The model splits words into sub-word pieces ("replica" → "rep", "lica"); `offsets` gives each piece's
    character span in the text (words joined by single spaces). A word's score is its best piece's score.
    """
    scores, start = [], 0
    for word in words:
        end = start + len(word)
        pieces = [p for (s, e), p in zip(offsets, probabilities) if (s, e) != (0, 0) and e > start and s < end]
        scores.append(max(pieces, default=0.0))  # (0, 0) = the model's special tokens, not text
        start = end + 1
    return scores
