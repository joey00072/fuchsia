import re
from typing import Iterable, Optional, Sequence

try:
    from transformers import PreTrainedTokenizerBase
except Exception:  # pragma: no cover - optional dependency in some runtimes
    PreTrainedTokenizerBase = object  # type: ignore[misc,assignment]


_LEADING_TOKENS_RE = re.compile(
    r"^(?:\s*(?:<\|begin_of_text\|>|<\|bos\|>|<\|startoftext\|>))+",
    flags=re.IGNORECASE,
)

_ASSISTANT_HEADER_RE = re.compile(
    r"(?m)(?:^|\n)\s*(?:"
    r"<\|start_header_id\|>\s*assistant\s*<\|end_header_id\|>|"
    r"<\|im_start\|>\s*assistant|"
    r"<\|assistant\|>|"
    r"assistant\s*:)"
    r"\s*",
    flags=re.IGNORECASE,
)

_TRAILING_TOKENS_RE = re.compile(
    r"(?:\s*(?:<\|eot_id\|>|<\|im_end\|>|<\|end_of_text\|>|<\|endoftext\|>|</s>))+"
    r"\s*$",
    flags=re.IGNORECASE,
)


def clean_completion(
    text: str,
    *,
    tokenizer: Optional["PreTrainedTokenizerBase"] = None,
    token_ids: Optional[Sequence[int]] = None,
) -> str:
    """Normalize a completion by stripping common chat template markers."""
    if token_ids is not None and tokenizer is not None:
        try:
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
        except Exception:
            pass

    if not text:
        return ""

    text = _LEADING_TOKENS_RE.sub("", text)

    matches = list(_ASSISTANT_HEADER_RE.finditer(text))
    if matches:
        text = text[matches[-1].end():]

    text = _TRAILING_TOKENS_RE.sub("", text)
    return text.strip()


def clean_completions(
    texts: Iterable[str],
    *,
    tokenizer: Optional["PreTrainedTokenizerBase"] = None,
    token_ids_list: Optional[Sequence[Sequence[int]]] = None,
) -> list[str]:
    cleaned: list[str] = []
    if token_ids_list is None:
        for text in texts:
            cleaned.append(clean_completion(text, tokenizer=tokenizer))
        return cleaned

    for text, token_ids in zip(texts, token_ids_list):
        cleaned.append(
            clean_completion(
                text,
                tokenizer=tokenizer,
                token_ids=token_ids,
            )
        )
    return cleaned
