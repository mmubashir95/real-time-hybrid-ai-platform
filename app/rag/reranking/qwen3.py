"""Load and score with the pinned Qwen3 generative reranker.

This adapter follows Qwen's documented yes/no next-token scoring. It owns model
input formatting and inference only; authorization and retrieval stay in Phase 7.
"""

from __future__ import annotations

from collections.abc import Sequence
import logging
from threading import Lock
from time import perf_counter

from app.rag.reranking.models import PairScore, RerankerConfig


logger = logging.getLogger(__name__)
SYSTEM_PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on the '
    'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    '<|im_end|>\n<|im_start|>user\n'
)
ASSISTANT_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


class RerankerInferenceError(RuntimeError):
    """Report model loading or prediction failures without silent fallback."""


class Qwen3Reranker:
    """Lazily reuse one pinned Qwen3 model and tokenizer for bounded scoring."""

    def __init__(self, config: RerankerConfig = RerankerConfig()) -> None:
        """Store trusted configuration without downloading artifacts at import."""
        self.config = config
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._model_load_seconds = 0.0
        # Serialize initialization and inference for one shared mutable model.
        # Request concurrency remains bounded instead of racing to load twice.
        self._inference_lock = Lock()

    @property
    def model_load_seconds(self) -> float:
        """Return measured initialization time, or zero before first inference."""
        return self._model_load_seconds

    def score(self, query: str, passages: Sequence[str]) -> tuple[PairScore, ...]:
        """Score ordered question-passage pairs in bounded inference batches."""
        if not passages:
            return ()
        with self._inference_lock:
            self._ensure_loaded()
            scores: list[PairScore] = []
            for start in range(0, len(passages), self.config.batch_size):
                batch = passages[start : start + self.config.batch_size]
                scores.extend(self._score_batch(query, batch))
        if len(scores) != len(passages):
            raise RerankerInferenceError("Reranker returned an unexpected score count")
        return tuple(scores)

    def _ensure_loaded(self) -> None:
        """Initialize the official causal-LM scoring dependencies exactly once."""
        if self._model is not None:
            return
        started = perf_counter()
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            dtype = {
                "auto": "auto",
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[self.config.precision]
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                revision=self.config.tokenizer_revision,
                padding_side="left",
                local_files_only=self.config.local_files_only,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                revision=self.config.model_revision,
                torch_dtype=dtype,
                local_files_only=self.config.local_files_only,
            ).to(self.config.device)
            model.eval()
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            raise RerankerInferenceError("Qwen3 reranker initialization failed") from error
        self._torch, self._tokenizer, self._model = torch, tokenizer, model
        self._model_load_seconds = perf_counter() - started
        logger.info(
            "reranker_model_loaded model_id=%s revision=%s device=%s seconds=%.6f",
            self.config.model_id, self.config.model_revision,
            self.config.device, self._model_load_seconds,
        )

    def _prepare_input(self, query: str, passage: str) -> tuple[list[int], bool]:
        """Preserve instruction/query tokens and truncate only passage content."""
        assert self._tokenizer is not None
        user_prefix = (
            f"<Instruct>: {self.config.instruction}\n<Query>: {query}\n<Document>: "
        )
        fixed_tokens = self._tokenizer.encode(
            SYSTEM_PREFIX + user_prefix, add_special_tokens=False
        )
        suffix_tokens = self._tokenizer.encode(ASSISTANT_SUFFIX, add_special_tokens=False)
        available_passage_tokens = self.config.max_length - len(fixed_tokens) - len(suffix_tokens)
        if available_passage_tokens <= 0:
            raise ValueError("Query and reranking instruction exceed the input token budget")
        passage_tokens = self._tokenizer.encode(passage, add_special_tokens=False)
        truncated = len(passage_tokens) > available_passage_tokens
        # Passage-only truncation prevents a long chunk from removing the task or question.
        return (
            fixed_tokens + passage_tokens[:available_passage_tokens] + suffix_tokens,
            truncated,
        )

    def _score_batch(self, query: str, passages: Sequence[str]) -> list[PairScore]:
        """Return Qwen's raw yes/no logit differences for one bounded batch."""
        assert self._tokenizer is not None and self._model is not None and self._torch is not None
        prepared = [self._prepare_input(query, passage) for passage in passages]
        features = self._tokenizer.pad(
            {"input_ids": [item[0] for item in prepared]},
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        features = {name: value.to(self._model.device) for name, value in features.items()}
        try:
            with self._torch.inference_mode():
                logits = self._model(**features).logits[:, -1, :]
            yes_id = self._tokenizer.convert_tokens_to_ids("yes")
            no_id = self._tokenizer.convert_tokens_to_ids("no")
            if not isinstance(yes_id, int) or not isinstance(no_id, int) or yes_id == no_id:
                raise ValueError("Tokenizer does not expose distinct yes/no scoring tokens")
            values = (logits[:, yes_id] - logits[:, no_id]).float().cpu().tolist()
        except (RuntimeError, ValueError, IndexError) as error:
            raise RerankerInferenceError("Qwen3 reranker prediction failed") from error
        if len(values) != len(prepared):
            raise RerankerInferenceError("Qwen3 returned an unexpected score count")
        return [
            PairScore(float(value), len(token_ids), truncated)
            for value, (token_ids, truncated) in zip(values, prepared, strict=True)
        ]

