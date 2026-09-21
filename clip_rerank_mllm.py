from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Callable

from config import settings

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable: Any, **_: Any) -> Any:  # type: ignore[misc]
        return iterable


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except Exception:
        return default
    if math.isnan(x):
        return default
    return x


def _clamp01(value: Any, default: float = 0.0) -> float:
    x = _safe_float(value, default=default)
    return max(0.0, min(1.0, x))


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-float(x)))
    except Exception:
        return 0.0


@dataclass
class ClipCandidate:
    idx: int
    source: str
    text: str
    clip_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0


class ClipRerankMLLMPipeline:
    """
    CLIP 检索 + BGE reranker + MLLM 作答。
    只消费外部候选证据 corpus，不调用 web search；
    默认由 ENABLE_CLIP_RERANK_MLLM 控制，关闭时不会影响原逻辑；
    CLIP / BGE 模型加载失败时，返回 trace，并保守退回候选原顺序，避免整轮评测崩溃。
    """

    def __init__(self) -> None:
        self.clip_model_name = str(
            getattr(settings, "clip_model_name", os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32"))
            or "openai/clip-vit-base-patch32"
        ).strip()
        self.clip_device = str(
            getattr(settings, "clip_device", os.getenv("CLIP_DEVICE", "auto"))
            or "auto"
        ).strip().lower()
        self.clip_local_files_only = bool(
            getattr(settings, "clip_local_files_only", False)
        )

        self.recall_top_k = max(
            1,
            int(getattr(settings, "clip_recall_top_k", os.getenv("CLIP_RECALL_TOP_K", "16")) or 16),
        )
        self.context_top_k = max(
            1,
            int(getattr(settings, "clip_context_top_k", os.getenv("CLIP_CONTEXT_TOP_K", "6")) or 6),
        )
        self.batch_size = max(
            1,
            int(getattr(settings, "clip_batch_size", os.getenv("CLIP_BATCH_SIZE", "16")) or 16),
        )
        self.max_doc_chars = max(
            200,
            int(getattr(settings, "clip_doc_max_chars", os.getenv("CLIP_DOC_MAX_CHARS", "1200")) or 1200),
        )
        
        self.show_progress = str(
            os.getenv(
                "ENABLE_CLIP_RERANK_PROGRESS",
                str(getattr(settings, "enable_clip_rerank_progress", "1")),
            )
        ).strip().lower() in {"1", "true", "yes", "on"}

        self.progress_leave = str(
            os.getenv(
                "CLIP_RERANK_PROGRESS_LEAVE",
                str(getattr(settings, "clip_rerank_progress_leave", "0")),
            )
        ).strip().lower() in {"1", "true", "yes", "on"}

        self.enable_bge_reranker = bool(
            getattr(settings, "enable_bge_reranker", True)
        )
        self.bge_reranker_model_name = str(
            getattr(
                settings,
                "bge_reranker_model_name",
                os.getenv("BGE_RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3"),
            )
            or "BAAI/bge-reranker-v2-m3"
        ).strip()
        self.bge_reranker_device = str(
            getattr(settings, "bge_reranker_device", os.getenv("BGE_RERANKER_DEVICE", self.clip_device))
            or self.clip_device
        ).strip().lower()
        self.bge_batch_size = max(
            1,
            int(getattr(settings, "bge_reranker_batch_size", os.getenv("BGE_RERANKER_BATCH_SIZE", "8")) or 8),
        )

        self._torch = None
        self._clip_model = None
        self._clip_processor = None
        self._cross_encoder = None
        self._load_error = ""

        self._init_clip()
        self._init_reranker()

    def _resolve_device(self, requested: str) -> str:
        requested = str(requested or "auto").strip().lower()
        try:
            import torch

            if requested in {"cuda", "cpu", "mps"}:
                return requested
            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        except Exception:
            return "cpu"

    def _init_clip(self) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._torch = torch
            device = self._resolve_device(self.clip_device)
            self.clip_device = device

            self._clip_processor = CLIPProcessor.from_pretrained(
                self.clip_model_name,
                local_files_only=self.clip_local_files_only,
            )
            self._clip_model = CLIPModel.from_pretrained(
                self.clip_model_name,
                local_files_only=self.clip_local_files_only,
            )
            self._clip_model.to(device)
            self._clip_model.eval()
        except Exception as exc:
            self._load_error = f"clip_load_failed:{type(exc).__name__}: {exc}"
            self._clip_model = None
            self._clip_processor = None

    def _init_reranker(self) -> None:
        if not self.enable_bge_reranker:
            return
        try:
            from sentence_transformers import CrossEncoder

            device = self._resolve_device(self.bge_reranker_device)
            self.bge_reranker_device = device

            self._cross_encoder = CrossEncoder(
                self.bge_reranker_model_name,
                device=device,
            )
        except Exception as exc:
            msg = f"bge_load_failed:{type(exc).__name__}: {exc}"
            self._load_error = f"{self._load_error} | {msg}".strip(" | ")
            self._cross_encoder = None

    def _normalize_text_for_clip(self, text: str) -> str:
        text = " ".join(str(text or "").split()).strip()
        return text[: self.max_doc_chars]

    def _corpus_to_candidates(self, corpus: list[tuple[str, str]]) -> list[ClipCandidate]:
        out: list[ClipCandidate] = []
        seen: set[tuple[str, str]] = set()

        for source, text in corpus or []:
            src = str(source or "unknown").strip() or "unknown"
            txt = self._normalize_text_for_clip(str(text or ""))
            if not txt:
                continue

            key = (src, txt[:300])
            if key in seen:
                continue
            seen.add(key)

            out.append(
                ClipCandidate(
                    idx=len(out),
                    source=src,
                    text=txt,
                )
            )

        return out

    def _encode_texts(self, texts: list[str], desc: str = "CLIP encode"):
        if self._clip_model is None or self._clip_processor is None or self._torch is None:
            return None
        if not texts:
            return None

        torch = self._torch
        all_vecs = []

        batch_starts = range(0, len(texts), self.batch_size)
        if self.show_progress:
            batch_starts = tqdm(
                batch_starts,
                total=(len(texts) + self.batch_size - 1) // self.batch_size,
                desc=desc,
                leave=self.progress_leave,
                dynamic_ncols=True,
            )

        with torch.no_grad():
            for start in batch_starts:
                batch = texts[start : start + self.batch_size]
                inputs = self._clip_processor(
                    text=batch,
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.clip_device) for k, v in inputs.items()}
                vec = self._clip_model.get_text_features(**inputs)
                vec = vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                all_vecs.append(vec.detach().cpu())

        if not all_vecs:
            return None
        return torch.cat(all_vecs, dim=0)

    def _clip_recall(self, question: str, candidates: list[ClipCandidate]) -> tuple[list[ClipCandidate], dict[str, Any]]:
        trace: dict[str, Any] = {
            "clip_model": self.clip_model_name,
            "clip_device": self.clip_device,
            "clip_recall_top_k": self.recall_top_k,
            "clip_ok": False,
            "clip_error": self._load_error,
        }

        if not candidates:
            trace["clip_error"] = "empty_candidates"
            return [], trace

        if self._clip_model is None:
            # CLIP 加载失败时保守返回前 recall_top_k，避免评测中断。
            fallback = candidates[: self.recall_top_k]
            for item in fallback:
                item.clip_score = 0.0
            return fallback, trace

        try:
            q_vec = self._encode_texts([question], desc="CLIP encode query")
            d_vecs = self._encode_texts(
                [f"{c.source}\n{c.text}" for c in candidates],
                desc=f"CLIP recall docs n={len(candidates)}",
            )
            if q_vec is None or d_vecs is None:
                trace["clip_error"] = "clip_encode_empty"
                return candidates[: self.recall_top_k], trace

            scores = (d_vecs @ q_vec[0]).tolist()
            scored: list[ClipCandidate] = []
            for cand, score in zip(candidates, scores):
                row = ClipCandidate(
                    idx=cand.idx,
                    source=cand.source,
                    text=cand.text,
                    clip_score=float(score),
                )
                scored.append(row)

            scored.sort(key=lambda x: x.clip_score, reverse=True)
            trace["clip_ok"] = True
            trace["clip_error"] = ""
            return scored[: self.recall_top_k], trace
        except Exception as exc:
            trace["clip_error"] = f"clip_recall_failed:{type(exc).__name__}: {exc}"
            return candidates[: self.recall_top_k], trace

    def _bge_rerank(self, question: str, candidates: list[ClipCandidate]) -> tuple[list[ClipCandidate], dict[str, Any]]:
        trace: dict[str, Any] = {
            "bge_enabled": self.enable_bge_reranker,
            "bge_model": self.bge_reranker_model_name,
            "bge_device": self.bge_reranker_device,
            "bge_ok": False,
            "bge_error": "",
        }

        if not candidates:
            trace["bge_error"] = "empty_candidates"
            return [], trace

        if not self.enable_bge_reranker or self._cross_encoder is None:
            out = []
            for cand in candidates:
                row = ClipCandidate(
                    idx=cand.idx,
                    source=cand.source,
                    text=cand.text,
                    clip_score=cand.clip_score,
                    rerank_score=cand.clip_score,
                    final_score=cand.clip_score,
                )
                out.append(row)
            out.sort(key=lambda x: x.final_score, reverse=True)
            if self.enable_bge_reranker and self._cross_encoder is None:
                trace["bge_error"] = self._load_error or "bge_unavailable"
            return out[: self.context_top_k], trace

        try:
            pairs = [(question, f"{c.source}\n{c.text}") for c in candidates]
            raw_scores = self._cross_encoder.predict(
                pairs,
                batch_size=self.bge_batch_size,
                show_progress_bar=self.show_progress,
            )

            out: list[ClipCandidate] = []
            for cand, raw_score in zip(candidates, raw_scores):
                rerank_score = _sigmoid(float(raw_score))
                final_score = 0.35 * _clamp01((cand.clip_score + 1.0) / 2.0) + 0.65 * rerank_score
                out.append(
                    ClipCandidate(
                        idx=cand.idx,
                        source=cand.source,
                        text=cand.text,
                        clip_score=cand.clip_score,
                        rerank_score=rerank_score,
                        final_score=final_score,
                    )
                )

            out.sort(key=lambda x: x.final_score, reverse=True)
            trace["bge_ok"] = True
            return out[: self.context_top_k], trace
        except Exception as exc:
            trace["bge_error"] = f"bge_rerank_failed:{type(exc).__name__}: {exc}"
            out = []
            for cand in candidates:
                out.append(
                    ClipCandidate(
                        idx=cand.idx,
                        source=cand.source,
                        text=cand.text,
                        clip_score=cand.clip_score,
                        rerank_score=cand.clip_score,
                        final_score=cand.clip_score,
                    )
                )
            out.sort(key=lambda x: x.final_score, reverse=True)
            return out[: self.context_top_k], trace

    def select_evidence(self, question: str, corpus: list[tuple[str, str]]) -> tuple[list[ClipCandidate], dict[str, Any]]:
        candidates = self._corpus_to_candidates(corpus)
        recalled, clip_trace = self._clip_recall(question, candidates)
        reranked, bge_trace = self._bge_rerank(question, recalled)

        trace = {
            "executor": "clip_rerank_mllm",
            "candidate_count": len(candidates),
            "recalled_count": len(recalled),
            "selected_count": len(reranked),
            **clip_trace,
            **bge_trace,
            "selected": [
                {
                    "rank": i + 1,
                    "source": item.source,
                    "clip_score": item.clip_score,
                    "rerank_score": item.rerank_score,
                    "final_score": item.final_score,
                    "text_preview": item.text[:200],
                }
                for i, item in enumerate(reranked)
            ],
        }
        return reranked, trace


_PIPELINE: ClipRerankMLLMPipeline | None = None


def get_clip_rerank_pipeline() -> ClipRerankMLLMPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = ClipRerankMLLMPipeline()
    return _PIPELINE


def build_clip_rerank_context(selected: list[ClipCandidate]) -> str:
    if not selected:
        return ""

    lines: list[str] = []
    for i, item in enumerate(selected, start=1):
        lines.append(
            (
                f"[CLIP_RERANK_EVIDENCE_{i}] source={item.source}\n"
                f"clip_score={item.clip_score:.4f}; "
                f"rerank_score={item.rerank_score:.4f}; "
                f"final_score={item.final_score:.4f}\n"
                f"{item.text}"
            )
        )
    return "\n\n".join(lines)


def candidates_to_evidence(selected: list[ClipCandidate]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for item in selected:
        evidence.append(
            {
                "source": item.source,
                "claim": "CLIP 检索 + BGE 重排选中的候选证据",
                "excerpt": item.text,
                "confidence": _clamp01(item.final_score, default=0.0),
                "supports": [],
                "contradicts": [],
                "evidence_type": "kb_chunk",
                "premise_dependencies": [],
                "ambiguity": 0.20,
                "exclusivity": _clamp01(item.rerank_score, default=0.0),
                "clip_score": item.clip_score,
                "rerank_score": item.rerank_score,
                "final_score": item.final_score,
            }
        )
    return evidence


def run_clip_rerank_mllm(
    *,
    llm: Any,
    sample: dict[str, Any],
    image_ref: str | None,
    corpus: list[tuple[str, str]],
    answer_fn: Callable[..., str],
    choices: list[str],
    map_answer_fn: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    question = str(sample.get("question", "")).strip()
    dataset_type = str(sample.get("dataset_type", "generic")).strip() or "generic"

    if llm is None:
        return {
            "mode": "clip_rerank_mllm",
            "answer": "",
            "context": "",
            "evidence": [],
            "retrieval_trace": {
                "executor": "clip_rerank_mllm",
                "error": "llm_missing",
            },
            "error": "llm_missing",
            "final_confidence": 0.0,
            "decision_status": "fallback",
            "decision_reason": "clip_rerank_mllm_llm_missing",
        }

    pipeline = get_clip_rerank_pipeline()
    selected, trace = pipeline.select_evidence(question, corpus)
    context = build_clip_rerank_context(selected)
    evidence = candidates_to_evidence(selected)

    try:
        pred = answer_fn(
            llm=llm,
            question=question,
            context=context,
            image_ref=image_ref,
            dataset_type=dataset_type,
            choices=choices,
        )
        pred = str(pred or "").strip()
        if map_answer_fn is not None:
            pred = map_answer_fn(pred)

        confidence = 0.0
        if selected:
            confidence = sum(_clamp01(x.final_score) for x in selected) / max(len(selected), 1)

        return {
            "mode": "clip_rerank_mllm",
            "answer": pred,
            "context": context,
            "evidence": evidence,
            "retrieval_trace": trace,
            "error": None,
            "final_confidence": _clamp01(confidence, default=0.0),
            "decision_status": "finalized" if pred else "fallback",
            "decision_reason": "clip_rerank_mllm_success" if pred else "clip_rerank_mllm_empty_answer",
        }
    except Exception as exc:
        return {
            "mode": "clip_rerank_mllm",
            "answer": "",
            "context": context,
            "evidence": evidence,
            "retrieval_trace": {
                **trace,
                "exception": f"{type(exc).__name__}: {exc}",
            },
            "error": f"{type(exc).__name__}: {exc}",
            "final_confidence": 0.0,
            "decision_status": "fallback",
            "decision_reason": "clip_rerank_mllm_exception",
        }