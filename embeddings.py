from __future__ import annotations

import json
import math
import time
from typing import Any

from openai import OpenAI

from config import settings


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a)) or 1e-8


def cosine_similarity(a: list[float], b: list[float]) -> float:
    return _dot(a, b) / (_norm(a) * _norm(b))


class EmbeddingReranker:
    """
    第一阶段：embedding 相似度排序
    可选第二阶段：用 qwen chat 对 top-N 再细排
    """

    def __init__(self) -> None:
        self.enabled = bool(getattr(settings, "enable_embedding_rerank", False))

        emb_base = str(getattr(settings, "embedding_api_base", "") or getattr(settings, "api_base", "")).strip()
        emb_key = str(getattr(settings, "embedding_api_key", "") or getattr(settings, "api_key", "")).strip()
        self.embedding_client = OpenAI(
            base_url=emb_base,
            api_key=emb_key,
            timeout=getattr(settings, "api_timeout_s", 30),
        ) if emb_base and emb_key else None

        self.embedding_model = str(getattr(settings, "embedding_model_name", "text-embedding-v4")).strip()

        self.enable_chat_rerank = bool(getattr(settings, "enable_qwen_chat_rerank", False))
        rerank_base = str(getattr(settings, "rerank_api_base", "") or getattr(settings, "api_base", "")).strip()
        rerank_key = str(getattr(settings, "rerank_api_key", "") or getattr(settings, "api_key", "")).strip()
        self.rerank_chat_client = OpenAI(
            base_url=rerank_base,
            api_key=rerank_key,
            timeout=getattr(settings, "api_timeout_s", 30),
        ) if self.enable_chat_rerank and rerank_base and rerank_key else None
        self.rerank_chat_model = str(getattr(settings, "rerank_chat_model_name", "qwen-plus")).strip()

    
    def _retry_attempts(self) -> int:
        try:
            n = int(getattr(settings, "api_retries", 3) or 3)
        except Exception:
            n = 3
        return max(1, min(n, 3))


    def _retry_sleep_s(self, attempt_idx: int) -> float:
        try:
            base = float(getattr(settings, "api_retry_backoff_s", 2) or 2)
        except Exception:
            base = 2.0
        try:
            cap = float(getattr(settings, "api_retry_max_wait_s", 120) or 120)
        except Exception:
            cap = 120.0
        return max(0.5, min(base * (2 ** attempt_idx), cap, 120.0))


    def _with_retry(self, fn):
        last_error: Exception | None = None
        attempts = self._retry_attempts()

        for i in range(attempts):
            try:
                return fn()
            except Exception as exc:
                last_error = exc
                if i < attempts - 1:
                    time.sleep(self._retry_sleep_s(i))
                    continue
                raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("unexpected_retry_exit")
    
    
    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not self.embedding_client or not texts:
            return []

        resp = self._with_retry(
            lambda: self.embedding_client.embeddings.create(
                model=self.embedding_model,
                input=texts,
            )
        )
        return [list(item.embedding) for item in resp.data]

    
    def _rank_texts_by_embedding(self, query: str, texts: list[str], top_k: int) -> list[tuple[int, float]]:
        if not texts:
            return []

        vecs = self._embed([query] + texts)
        if len(vecs) != len(texts) + 1:
            return []

        qv = vecs[0]
        scored: list[tuple[int, float]] = []
        for i, dv in enumerate(vecs[1:]):
            scored.append((i, cosine_similarity(qv, dv)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: max(1, top_k)]

    
    def _chat_rerank_topn(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.rerank_chat_client or not candidates:
            return candidates

        prompt = (
            "假设你是一个专业的相关性重排器。"
            "请根据 query 对候选文档按相关性打分，分数范围 0~1。"
            "只输出 JSON 数组，格式为："
            '[{"idx":0,"score":0.95},{"idx":1,"score":0.41}]'
        )

        payload = []
        for i, item in enumerate(candidates):
            payload.append(
                {
                    "idx": i,
                    "text": str(item.get("_rank_text", ""))[:1800],
                }
            )

        try:
            resp = self._with_retry(
                lambda: self.rerank_chat_client.chat.completions.create(
                    model=self.rerank_chat_model,
                    temperature=0.0,
                    max_tokens=600,
                    messages=[
                        {"role": "system", "content": prompt},
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "query": query,
                                    "candidates": payload,
                                },
                                ensure_ascii=False,
                            ),
                        },
                    ],
                )
            )
            raw = (resp.choices[0].message.content or "").strip()
            data = json.loads(raw)
            if not isinstance(data, list):
                return candidates
        except Exception:
            return candidates

        scored: list[tuple[int, float]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            idx = item.get("idx")
            if not isinstance(idx, int) or not (0 <= idx < len(candidates)):
                continue
            score = _safe_float(item.get("score", 0.0), 0.0)
            scored.append((idx, score))

        if not scored:
            return candidates

        scored.sort(key=lambda x: x[1], reverse=True)

        out: list[dict[str, Any]] = []
        used: set[int] = set()
        for idx, score in scored:
            row = dict(candidates[idx])
            row["_chat_rerank_score"] = score
            out.append(row)
            used.add(idx)

        for i, row in enumerate(candidates):
            if i not in used:
                out.append(row)

        return out


    def rerank_docs(self, question: str, docs: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not docs:
            return []

        def _doc_prior(doc: dict[str, Any]) -> float:
            source = str(doc.get("source", "")).strip().lower()
            text = str(doc.get("text", "")).strip()

            score = _safe_float(doc.get("rerank_score", doc.get("score", 0.0)), 0.0)

            if source.startswith(("local_image", "seed_local", "vision_recheck", "current_image")):
                score += 0.45
            elif source == "rationale" or source.startswith("rationale") or source == "reasoning_chains":
                score += 0.20
            elif source.startswith("lexical:"):
                score += 0.35
            elif source in {"graph_paths", "graph_meta"}:
                score += 0.25

            if text.startswith("搜索结果摘要:"):
                score -= 0.40

            return score

        if not self.enabled or self.embedding_client is None:
            rows = [dict(d) for d in docs]
            for row in rows:
                row["rerank_score"] = _doc_prior(row)
            rows.sort(key=lambda x: _safe_float(x.get("rerank_score", 0.0), 0.0), reverse=True)
            return rows[:top_k]

        texts = [
            f"{str(d.get('source', ''))}\n{str(d.get('text', ''))[:2000]}"
            for d in docs
        ]

        try:
            ranked = self._rank_texts_by_embedding(
                question,
                texts,
                top_k=max(1, len(docs)),
            )
            embed_scores = {idx: sim for idx, sim in ranked}
        except Exception:
            rows = [dict(d) for d in docs]
            for row in rows:
                row["rerank_score"] = _doc_prior(row)
            rows.sort(key=lambda x: _safe_float(x.get("rerank_score", 0.0), 0.0), reverse=True)
            return rows[:top_k]

        out: list[dict[str, Any]] = []
        for idx, doc in enumerate(docs):
            item = dict(doc)
            item["_rank_text"] = texts[idx]
            item["_embed_score"] = _safe_float(embed_scores.get(idx, 0.0), 0.0)
            item["rerank_score"] = 0.60 * item["_embed_score"] + 0.40 * _doc_prior(item)
            out.append(item)

        out.sort(key=lambda x: _safe_float(x.get("rerank_score", 0.0), 0.0), reverse=True)

        if self.enable_chat_rerank and self.rerank_chat_client is not None:
            rerank_top_n = min(len(out), max(1, int(getattr(settings, "rerank_chat_top_n", 3))))
            head = self._chat_rerank_topn(question, out[:rerank_top_n])
            out = head + out[rerank_top_n:]

        for item in out:
            item.pop("_rank_text", None)

        return out[:top_k]

    def rerank_rows(self, question: str, rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not rows:
            return []

        if not self.enabled or self.embedding_client is None:
            rows = list(rows)
            rows.sort(key=lambda x: _safe_float(x.get("rerank_score", x.get("score", 0.0)), 0.0), reverse=True)
            return rows[:top_k]

        texts = []
        for r in rows:
            text = " ".join(
                [
                    str(r.get("doc_name", "")),
                    str(r.get("text", "")),
                    str(r.get("subject", "")),
                    str(r.get("predicate", "")),
                    str(r.get("object", "")),
                ]
            ).strip()
            texts.append(text[:2400])

        try:
            ranked = self._rank_texts_by_embedding(question, texts, top_k=min(len(rows), max(top_k, 1)))
        except Exception:
            rows = list(rows)
            rows.sort(key=lambda x: _safe_float(x.get("rerank_score", x.get("score", 0.0)), 0.0), reverse=True)
            return rows[:top_k]

        out: list[dict[str, Any]] = []
        for idx, sim in ranked:
            item = dict(rows[idx])
            base = _safe_float(item.get("rerank_score", item.get("score", 0.0)), 0.0)
            item["_rank_text"] = texts[idx]
            item["_embed_score"] = sim
            item["rerank_score"] = 0.65 * base + 0.35 * sim
            out.append(item)

        if self.enable_chat_rerank and self.rerank_chat_client is not None:
            rerank_top_n = min(len(out), max(1, int(getattr(settings, "rerank_chat_top_n", 3))))
            head = self._chat_rerank_topn(question, out[:rerank_top_n])
            out = head + out[rerank_top_n:]

        for item in out:
            item.pop("_rank_text", None)

        out.sort(key=lambda x: _safe_float(x.get("rerank_score", 0.0), 0.0), reverse=True)
        return out[:top_k]