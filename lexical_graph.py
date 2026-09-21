from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

TOKEN_RE = re.compile(r"[A-Za-z0-9_\-\u4e00-\u9fff]+")
STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "to",
    "of",
    "in",
    "and",
    "for",
    "on",
    "with",
    "by",
    "what",
    "which",
    "how",
    "why",
    "when",
    "where",
    "是",
    "的",
    "了",
    "和",
    "在",
    "与",
    "及",
    "对",
    "中",
    "将",
    "其",
    "一个",
}


@dataclass
class ChunkNode:
    chunk_id: str
    source: str
    text: str
    tokens: list[str]
    entities: list[str]


@dataclass
class RetrievalResult:
    chunk_id: str
    source: str
    text: str
    score: float
    reasons: list[str] = field(default_factory=list)


class LexicalGraphIndex:
    """In-memory lexical graph inspired by awslabs lexical-graph hierarchy.

    Hierarchy we keep lightweight:
      source -> chunk -> entity
    plus inverted indexes for lexical retrieval and entity traversal.
    """

    def __init__(self) -> None:
        self.chunks: dict[str, ChunkNode] = {}
        self.source_to_chunks: dict[str, list[str]] = defaultdict(list)
        self.entity_to_chunks: dict[str, set[str]] = defaultdict(set)
        self.token_to_chunks: dict[str, set[str]] = defaultdict(set)

    def add_document(self, source: str, text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> None:
        clean = " ".join(text.split())
        if not clean:
            return

        chunk_size = max(100, int(chunk_size))
        chunk_overlap = max(0, min(int(chunk_overlap), chunk_size - 1))
        step = max(1, chunk_size - chunk_overlap)

        chunk_idx = 0
        for i in range(0, len(clean), step):
            chunk = clean[i : i + chunk_size]
            if not chunk:
                continue

            chunk_id = f"{source}#c{chunk_idx}"
            tokens = _tokenize(chunk)
            entities = _extract_entities(chunk)

            node = ChunkNode(
                chunk_id=chunk_id,
                source=source,
                text=chunk,
                tokens=tokens,
                entities=entities,
            )
            self.chunks[chunk_id] = node
            self.source_to_chunks[source].append(chunk_id)

            for t in set(tokens):
                self.token_to_chunks[t].add(chunk_id)
            for e in set(entities):
                self.entity_to_chunks[e].add(chunk_id)

            chunk_idx += 1

            if i + chunk_size >= len(clean):
                break

    def has_data(self) -> bool:
        return bool(self.chunks)


class LexicalGraphRetriever:
    """
    基于遍历的搜索近似方法。

    步骤 1：通过词元重叠确定词汇种子。

    步骤 2：扩展实体以遍历相关词块。

    步骤 3：结合词汇支持和遍历支持进行重新排序。
    """

    def __init__(self, index: LexicalGraphIndex) -> None:
        self.index = index

    def retrieve(self, question: str, top_k: int = 5) -> list[RetrievalResult]:
        q_tokens = _tokenize(question)
        if not q_tokens or not self.index.has_data():
            return []

        seed_scores: dict[str, float] = defaultdict(float)
        for t in set(q_tokens):
            chunks = self.index.token_to_chunks.get(t, set())
            idf = math.log((1 + len(self.index.chunks)) / (1 + len(chunks))) + 1.0
            for cid in chunks:
                seed_scores[cid] += idf

        seed_ids = sorted(seed_scores, key=lambda c: seed_scores[c], reverse=True)[: max(top_k, 3)]

        expanded: dict[str, float] = defaultdict(float)
        for cid in seed_ids:
            expanded[cid] += seed_scores[cid]
            entities = self.index.chunks[cid].entities
            for ent in entities:
                for neighbor_cid in self.index.entity_to_chunks.get(ent, set()):
                    if neighbor_cid == cid:
                        continue
                    expanded[neighbor_cid] += 0.35 * seed_scores[cid]

        results: list[RetrievalResult] = []
        for cid, sc in expanded.items():
            chunk = self.index.chunks[cid]
            overlap = len(set(q_tokens) & set(chunk.tokens))
            entity_overlap = len(set(_extract_entities(question)) & set(chunk.entities))
            final_score = sc + 0.2 * overlap + 0.3 * entity_overlap
            reasons = []
            if cid in seed_ids:
                reasons.append("lexical-seed")
            if entity_overlap > 0:
                reasons.append("entity-traversal")
            results.append(
                RetrievalResult(
                    chunk_id=cid,
                    source=chunk.source,
                    text=chunk.text,
                    score=final_score,
                    reasons=reasons,
                )
            )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]


def _tokenize(text: str) -> list[str]:
    tokens = [x.lower() for x in TOKEN_RE.findall(text)]
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def _extract_entities(text: str) -> list[str]:
    raw = TOKEN_RE.findall(text)
    freq = Counter(
        tok.lower()
        for tok in raw
        if len(tok) >= 3 and tok.lower() not in STOPWORDS
    )

    weak_words = {
        "look", "looks", "like", "very", "same", "similar",
        "this", "that", "these", "those", "there", "here",
        "using", "used", "make", "made", "show", "shown",
    }

    generic_words = {
        "standing", "sitting", "holding", "looking", "wearing",
        "showing", "pictured", "located", "appears",
        "scene", "image", "picture", "photo",
        "object", "objects", "person", "people",
        "place", "location", "area",
        "animal", "vehicle", "building", "street", "road",
        "question", "answer", "option", "choice",
    }

    ents: list[str] = []
    seen: set[str] = set()

    for tok in raw:
        if len(tok) < 2:
            continue

        low = tok.lower()
        keep = False
        val = tok

        if re.search(r"[\u4e00-\u9fff]", tok):
            keep = True
            val = tok
        elif tok[:1].isupper() and any(c.islower() for c in tok[1:]):
            keep = True
            val = tok
        elif tok.isupper() and len(tok) <= 8:
            keep = True
            val = tok
        elif (
            tok.isalpha()
            and len(low) >= 5
            and low not in STOPWORDS
            and low not in weak_words
            and low not in generic_words
            and freq.get(low, 0) >= 2
            and not low.endswith("ing")
            and not low.endswith("ed")
        ):
            keep = True
            val = low

        if keep and val not in seen:
            seen.add(val)
            ents.append(val)

    return ents


def build_graph_from_corpus(corpus: Iterable[tuple[str, str]]) -> LexicalGraphIndex:
    index = LexicalGraphIndex()
    for source, text in corpus:
        index.add_document(source=source, text=text)
    return index
