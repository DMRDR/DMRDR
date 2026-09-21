from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from embeddings import EmbeddingReranker
from config import settings


def _uniq(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in seq:
        s = str(x).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _is_noise_entity(v: str) -> bool:
    s = v.strip()
    if not s:
        return True
    if len(s) <= 1:
        return True
    if re.fullmatch(r"[\W_]+", s):
        return True
    return False


def _clip_float(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _strip_inline_data_uris(text: str) -> str:
    text = str(text or "")

    text = re.sub(r"\[\]\(data:image[^)]*\)", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+",
        "[STRIPPED_INLINE_IMAGE]",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


@dataclass
class RerankerConfig:
    hop_l: int
    top_k: int


@dataclass
class StrategyResult:
    strategy: str
    entities: list[str]
    paths: list[dict[str, Any]]
    dsl: dict[str, Any]
    score: float


class CypherDSLCompiler:
    """Compile schema-constrained DSL JSON to OpenCypher."""

    def compile(self, dsl: dict[str, Any]) -> str:
        match_parts: list[str] = []

        for node in dsl.get("match", []):
            alias = str(node.get("alias", "n")).strip() or "n"
            ntype = str(node.get("node_type", "")).strip()
            cons = node.get("constraints", {}) or {}
            where_props = []
            for k, v in cons.items():
                if isinstance(v, str):
                    escaped = v.replace(chr(39), chr(92) + chr(39))
                    where_props.append(f"{alias}.{k} = '{escaped}'")
                else:
                    where_props.append(f"{alias}.{k} = {json.dumps(v, ensure_ascii=False)}")
            label = f":{ntype}" if ntype else ""
            match_parts.append(f"({alias}{label})")
            node["__where_props"] = where_props

        edge_parts: list[str] = []
        for e in dsl.get("edges", []):
            frm = str(e.get("from", "")).strip()
            to = str(e.get("to", "")).strip()
            rel = str(e.get("rel_type", "")).strip()
            if not frm or not to:
                continue
            edge_parts.append(f"({frm})-[:{rel}]->({to})" if rel else f"({frm})--({to})")

        query_parts: list[str] = []
        if edge_parts:
            query_parts.append("MATCH " + ", ".join(edge_parts))
        elif match_parts:
            query_parts.append("MATCH " + ", ".join(match_parts))
        else:
            query_parts.append("MATCH (n)")

        where_clauses: list[str] = []
        for node in dsl.get("match", []):
            where_clauses.extend(node.get("__where_props", []))
        for cond in dsl.get("where", []) or []:
            if isinstance(cond, str) and cond.strip():
                where_clauses.append(cond.strip())
        if where_clauses:
            query_parts.append("WHERE " + " AND ".join(where_clauses))

        returns = dsl.get("return", ["n"])
        if not isinstance(returns, list) or not returns:
            returns = ["n"]
        query_parts.append("RETURN " + ", ".join(str(x) for x in returns))

        sort = dsl.get("sort")
        if isinstance(sort, str) and sort.strip():
            query_parts.append("ORDER BY " + sort.strip())

        limit = dsl.get("limit")
        if isinstance(limit, int) and limit > 0:
            query_parts.append(f"LIMIT {limit}")

        return "\n".join(query_parts)


class BYOKGValidator:
    def __init__(self, dry_run: Callable[[str], bool] | None = None) -> None:
        self.dry_run = dry_run

    def validate(self, artifacts: dict[str, Any], schema: dict[str, Any]) -> tuple[bool, list[str]]:
        errors: list[str] = []

        entities = [str(x) for x in artifacts.get("entities", [])]
        entities = [x for x in _uniq(entities) if not _is_noise_entity(x)]
        artifacts["entities"] = entities
        if not entities:
            errors.append("entities_empty")

        allowed_rels = {str(x) for x in schema.get("relation_types", [])} if isinstance(schema, dict) else set()
        paths = artifacts.get("paths", [])
        if isinstance(paths, list):
            for p in paths:
                if not isinstance(p, dict):
                    continue
                rel = str(p.get("relation", "")).strip()
                if rel and allowed_rels and rel not in allowed_rels:
                    errors.append(f"invalid_relation:{rel}")

        cypher = str(artifacts.get("opencypher", "")).strip()
        if cypher:
            c_up = cypher.upper()
            if "MATCH" not in c_up or "RETURN" not in c_up:
                errors.append("cypher_missing_match_or_return")
            if self.dry_run is not None:
                try:
                    if not self.dry_run(cypher + "\nLIMIT 1"):
                        errors.append("cypher_dry_run_failed")
                except Exception:
                    errors.append("cypher_dry_run_failed")
        else:
            errors.append("cypher_empty")

        return (len(errors) == 0), errors


def _norm_entity_key(text: str) -> str:
    s = re.sub(r"[_\-\s]+", " ", str(text or "").strip().lower())
    return s.strip()

class BYOKGRAGProvider:
    """Graph Evidence Provider adapted from BYOKG-RAG multi-strategy refinement.

    支持 7 种 retrieval_backend:
    - graph:       只走图谱 / Cypher 逻辑
    - qianfan:     只走百度千帆知识库检索
    - dbpedia:     只走 DBpedia 在线 KG 检索
    - auto:        优先图谱，其次千帆，最后 DBpedia
    - hybrid:      图谱 + 千帆混合返回
    - proxy_graph: 仅走本地代理图谱推理
    - light_rag:   proxy_graph 的别名
    """

    def __init__(
        self,
        linker: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        strategy_linker: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cypher_dry_run: Callable[[str], bool] | None = None,
        cypher_executor: Callable[[str], list[dict[str, Any]]] | None = None,
        max_refine_rounds: int = 2,
        max_query_results: int = 20,
        retrieval_backend: str = "auto",
        qianfan_api_key: str | None = None,
        qianfan_kb_ids: list[str] | str | None = None,
        qianfan_search_url: str = "https://qianfan.baidubce.com/v2/knowledgebases/search",
        qianfan_timeout: int = 20,
        qianfan_top_k: int = 5,
        proxy_top_k: int = 8,
        proxy_max_paths: int = 12,
        dbpedia_endpoint: str = "https://dbpedia.org/sparql",
        dbpedia_timeout: int = 12,
        dbpedia_top_k: int = 6,
    ) -> None:
        self.linker = linker
        self.strategy_linker = strategy_linker
        self.max_refine_rounds = max_refine_rounds
        self.max_query_results = max_query_results
        self.validator = BYOKGValidator(dry_run=cypher_dry_run)
        self.cypher_executor = cypher_executor
        self.compiler = CypherDSLCompiler()

        self.retrieval_backend = self._normalize_retrieval_backend_name(retrieval_backend)        
        self.qianfan_api_key = (qianfan_api_key or os.getenv("QIANFAN_API_KEY", "")).strip()
        self.qianfan_kb_ids = self._normalize_kb_ids(qianfan_kb_ids)
        self.qianfan_search_url = qianfan_search_url
        self.qianfan_timeout = qianfan_timeout
        self.qianfan_top_k = qianfan_top_k

        self.proxy_top_k = proxy_top_k
        self.proxy_max_paths = proxy_max_paths
        self.proxy_graph_min_rerank_score = float(os.getenv("PROXY_GRAPH_MIN_RERANK_SCORE", "0.02") or 0.02)
        self.qianfan_min_rerank_score = float(os.getenv("QIANFAN_MIN_RERANK_SCORE", "0.0") or 0.0)
        self._last_external_errors: list[dict[str, Any]] = []
        self._last_external_stats: dict[str, Any] = {}

        self.dbpedia_endpoint = str(dbpedia_endpoint).strip()
        self.dbpedia_timeout = int(dbpedia_timeout)
        self.dbpedia_top_k = int(dbpedia_top_k)
    
    
    def _generic_dbpedia_terms(self) -> set[str]:
        raw = getattr(settings, "dbpedia_generic_entity_blocklist", []) or []
        out = {_norm_entity_key(x) for x in raw if _norm_entity_key(x)}
        if not out:
            out = {
                "person", "people", "human", "humans",
                "place", "location", "area",
                "thing", "object", "entity",
                "man", "woman", "child",
                "city", "country", "animal",
                "this", "that", "these", "those",
                "what", "which", "who", "where", "when", "why", "how",
                "is", "are", "was", "were", "does", "do", "did", "has", "have",
                "type", "kind", "image", "picture", "photo",
                "scene", "setting", "view",
                "vehicle", "car", "building", "street",
                "urban", "rural", "residential", "commercial", "private", "public",
                "domestic", "wild", "aquatic", "stuffed",
                "road", "roadside", "category", "animal type", "habitat", "breed", "species",
            }
        return out

    
    @staticmethod
    def _normalize_retrieval_backend_name(name: str) -> str:
        raw = str(name or "").strip().lower()
        alias_map = {
            "default": "auto",
            "kg": "auto",
            "lexical_graph": "light_rag",
        }
        return alias_map.get(raw, raw or "auto")
    
    
    def _is_generic_dbpedia_term(self, text: str) -> bool:
        s = _norm_entity_key(text)
        if not s:
            return True

        generic = self._generic_dbpedia_terms()
        if s in generic:
            return True

        singular = s[:-1] if s.endswith("s") else s
        if singular in generic:
            return True

        return False
    
    
    def _is_meta_dbpedia_predicate(self, pred: str) -> bool:
        p = str(pred or "").strip().lower()
        if not p:
            return True

        tail = p.split("/")[-1].split("#")[-1]
        blocked = {
            "wikipagewikilink",
            "wikipagedisambiguates",
            "wikipageredirects",
            "thumbnail",
            "description",
            "name",
            "abstract",
            "seealso",
            "sameas",
            "subject",
            "type",
        }
        return tail in blocked
    
    
    def _has_strong_dbpedia_anchor_terms(self, terms: list[str]) -> bool:
        for term in terms:
            s = str(term or "").strip()
            if not s:
                continue
            if self._is_generic_dbpedia_term(s):
                continue

            norm = _norm_entity_key(s)
            if len(norm) < 4:
                continue

            # 更倾向于专名、多词短语、较稳定名词；压制 standing / holding 这类普通词
            if re.search(r"[A-Z]", s):
                return True
            if len(s.split()) >= 2:
                return True
            if len(norm) >= 6 and not norm.endswith("ing") and not norm.endswith("ed"):
                return True

        return False
    
    

    def _has_valid_dbpedia_anchor(self, rows: list[dict[str, Any]]) -> bool:
        for row in rows:
            if not isinstance(row, dict):
                continue
            subj = str(row.get("subject", "")).strip()
            matched = str(row.get("matched_entity", "")).strip()
            obj = str(row.get("object", "")).strip()

            for x in [subj, matched, obj]:
                if not x:
                    continue
                if self._is_generic_dbpedia_term(x):
                    continue
                if len(_norm_entity_key(x)) < 3:
                    continue
                return True
        return False

    def _record_external_error(self, backend: str, status: str, detail: str = "") -> None:
        errors = getattr(self, "_last_external_errors", None)
        if not isinstance(errors, list):
            errors = []
            self._last_external_errors = errors
        errors.append(
            {
                "backend": str(backend or "external_kg"),
                "status": str(status or "error"),
                "detail": str(detail or "")[:300],
            }
        )

    def _consume_external_errors(self) -> list[dict[str, Any]]:
        errors = getattr(self, "_last_external_errors", [])
        self._last_external_errors = []
        return [e for e in errors if isinstance(e, dict)]

    def _set_external_stat(self, backend: str, key: str, value: Any) -> None:
        stats = getattr(self, "_last_external_stats", None)
        if not isinstance(stats, dict):
            stats = {}
            self._last_external_stats = stats
        bucket = stats.setdefault(str(backend or "external_kg"), {})
        if isinstance(bucket, dict):
            bucket[str(key)] = value

    def _consume_external_stats(self) -> dict[str, Any]:
        stats = getattr(self, "_last_external_stats", {})
        self._last_external_stats = {}
        return stats if isinstance(stats, dict) else {}
    
    
    def _semantic_rerank_external_rows(
        self,
        question: str,
        rows: list[dict[str, Any]],
        keep_k: int,
    ) -> list[dict[str, Any]]:
        if not rows:
            return []

        if not hasattr(self, "_semantic_reranker"):
            try:
                self._semantic_reranker = EmbeddingReranker()
            except Exception:
                self._semantic_reranker = None

        reranker = getattr(self, "_semantic_reranker", None)
        if reranker is None:
            rows = list(rows)
            rows.sort(key=lambda x: float(x.get("rerank_score", x.get("score", 0.0)) or 0.0), reverse=True)
            return rows[:keep_k]

        try:
            reranked = reranker.rerank_rows(question, rows, top_k=max(keep_k, 1))
            reranked.sort(key=lambda x: float(x.get("rerank_score", x.get("score", 0.0)) or 0.0), reverse=True)
            return reranked[:keep_k]
        except Exception:
            rows = list(rows)
            rows.sort(key=lambda x: float(x.get("rerank_score", x.get("score", 0.0)) or 0.0), reverse=True)
            return rows[:keep_k]

    def _dbpedia_enabled(self) -> bool:
        return bool(self.dbpedia_endpoint)

    def _dbpedia_query_json(self, sparql: str) -> dict[str, Any]:
        if not self._dbpedia_enabled():
            self._record_external_error("dbpedia", "disabled", "empty_endpoint")
            return {}

        url = (
            f"{self.dbpedia_endpoint}"
            f"?query={quote(sparql)}"
            f"&format=application%2Fsparql-results%2Bjson"
        )
        req = Request(
            url,
            method="GET",
            headers={
                "Accept": "application/sparql-results+json",
                "User-Agent": "DeepResearch-DBpedia/1.0",
            },
        )

        try:
            with urlopen(req, timeout=self.dbpedia_timeout) as resp:
                payload = resp.read().decode("utf-8", errors="ignore")
            data = json.loads(payload)
            return data if isinstance(data, dict) else {}
        except HTTPError as exc:
            self._record_external_error("dbpedia", "http", str(exc))
            return {}
        except URLError as exc:
            self._record_external_error("dbpedia", "timeout" if isinstance(getattr(exc, "reason", None), TimeoutError) else "url", str(exc))
            return {}
        except TimeoutError as exc:
            self._record_external_error("dbpedia", "timeout", str(exc))
            return {}
        except json.JSONDecodeError as exc:
            self._record_external_error("dbpedia", "json", str(exc))
            return {}
        
    def _should_skip_dbpedia_for_question(
        self,
        question: str,
        query_variants: list[str] | None = None,
    ) -> bool:
        q = str(question or "").strip().lower()
        if not q:
            return True

        focus_terms = list(self._question_focus_terms(q))
        focus_norm = [_norm_entity_key(t) for t in focus_terms if _norm_entity_key(t)]

        optionish_terms = {
            "urban", "rural", "residential", "commercial", "private", "public",
            "scene", "setting", "area", "place", "location", "street", "building",
            "domestic", "wild", "aquatic", "stuffed",
            "road", "roadside", "category", "animal type", "habitat", "breed", "species",
        }

        has_choices = ("选项:" in q) or bool(re.search(r"(?:^|\n)\s*[a-h]\.\s+", q))

        if has_choices:
            non_option_focus = [
                t for t in focus_norm
                if t not in optionish_terms and not self._is_generic_dbpedia_term(t)
            ]
            if len(non_option_focus) <= 1:
                return True

            if all(t in optionish_terms or self._is_generic_dbpedia_term(t) for t in focus_norm):
                return True

        if len(focus_terms) >= 2:
            return False

        variants = [str(x).strip().lower() for x in (query_variants or []) if str(x).strip()]
        compact_variants = [v for v in variants if len(v.split()) <= 2]

        if len(focus_terms) <= 1 and compact_variants:
            return True

        return False

    def _dbpedia_terms_from_text(self, text: str, max_terms: int = 4) -> list[str]:
        stop = {
            "what", "which", "when", "where", "why", "how", "who",
            "the", "and", "for", "with", "from", "into", "than",
            "image", "picture", "photo", "question", "answer",
            "type", "kind", "among", "listed", "following", "option", "options",
            "best", "match", "likely", "based", "shown", "showing",
            "questiondriven", "summary", "candidate",
        }

        out: list[str] = []
        seen: set[str] = set()

        def _push(x: str) -> None:
            s = re.sub(r"\s+", " ", str(x or "")).strip(" \t\r\n,.;:!?()[]{}\"'")
            low = s.lower().replace("-", "").replace("_", "").replace(" ", "")
            if len(s) < 3 or low in stop or low in seen:
                return
            seen.add(low)
            out.append(s)

        text = re.sub(r"\s+", " ", str(text or "")).strip()

        for m in re.findall(r"\b(?:[A-Z][A-Za-z0-9_-]*)(?:\s+[A-Z][A-Za-z0-9_-]*){0,3}\b", text):
            _push(m)

        for tok in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text):
            if tok.lower() in stop:
                continue
            _push(tok)

        return out[:max_terms]

    def _dbpedia_relation_hints(self, question: str, schema: dict[str, Any]) -> list[str]:
        stop = {
            "what", "which", "when", "where", "why", "how", "who",
            "the", "and", "for", "with", "from", "into", "than",
        }

        hints: list[str] = []
        seen: set[str] = set()

        def _push(x: str) -> None:
            s = str(x or "").strip().lower()
            if len(s) < 3 or s in stop or s in seen:
                return
            seen.add(s)
            hints.append(s)

        for tok in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", str(question or "").lower()):
            _push(tok)

        if isinstance(schema, dict):
            rels: list[str] = []
            for key in ["focus_relation_types", "relation_types"]:
                value = schema.get(key, [])
                if isinstance(value, list):
                    rels.extend([str(x) for x in value])

            for rel in rels:
                for tok in re.split(r"[_\W]+", rel.lower()):
                    _push(tok)

        return hints[:20]

    def _dbpedia_resource_candidates(self, term: str) -> list[dict[str, str]]:
        term = re.sub(r"\s+", " ", str(term or "")).strip(" \t\r\n,.;:!?()[]{}\"'")
        if not term:
            return []
        if self._is_generic_dbpedia_term(term):
            return []

        candidates: list[dict[str, str]] = []
        seen: set[str] = set()

        def _push(uri: str, label: str) -> None:
            u = str(uri or "").strip()
            if not u or u in seen:
                return
            seen.add(u)
            candidates.append({"uri": u, "label": str(label or "").strip()})

        parts = [p for p in re.split(r"\s+", term) if p]
        guessed = "_".join([p[:1].upper() + p[1:] if p[:1].islower() else p for p in parts])
        _push(f"http://dbpedia.org/resource/{guessed}", term)

        escaped = term.replace("\\", "\\\\").replace("'", "\\'")
        sparql = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?s ?label WHERE {{
          ?s rdfs:label ?label .
          FILTER (lang(?label) = 'en')
          FILTER (STRSTARTS(STR(?s), 'http://dbpedia.org/resource/'))
          FILTER (
            LCASE(STR(?label)) = LCASE('{escaped}')
            || CONTAINS(LCASE(STR(?label)), LCASE('{escaped}'))
          )
        }}
        LIMIT 5
        """
        data = self._dbpedia_query_json(sparql)
        bindings = data.get("results", {}).get("bindings", []) if isinstance(data, dict) else []
        if isinstance(bindings, list):
            for b in bindings:
                if not isinstance(b, dict):
                    continue
                uri = str(b.get("s", {}).get("value", "")).strip()
                label = str(b.get("label", {}).get("value", "")).strip()
                if uri:
                    _push(uri, label or term)

        return candidates[:5]

    def _probe_dbpedia_resource(
        self,
        resource_uri: str,
        relation_hints: list[str],
        limit: int = 60,
    ) -> list[dict[str, Any]]:
        if not resource_uri:
            return []

        sparql = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?p ?o ?oLabel WHERE {{
          <{resource_uri}> ?p ?o .
          OPTIONAL {{ ?o rdfs:label ?oLabel FILTER (lang(?oLabel) = 'en') }}
          FILTER (
            STRSTARTS(STR(?p), 'http://dbpedia.org/ontology/')
            || STRSTARTS(STR(?p), 'http://dbpedia.org/property/')
          )
        }}
        LIMIT {int(limit)}
        """
        data = self._dbpedia_query_json(sparql)
        bindings = data.get("results", {}).get("bindings", []) if isinstance(data, dict) else []
        if not isinstance(bindings, list):
            return []

        subject_label = resource_uri.rsplit("/", 1)[-1].replace("_", " ")
        if self._is_generic_dbpedia_term(subject_label):
            return []
        rows: list[dict[str, Any]] = []

        for b in bindings:
            if not isinstance(b, dict):
                continue

            p_uri = str(b.get("p", {}).get("value", "")).strip()
            o_uri = str(b.get("o", {}).get("value", "")).strip()
            o_label = str(b.get("oLabel", {}).get("value", "")).strip()

            if not p_uri or not o_uri:
                continue

            pred = p_uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
            obj = o_label or o_uri.rsplit("/", 1)[-1].replace("_", " ")

            pred_low = pred.lower()
            obj_low = obj.lower()

            overlap = sum(1 for h in relation_hints if h in pred_low or h in obj_low)
            if relation_hints and overlap == 0:
                continue

            score = 0.45 + 0.15 * min(overlap, 3)
            rows.append(
                {
                    "source": "dbpedia",
                    "doc_name": resource_uri,
                    "text": f"{subject_label} -[{pred}]-> {obj}",
                    "subject": subject_label,
                    "predicate": pred,
                    "object": obj,
                    "resource_uri": resource_uri,
                    "predicate_uri": p_uri,
                    "object_uri": o_uri,
                    "rerank_score": _clip_float(score),
                }
            )

        rows.sort(key=lambda x: float(x.get("rerank_score", 0.0)), reverse=True)
        return rows[: max(self.dbpedia_top_k, 5)]

    def _search_dbpedia_multi(
        self,
        question: str,
        schema: dict[str, Any],
        query_variants: list[str] | None = None,
        documents: list[dict[str, str]] | None = None,
    ) -> list[dict[str, Any]]:

        if self._should_skip_dbpedia_for_question(
            question=question,
            query_variants=query_variants,
        ):
            self._record_external_error("dbpedia", "skipped", "question_not_suitable")
            return []

        terms: list[str] = []
        seen_terms: set[str] = set()

        def _push_term(x: str) -> None:
            s = str(x or "").strip()
            low = s.lower()
            if not s or low in seen_terms:
                return
            if self._is_generic_dbpedia_term(s):
                return
            seen_terms.add(low)
            terms.append(s)

        linked = self._invoke_linker({"question": question, "schema": schema})
        for ent in linked.get("entities", []):
            _push_term(str(ent))

        for text in [question, *(query_variants or [])]:
            for term in self._dbpedia_terms_from_text(text, max_terms=4):
                _push_term(term)

        # 不再从 documents[].source 提取 term，避免 local_image/current_image/rationale 等元数据把 DBpedia 带偏

        if not self._has_strong_dbpedia_anchor_terms(terms):
            self._record_external_error("dbpedia", "no_hits", "no_strong_anchor_terms")
            return []

        relation_hints = self._dbpedia_relation_hints(question, schema)
        focus_terms = self._question_focus_terms(question)

        rows_all: list[dict[str, Any]] = []
        seen_rows: set[tuple[str, str, str]] = set()

        for term in terms[:6]:
            for cand in self._dbpedia_resource_candidates(term):
                for row in self._probe_dbpedia_resource(cand["uri"], relation_hints):
                    key = (
                        str(row.get("subject", "")).strip(),
                        str(row.get("predicate", "")).strip(),
                        str(row.get("object", "")).strip(),
                    )
                    if key in seen_rows:
                        continue
                    seen_rows.add(key)
                    row["matched_entity"] = term

                    merged = " ".join(
                        [
                            str(row.get("subject", "")),
                            str(row.get("predicate", "")),
                            str(row.get("object", "")),
                            str(term),
                        ]
                    ).lower()

                    overlap = sum(1 for t in focus_terms if t in merged)
                    if focus_terms and overlap == 0:
                        continue

                    rows_all.append(row)

        rows_all.sort(key=lambda x: float(x.get("rerank_score", 0.0)), reverse=True)
        rows = rows_all[: max(self.dbpedia_top_k, 5)]
        if not rows:
            self._record_external_error("dbpedia", "no_hits", "empty_after_filter")
        return rows

    def _build_dbpedia_result(self, question: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        rows = self._semantic_rerank_external_rows(
            question=question,
            rows=rows,
            keep_k=max(self.dbpedia_top_k, 5),
        )
        if not rows:
            return self._empty_result("dbpedia_no_hits", ["dbpedia_no_hits"])

        if not self._has_valid_dbpedia_anchor(rows):
            return self._empty_result("dbpedia_no_valid_anchor", ["dbpedia_no_valid_anchor"])

        paths: list[dict[str, Any]] = []
        query_results: list[dict[str, Any]] = []
        entities: list[str] = []

        for row in rows[: max(self.dbpedia_top_k, 5)]:
            subj = str(row.get("subject", "")).strip()
            pred = str(row.get("predicate", "")).strip()
            obj = str(row.get("object", "")).strip()
            src = str(row.get("resource_uri", "")).strip() or "dbpedia"

            if self._is_generic_dbpedia_term(subj):
                continue

            if self._is_meta_dbpedia_predicate(pred):
                continue

            matched_entity = str(row.get("matched_entity", "")).strip()
            
            if matched_entity and self._is_generic_dbpedia_term(matched_entity):
                continue
            
            if subj:
                entities.append(subj)
            if obj:
                entities.append(obj)

            if subj and pred and obj:
                paths.append(
                    {
                        "subject": subj,
                        "relation": pred,
                        "object": obj,
                        "source": src,
                    }
                )

            query_results.append(
                {
                    "source": "dbpedia",
                    "doc_name": src,
                    "text": f"{subj} -[{pred}]-> {obj}",
                    "rerank_score": float(row.get("rerank_score", 0.0)),
                    "meta": {
                        "predicate_uri": str(row.get("predicate_uri", "")).strip(),
                        "object_uri": str(row.get("object_uri", "")).strip(),
                        "matched_entity": str(row.get("matched_entity", "")).strip(),
                    },
                }
            )

        confidence = _clip_float(_mean([float(r.get("rerank_score", 0.0)) for r in rows[:3]]))
        coverage = _clip_float(len(query_results) / max(self.dbpedia_top_k, 1))

        return {
            "graph_evidence": {
                "triples": paths,
                "paths": paths,
                "query_results": query_results,
                "image_evidence": [],
            },
            "linking_artifacts": {
                "entities": _uniq(entities)[:16],
                "paths": paths,
                "opencypher": "",
                "dsl": {},
                "reranker": {"L": 0, "k": len(query_results)},
                "strategy_ranking": [
                    {
                        "strategy": "dbpedia_sparql",
                        "score": confidence,
                        "entities": len(_uniq(entities)),
                        "paths": len(paths),
                    }
                ],
                "selected_strategy": "dbpedia_sparql",
                "refinement_history": [{"round": 1, "ok": True, "errors": []}],
                "external_backend": "dbpedia",
            },
            "confidence": confidence,
            "coverage": coverage,
        }

    def _normalize_kb_ids(self, kb_ids: list[str] | str | None) -> list[str]:
        if kb_ids is None:
            env_ids = os.getenv("QIANFAN_KB_IDS", "")
            return [x.strip() for x in env_ids.split(",") if x.strip()]
        if isinstance(kb_ids, str):
            return [x.strip() for x in kb_ids.split(",") if x.strip()]
        return [str(x).strip() for x in kb_ids if str(x).strip()]

    def _dynamic_reranker(self, question: str, schema: dict[str, Any]) -> RerankerConfig:
        rels = schema.get("relation_types", []) if isinstance(schema, dict) else []
        dense = len(rels) > 80
        multi_hop = any(
            k in question.lower()
            for k in ["then", "after", "before", "compare", "path", "multi-hop", "并且", "再", "然后"]
        )

        l = 1 if dense else 2
        k = 10 if dense else 30
        if multi_hop:
            k = min(k + 20, 80)
        return RerankerConfig(hop_l=l, top_k=k)

    def _default_linker(self, payload: dict[str, Any]) -> dict[str, Any]:
        question = str(payload.get("question", ""))
        entities = self._dbpedia_terms_from_text(question, max_terms=4)

        dsl = {
            "match": [{"node_type": "Entity", "alias": "e", "constraints": {"name": entities[0] if entities else ""}}],
            "return": ["e"],
            "limit": 20,
        }
        return {
            "entities": entities,
            "paths": [],
            "dsl": dsl,
        }
    
    def _question_focus_terms(self, question: str) -> set[str]:
        stop = {
            "what", "which", "when", "where", "why", "how", "who",
            "the", "and", "for", "with", "from", "into", "than",
            "image", "picture", "photo", "question", "answer",
            "type", "kind", "among", "listed", "following", "option", "options",
            "best", "match", "likely", "based", "shown", "showing",
        }
        toks = re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", str(question or "").lower())
        return {t for t in toks if t not in stop}

    def _invoke_linker(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.linker is not None:
            out = self.linker(payload)
            return out if isinstance(out, dict) else {}
        return self._default_linker(payload)

    def _invoke_strategy_linker(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.strategy_linker is not None:
            out = self.strategy_linker(payload)
            return out if isinstance(out, dict) else {}
        return self._invoke_linker(payload)

    def _strategy_payloads(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        schema = payload.get("schema", {}) if isinstance(payload.get("schema"), dict) else {}
        return [
            {**payload, "strategy": "entity_path", "hints": ["entity-first", "path-expand"]},
            {
                **payload,
                "strategy": "relation_path",
                "hints": ["relation-first", "schema-filter"],
                "focus_relations": schema.get("relation_types", [])[:20],
            },
            {**payload, "strategy": "query_shape", "hints": ["query-structure", "answer-type"]},
        ]

    def _score_strategy_result(self, entities: list[str], paths: list[dict[str, Any]], schema: dict[str, Any]) -> float:
        allowed_rels = {str(x) for x in schema.get("relation_types", [])} if isinstance(schema, dict) else set()
        valid_paths = 0
        for p in paths:
            if not isinstance(p, dict):
                continue
            rel = str(p.get("relation", "")).strip()
            if (not allowed_rels) or (rel in allowed_rels):
                valid_paths += 1
        entity_term = min(len(entities), 6) / 6.0
        path_term = min(valid_paths, 10) / 10.0
        return _clip_float(0.55 * entity_term + 0.45 * path_term)

    def _ensemble_linking(self, payload: dict[str, Any], schema: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        candidates: list[StrategyResult] = []
        for p in self._strategy_payloads(payload):
            linked = self._invoke_strategy_linker(p)
            entities = _uniq([str(x) for x in linked.get("entities", [])])
            paths = linked.get("paths", []) if isinstance(linked.get("paths"), list) else []
            dsl = linked.get("dsl", {}) if isinstance(linked.get("dsl"), dict) else {}
            score = self._score_strategy_result(entities, paths, schema)
            candidates.append(
                StrategyResult(
                    strategy=str(p.get("strategy", "default")),
                    entities=entities,
                    paths=paths,
                    dsl=dsl,
                    score=score,
                )
            )

        if not candidates:
            fallback = self._invoke_linker(payload)
            return fallback, []

        candidates.sort(key=lambda x: x.score, reverse=True)
        best = candidates[0]
        merged_entities = _uniq([e for c in candidates[:2] for e in c.entities])
        merged_paths = [p for c in candidates[:2] for p in c.paths if isinstance(p, dict)]
        best_artifact = {
            "entities": merged_entities,
            "paths": merged_paths,
            "dsl": best.dsl,
            "linking_strategy": best.strategy,
        }
        ranking = [
            {"strategy": c.strategy, "score": c.score, "entities": len(c.entities), "paths": len(c.paths)}
            for c in candidates
        ]
        return best_artifact, ranking

    def _expand_schema_focus(self, question: str, schema: dict[str, Any]) -> dict[str, Any]:
        rels = [str(x) for x in schema.get("relation_types", [])] if isinstance(schema, dict) else []
        q_tokens = set(t.lower() for t in re.split(r"\W+", question) if t)
        focus_rels = [r for r in rels if any(t and t in r.lower() for t in q_tokens)]
        if not focus_rels:
            focus_rels = rels[:15]
        return {
            **schema,
            "focus_relation_types": focus_rels[:25],
            "schema_pruned": True,
        }

    def _run_query(self, cypher: str) -> list[dict[str, Any]]:
        if self.cypher_executor is None or not cypher:
            return []
        try:
            rows = self.cypher_executor(cypher)
        except Exception:
            return []
        if not isinstance(rows, list):
            return []
        out = [r for r in rows if isinstance(r, dict)]
        return out[: self.max_query_results]

    def _tokenize_question_terms(self, question: str) -> list[str]:
        stopwords = {
            "what", "which", "when", "where", "why", "how", "who", "whom",
            "the", "and", "for", "with", "from", "into", "than", "then",
            "above", "below", "shown", "showing", "following", "image", "picture", "photo",
            "take", "favorite", "favourite", "meal", "object", "thing",
            "look", "looks", "looking", "like", "seen", "here", "there",
            "does", "do", "did", "is", "are", "was", "were",
            "on", "in", "at", "to", "of", "by", "as",
            "请问", "请", "怎么", "如何", "为什么", "什么", "哪个", "哪些", "多少",
            "一下", "一个", "一种", "问题", "是否", "进行", "比较", "详细", "分析",
            "上面", "下面", "图片", "图中", "这张图", "这个图", "这个", "这个人", "这个物体",
        }

        parts = re.split(r"[^\w\u4e00-\u9fff]+", str(question or ""))
        out: list[str] = []
        seen: set[str] = set()

        for p in parts:
            p = p.strip()
            if len(p) < 2:
                continue
            key = p.lower()
            if key in stopwords:
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(p)

        return out[:12]

    def _parse_graph_context_docs(self, graph_context: str) -> list[dict[str, str]]:
        docs: list[dict[str, str]] = []

        current_source = "graph_context"
        buffer: list[str] = []

        def _flush() -> None:
            nonlocal buffer, current_source
            text = "\n".join([x for x in buffer if str(x).strip()]).strip()
            text = _strip_inline_data_uris(text)
            if text:
                docs.append({"source": current_source or "graph_context", "text": text[:3000]})
            buffer = []

        for raw in str(graph_context or "").splitlines():
            line = _strip_inline_data_uris(raw.strip())
            if not line:
                continue

            m = re.match(r"^\[(.*?)\]\s*(.*)$", line)
            if m:
                _flush()
                current_source = m.group(1).strip() or "graph_context"
                tail = _strip_inline_data_uris(m.group(2).strip())
                if tail:
                    buffer.append(tail)
            else:
                buffer.append(line)

        _flush()
        return docs

    def _normalize_proxy_documents(self, documents: Any) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()

        if not isinstance(documents, list):
            return out

        noisy_sources = {
            "question_type",
            "choice_hint",
        }

        for d in documents:
            if not isinstance(d, dict):
                continue

            source = str(d.get("source", "")).strip() or "proxy_doc"
            text = _strip_inline_data_uris(str(d.get("text", "")).strip())
            if not text:
                continue

            low_source = source.lower()
            low_text = text.lower()

            if source in noisy_sources:
                continue
            if low_source.endswith("#ambiguity"):
                continue
            if low_text.startswith("this is a 'what' question requiring"):
                continue
            if low_text.startswith("this is a 'where' question requiring"):
                continue
            if low_text.startswith("this is a 'why' question requiring"):
                continue
            if low_text.startswith("this is a 'how' question requiring"):
                continue
            if "图像歧义" in text and "未获得稳定视觉结论" in text:
                continue

            if "proxy_context" in low_text:
                continue
            if "fallback_proxy" in low_text:
                continue
            if "当前结果部分依赖网页/上下文代理描述" in text:
                continue
            if low_source.endswith("#hypothesis") and (
                "proxy_context" in low_text
                or "代理图片描述" in low_text
                or "网页图片候选信息" in low_text
            ):
                continue

            item = (source, text[:3000])
            if item in seen:
                continue
            seen.add(item)
            out.append({"source": source, "text": text[:3000]})

        return out

    def _pick_proxy_relation(self, schema: dict[str, Any]) -> str:
        if not isinstance(schema, dict):
            return "about"

        rels: list[str] = []
        for key in ["focus_relation_types", "relation_types"]:
            value = schema.get(key, [])
            if isinstance(value, list):
                rels.extend([str(x).strip() for x in value if str(x).strip()])

        if not rels:
            return "about"

        preferred = ["supports", "contradicts", "about", "co_mentions", "mentioned_in", "related_to"]
        rel_set = set(rels)

        for rel in preferred:
            if rel in rel_set:
                return rel

        return rels[0]

    def _proxy_row_bias(self, source: str, text: str) -> float:
        low_source = str(source or "").lower().strip()
        low_text = str(text or "").lower().strip()

        bias = 0.0

        if "#ocr_facts" in low_source:
            bias += 0.12

        if "local_image_file" in low_source and "#hypothesis" not in low_source:
            bias += 0.08

        if "#hypothesis" in low_source and any(
            marker in low_text
            for marker in [
                " not ",
                "not actually",
                "rather than",
                "instead of",
                "used as props",
                "staged",
                "不是",
                "并非",
                "不是真正",
                "不是真的",
                "并不是在",
                "更像是",
                "只是道具",
            ]
        ):
            bias += 0.05

        if "proxy_context" in low_text or "fallback_proxy" in low_text or "代理图片描述" in low_text:
            bias -= 0.20

        return bias

    def _retrieve_proxy_graph(
        self,
        question: str,
        schema: dict[str, Any],
        graph_context: str | None = None,
        documents: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        docs = self._normalize_proxy_documents(documents)
        if not docs:
            docs = self._parse_graph_context_docs(graph_context or "")
            docs = self._normalize_proxy_documents(docs)
        if not docs:
            return self._empty_result("proxy_graph_empty_context", ["proxy_graph_empty_context"])

        q_terms = self._tokenize_question_terms(question)
        q_terms_lower = {t.lower() for t in q_terms}
        chosen_rel = self._pick_proxy_relation(schema)
        question_focus = " / ".join(q_terms[:3]) if q_terms else "question"

        weak_terms = {
            "question", "questions", "answer", "answers", "logo", "symbol",
            "image", "picture", "photo", "object", "thing",
            "question_type", "proxy_doc",
        }

        def _doc_keywords(text: str) -> list[str]:
            toks = re.findall(r"[A-Za-z0-9_\-\u4e00-\u9fff]+", str(text or ""))
            out: list[str] = []
            seen: set[str] = set()
            for tok in toks:
                low = tok.lower().strip()
                if len(low) < 3:
                    continue
                if low in q_terms_lower:
                    continue
                if low in weak_terms:
                    continue
                if low in seen:
                    continue
                seen.add(low)
                out.append(low)
            return out[:6]

        scored_rows: list[dict[str, Any]] = []
        for d in docs:
            source = str(d.get("source", "proxy_doc")).strip() or "proxy_doc"
            text = _strip_inline_data_uris(str(d.get("text", "")).strip())
            if not text:
                continue

            lower_text = text.lower()
            lower_source = source.lower()

            matched_terms = [
                t for t in q_terms
                if t.lower() in lower_text or t.lower() in lower_source
            ]

            term_score = len(matched_terms) / max(len(q_terms), 1) if q_terms else 0.0

            question_norm = re.sub(r"[^\w\u4e00-\u9fff]+", "", str(question or "").lower())
            text_norm = re.sub(r"[^\w\u4e00-\u9fff]+", "", lower_text)
            source_norm = re.sub(r"[^\w\u4e00-\u9fff]+", "", lower_source)

            whole_query_hit = 1.0 if question_norm and (
                question_norm in text_norm or question_norm in source_norm
            ) else 0.0

            overlap_score = 0.75 * term_score + 0.25 * whole_query_hit
            if overlap_score <= 0.0:
                overlap_score = 0.02

            source_bias = self._proxy_row_bias(source, text)
            rerank_score = max(0.0, overlap_score + source_bias)

            scored_rows.append(
                {
                    "source": source,
                    "doc_name": source,
                    "text": text[:3000],
                    "rerank_score": rerank_score,
                    "recall_score": overlap_score,
                    "matched_terms": matched_terms,
                    "doc_keywords": _doc_keywords(text),
                }
            )

        if not scored_rows:
            return self._empty_result("proxy_graph_no_rows", ["proxy_graph_no_rows"])

        scored_rows.sort(key=lambda x: float(x.get("rerank_score", 0.0)), reverse=True)
        min_score = float(
            getattr(settings, "proxy_graph_min_rerank_score", self.proxy_graph_min_rerank_score)
            or self.proxy_graph_min_rerank_score
        )
        before_filter = len(scored_rows)
        scored_rows = [
            row for row in scored_rows
            if float(row.get("rerank_score", 0.0) or 0.0) >= min_score
        ]
        filtered_low_score = before_filter - len(scored_rows)
        if not scored_rows:
            result = self._empty_result("proxy_graph_low_relevance", ["proxy_graph_low_relevance"])
            la = result.setdefault("linking_artifacts", {})
            if isinstance(la, dict):
                la["proxy_graph_min_rerank_score"] = min_score
                la["proxy_graph_filtered_low_score"] = filtered_low_score
            return result
        scored_rows = scored_rows[: self.proxy_top_k]
        for row in scored_rows:
            row["evidence_origin"] = "proxy_graph"
            row["relation_is_inferred"] = True

        entities = _uniq(
            [
                *q_terms,
                *[str(t) for row in scored_rows for t in row.get("matched_terms", [])],
                *[str(t) for row in scored_rows for t in row.get("doc_keywords", [])],
            ]
        )[:16]

        paths: list[dict[str, Any]] = []
        seen_paths: set[tuple[str, str, str, str]] = set()

        for row in scored_rows:
            source = str(row.get("source", "proxy_doc"))
            matched = row.get("matched_terms", [])
            if not isinstance(matched, list):
                matched = []
            keywords = row.get("doc_keywords", [])
            if not isinstance(keywords, list):
                keywords = []

            low_text = str(row.get("text", "")).lower()
            has_negation = any(
                marker in low_text
                for marker in [
                    " not ",
                    "not actually",
                    "rather than",
                    "instead of",
                    "used as props",
                    "staged",
                    "不是",
                    "并非",
                    "不属于",
                    "不可能是",
                    "不是真正",
                    "不是真的",
                    "并不是在",
                    "更像是",
                    "只是道具",
                ]
            )

            for term in matched[:3]:
                head = str(term).strip()
                if not head:
                    continue
                key = (head, "mentioned_in", source, source)
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                paths.append(
                    {
                        "subject": head,
                        "relation": "mentioned_in",
                        "object": source,
                        "source": source,
                        "evidence_origin": "proxy_graph",
                        "relation_is_inferred": True,
                    }
                )

            if matched or keywords:
                rel = "contradicts" if has_negation else (
                    "supports" if chosen_rel in {"supports", "contradicts", "about", "co_mentions"} else "about"
                )
                key = (source, rel, question_focus, source)
                if key not in seen_paths:
                    seen_paths.add(key)
                    paths.append(
                        {
                            "subject": source,
                            "relation": rel,
                            "object": question_focus,
                            "source": source,
                            "evidence_origin": "proxy_graph",
                            "relation_is_inferred": True,
                        }
                    )

            if keywords:
                obj = ", ".join([str(x) for x in keywords[:3]])
                key = (source, "about", obj, source)
                if key not in seen_paths:
                    seen_paths.add(key)
                    paths.append(
                        {
                            "subject": source,
                            "relation": "about",
                            "object": obj,
                            "source": source,
                            "evidence_origin": "proxy_graph",
                            "relation_is_inferred": True,
                        }
                    )

            if len(paths) >= self.proxy_max_paths:
                break

        coverage = _clip_float(len(scored_rows) / max(min(len(docs), self.proxy_top_k), 1))
        confidence = _clip_float(
            0.6 * _mean([float(r.get("rerank_score", 0.0)) for r in scored_rows[:3]])
            + 0.2 * (min(len(paths), 5) / 5.0)
            + 0.2 * (1.0 if scored_rows else 0.0)
        )

        return {
            "graph_evidence": {
                "triples": paths,
                "paths": paths,
                "query_results": scored_rows,
                "image_evidence": [],
                "proxy_graph": True,
            },
            "linking_artifacts": {
                "entities": entities,
                "paths": paths,
                "opencypher": "",
                "dsl": {},
                "reranker": {"L": 0, "k": len(scored_rows)},
                "strategy_ranking": [
                    {
                        "strategy": "proxy_graph_from_context",
                        "score": confidence,
                        "entities": len(entities),
                        "paths": len(paths),
                    }
                ],
                "selected_strategy": "proxy_graph_from_context",
                "refinement_history": [{"round": 1, "ok": True, "errors": []}],
                "external_backend": "local_proxy_graph",
                "proxy_graph_min_rerank_score": min_score,
                "proxy_graph_filtered_low_score": filtered_low_score,
            },
            "confidence": confidence,
            "coverage": coverage,
        }

    def _qianfan_enabled(self) -> bool:
        return bool(self.qianfan_api_key and self.qianfan_kb_ids)

    def _extract_image_candidates_from_qianfan_chunk(self, ch: dict[str, Any]) -> list[str]:
        urls: list[str] = []
        seen: set[str] = set()

        def _push(u: str) -> None:
            u = str(u).strip()
            if not u or u in seen:
                return
            seen.add(u)
            urls.append(u)

        contents = ch.get("content", [])
        if isinstance(contents, list):
            for item in contents:
                if not isinstance(item, dict):
                    continue
                for key in ["image_url", "url", "file_url", "src"]:
                    v = item.get(key)
                    if isinstance(v, str) and v.strip():
                        _push(v)

        meta = ch.get("meta", {})
        blob = json.dumps(meta, ensure_ascii=False) if isinstance(meta, dict) else ""
        for u in re.findall(r"https?://[^\s\"'<>]+", blob):
            if re.search(r"\.(jpg|jpeg|png|webp)(\?|$)", u.lower()):
                _push(u)

        return urls[:4]

    def _search_qianfan_kb(self, question: str) -> list[dict[str, Any]]:
        if not self._qianfan_enabled():
            self._record_external_error("qianfan", "disabled", "missing_api_key_or_kb_ids")
            return []

        payload = {
            "query": question,
            "knowledgebase_ids": self.qianfan_kb_ids,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        req = Request(
            self.qianfan_search_url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.qianfan_api_key}",
            },
        )

        try:
            with urlopen(req, timeout=self.qianfan_timeout) as resp:
                raw = resp.read().decode("utf-8")
                obj = json.loads(raw)
        except HTTPError as exc:
            self._record_external_error("qianfan", "http", str(exc))
            return []
        except URLError as exc:
            self._record_external_error("qianfan", "timeout" if isinstance(getattr(exc, "reason", None), TimeoutError) else "url", str(exc))
            return []
        except TimeoutError as exc:
            self._record_external_error("qianfan", "timeout", str(exc))
            return []
        except json.JSONDecodeError as exc:
            self._record_external_error("qianfan", "json", str(exc))
            return []
        except Exception as exc:
            self._record_external_error("qianfan", "error", f"{type(exc).__name__}: {exc}")
            return []

        chunks = obj.get("chunks", [])
        if not isinstance(chunks, list):
            self._record_external_error("qianfan", "json", "chunks_not_list")
            return []

        rows: list[dict[str, Any]] = []
        filtered_empty_text = 0
        filtered_low_score = 0
        min_score = float(
            getattr(settings, "qianfan_min_rerank_score", self.qianfan_min_rerank_score)
            or self.qianfan_min_rerank_score
        )
        for ch in chunks[: self.qianfan_top_k]:
            if not isinstance(ch, dict):
                continue

            contents = ch.get("content", [])
            text_parts: list[str] = []
            if isinstance(contents, list):
                for item in contents:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(str(item.get("text", "")))

            text = _strip_inline_data_uris("\n".join(text_parts))

            meta = ch.get("meta", {}) if isinstance(ch.get("meta"), dict) else {}
            doc_info = meta.get("doc_info", {}) if isinstance(meta.get("doc_info"), dict) else {}
            rerank = ch.get("rerank", {}) if isinstance(ch.get("rerank"), dict) else {}
            recall = ch.get("recall", {}) if isinstance(ch.get("recall"), dict) else {}
            rerank_score = float(rerank.get("score", 0.0) or 0.0)

            if not text.strip():
                filtered_empty_text += 1
                continue
            if rerank_score < min_score:
                filtered_low_score += 1
                continue

            image_candidates = self._extract_image_candidates_from_qianfan_chunk(ch)

            rows.append(
                {
                    "source": "qianfan_kb",
                    "chunk_id": str(ch.get("chunk_id", "")),
                    "doc_id": str(doc_info.get("doc_id", "")),
                    "doc_name": str(doc_info.get("doc_name", "")),
                    "text": text,
                    "rerank_score": rerank_score,
                    "rerank_position": int(rerank.get("position", 0) or 0),
                    "recall_score": float(recall.get("score", 0.0) or 0.0),
                    "recall_position": int(recall.get("position", 0) or 0),
                    "tokens": int(meta.get("tokens", 0) or 0),
                    "word_count": int(meta.get("word_count", 0) or 0),
                    "image_candidates": image_candidates,
                    "meta": meta,
                }
            )

        self._set_external_stat("qianfan", "filtered_empty_text", filtered_empty_text)
        self._set_external_stat("qianfan", "filtered_low_score", filtered_low_score)
        self._set_external_stat("qianfan", "min_rerank_score", min_score)
        if not rows:
            self._record_external_error(
                "qianfan",
                "no_hits",
                f"chunks={len(chunks)} filtered_empty_text={filtered_empty_text} filtered_low_score={filtered_low_score}",
            )
        return rows

    def _build_qianfan_result(self, question: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        rows = [r for r in rows if isinstance(r, dict) and str(r.get("text", "")).strip()]
        if not rows:
            return self._empty_result("qianfan_no_hits", ["qianfan_no_hits"])

        rows = self._semantic_rerank_external_rows(
            question=question,
            rows=rows,
            keep_k=max(self.qianfan_top_k, 5),
        )
        coverage = _clip_float(len(rows) / max(self.qianfan_top_k, 1))
        confidence = _clip_float(
            0.7 * _mean([float(r.get("rerank_score", 0.0)) for r in rows[:3]])
            + 0.3 * (1.0 if rows else 0.0)
        )

        entities = _uniq([t for t in re.split(r"\W+", question) if len(t) > 2][:6])

        return {
            "graph_evidence": {
                "triples": [],
                "paths": [],
                "query_results": rows,
                "image_evidence": [],
                "kb_chunks": rows,
            },
            "linking_artifacts": {
                "entities": entities,
                "paths": [],
                "opencypher": "",
                "dsl": {},
                "reranker": {"L": 0, "k": self.qianfan_top_k},
                "strategy_ranking": [
                    {
                        "strategy": "qianfan_kb",
                        "score": confidence,
                        "entities": len(entities),
                        "paths": 0,
                    }
                ],
                "selected_strategy": "qianfan_kb",
                "refinement_history": [{"round": 1, "ok": True, "errors": []}],
                "external_backend": "qianfan_kb",
                "kb_ids": self.qianfan_kb_ids,
            },
            "confidence": confidence,
            "coverage": coverage,
        }

    def _annotate_retrieval_result(self, result: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(result, dict):
            return result

        result.setdefault("confidence_type", "retrieval")
        result.setdefault("confidence_label", "retrieval_confidence_not_final_answer")

        linking_artifacts = result.get("linking_artifacts", {})
        if isinstance(linking_artifacts, dict):
            linking_artifacts.setdefault("confidence_type", "retrieval")
            linking_artifacts.setdefault("confidence_label", "retrieval_confidence_not_final_answer")

        return result

    def _empty_result(self, strategy: str, errors: list[str] | None = None) -> dict[str, Any]:
        return {
            "graph_evidence": {
                "triples": [],
                "paths": [],
                "query_results": [],
                "image_evidence": [],
            },
            "linking_artifacts": {
                "entities": [],
                "paths": [],
                "opencypher": "",
                "dsl": {},
                "reranker": {"L": 0, "k": 0},
                "strategy_ranking": [],
                "selected_strategy": strategy,
                "refinement_history": [{"round": 1, "ok": False, "errors": errors or []}],
            },
            "confidence": 0.0,
            "coverage": 0.0,
        }
    
    def _result_has_signal(self, result: dict[str, Any]) -> bool:
        if not isinstance(result, dict):
            return False

        min_conf = float(getattr(settings, "graph_signal_min_confidence", 0.55) or 0.55)
        min_query_results = int(getattr(settings, "graph_signal_min_query_results", 2) or 2)
        min_kb_score = float(getattr(settings, "graph_signal_min_kb_score", 0.72) or 0.72)

        graph_evidence = result.get("graph_evidence", {})
        linking_artifacts = result.get("linking_artifacts", {})

        if not isinstance(graph_evidence, dict):
            graph_evidence = {}
        if not isinstance(linking_artifacts, dict):
            linking_artifacts = {}

        triples = graph_evidence.get("triples", [])
        if isinstance(triples, list) and len(triples) > 0:
            if all(isinstance(p, dict) and p.get("relation_is_inferred") is True for p in triples):
                query_results = graph_evidence.get("query_results", [])
                best_proxy_score = 0.0
                if isinstance(query_results, list):
                    for row in query_results:
                        if isinstance(row, dict):
                            best_proxy_score = max(
                                best_proxy_score,
                                float(row.get("rerank_score", row.get("score", 0.0)) or 0.0),
                            )
                if (
                    float(result.get("confidence", 0.0) or 0.0) >= min_conf
                    and isinstance(query_results, list)
                    and len(query_results) >= min_query_results
                    and best_proxy_score >= min_conf
                ):
                    return True
            else:
                return True

        paths = graph_evidence.get("paths", [])
        if isinstance(paths, list) and len(paths) > 0:
            if all(isinstance(p, dict) and p.get("relation_is_inferred") is True for p in paths):
                query_results = graph_evidence.get("query_results", [])
                best_proxy_score = 0.0
                if isinstance(query_results, list):
                    for row in query_results:
                        if isinstance(row, dict):
                            best_proxy_score = max(
                                best_proxy_score,
                                float(row.get("rerank_score", row.get("score", 0.0)) or 0.0),
                            )
                if (
                    float(result.get("confidence", 0.0) or 0.0) >= min_conf
                    and isinstance(query_results, list)
                    and len(query_results) >= min_query_results
                    and best_proxy_score >= min_conf
                ):
                    return True
            else:
                return True

        image_evidence = graph_evidence.get("image_evidence", [])
        if isinstance(image_evidence, list) and len(image_evidence) > 0:
            return True

        query_results = graph_evidence.get("query_results", [])
        if isinstance(query_results, list) and len(query_results) >= min_query_results:
            best_score = 0.0
            for row in query_results:
                if not isinstance(row, dict):
                    continue
                try:
                    best_score = max(
                        best_score,
                        float(row.get("rerank_score", row.get("score", 0.0)) or 0.0),
                    )
                except Exception:
                    continue
            if best_score >= min_conf:
                return True

        kb_chunks = graph_evidence.get("kb_chunks", [])
        if isinstance(kb_chunks, list) and len(kb_chunks) > 0:
            best_kb_score = 0.0
            for row in kb_chunks:
                if not isinstance(row, dict):
                    continue
                try:
                    best_kb_score = max(
                        best_kb_score,
                        float(row.get("rerank_score", row.get("score", 0.0)) or 0.0),
                    )
                except Exception:
                    continue
            if best_kb_score >= min_kb_score:
                return True

        linked_paths = linking_artifacts.get("paths", [])
        if isinstance(linked_paths, list) and len(linked_paths) > 0:
            return True

        return False

    def _retrieve_graph_only(
        self,
        question: str,
        schema: dict[str, Any],
        graph_context: str | None = None,
    ) -> dict[str, Any]:
        reranker = self._dynamic_reranker(question, schema)
        artifacts: dict[str, Any] = {"entities": [], "paths": [], "dsl": {}, "opencypher": ""}
        history: list[dict[str, Any]] = []

        payload = {
            "question": question,
            "schema": self._expand_schema_focus(question, schema),
            "graph_context": graph_context or "",
            "mode": "full",
            "reranker": {"L": reranker.hop_l, "k": reranker.top_k},
        }
        strategy_ranking: list[dict[str, Any]] = []
        linked: dict[str, Any] = {}

        for i in range(max(self.max_refine_rounds, 1)):
            linked, strategy_ranking = self._ensemble_linking(payload, schema)
            artifacts.update({k: v for k, v in linked.items() if k in {"entities", "paths", "dsl", "opencypher"}})

            if not artifacts.get("opencypher") and isinstance(artifacts.get("dsl"), dict):
                try:
                    artifacts["opencypher"] = self.compiler.compile(artifacts["dsl"])
                except Exception:
                    artifacts["opencypher"] = ""

            ok, errors = self.validator.validate(artifacts, schema)
            history.append({"round": i + 1, "ok": ok, "errors": errors})
            if ok:
                break

            payload = {
                "question": question,
                "schema": {
                    "summary": schema.get("summary", ""),
                    "relation_types": schema.get("relation_types", []),
                    "node_types": schema.get("node_types", []),
                },
                "graph_context": graph_context or "",
                "previous_artifacts": artifacts,
                "errors": errors,
                "mode": "repair-only",
                "allowed_patch_fields": ["entities", "paths", "dsl", "opencypher"],
            }

        entities = _uniq([str(x) for x in artifacts.get("entities", [])])
        paths = artifacts.get("paths", []) if isinstance(artifacts.get("paths"), list) else []
        cypher = str(artifacts.get("opencypher", "")).strip()
        query_results = self._run_query(cypher)
        image_evidence = [
            row for row in query_results
            if any(k in row for k in ["image", "image_url", "bbox", "ocr_text", "visual_object"])
        ]

        graph_evidence: dict[str, Any] = {
            "triples": [p for p in paths if isinstance(p, dict)],
            "paths": paths,
            "query_results": query_results,
            "image_evidence": image_evidence,
        }
        linking_artifacts = {
            "entities": entities,
            "paths": paths,
            "opencypher": cypher,
            "dsl": artifacts.get("dsl", {}),
            "reranker": {"L": reranker.hop_l, "k": reranker.top_k},
            "strategy_ranking": strategy_ranking,
            "selected_strategy": linked.get("linking_strategy", "unknown") if isinstance(linked, dict) else "unknown",
            "refinement_history": history,
        }

        coverage = _clip_float(
            (len(entities) / 4.0) * 0.5
            + (len(paths) / max(reranker.top_k, 1)) * 0.3
            + (min(len(query_results), 5) / 5.0) * 0.2
        )
        confidence = _clip_float(0.5 * coverage + (0.5 if cypher else 0.0))

        return {
            "graph_evidence": graph_evidence,
            "linking_artifacts": linking_artifacts,
            "confidence": confidence,
            "coverage": coverage,
        }

    def _search_qianfan_kb_multi(self, question: str, query_variants: list[str] | None = None) -> list[dict[str, Any]]:
        queries = [str(question or "").strip()]
        if isinstance(query_variants, list):
            queries.extend([str(x).strip() for x in query_variants if str(x).strip()])

        rows_all: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        for q in _uniq(queries):
            rows = self._search_qianfan_kb(q)
            for row in rows:
                key = (
                    str(row.get("chunk_id", "")).strip(),
                    str(row.get("doc_id", "")).strip(),
                    str(row.get("text", ""))[:200],
                )
                if key in seen:
                    continue
                seen.add(key)
                rows_all.append(row)

        rows_all.sort(key=lambda x: float(x.get("rerank_score", 0.0)), reverse=True)
        return rows_all[: max(self.qianfan_top_k, 5)]

    def retrieve_graph_evidence(
        self,
        question: str,
        schema: dict[str, Any],
        graph_context: str | None = None,
        documents: list[dict[str, str]] | None = None,
        query_variants: list[str] | None = None,
        retrieval_backend: str | None = None,
        allow_kb_augmentation: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        backend = str(retrieval_backend or self.retrieval_backend or "auto").strip().lower() or "auto"
        external_kg_blocked_by_strict_no_web = False
        if (
            bool(getattr(settings, "strict_no_external_kg_when_web_disabled", False))
            and os.getenv("ENABLE_WEB_SEARCH_TOOL", str(getattr(settings, "enable_web_search_tool", True))).strip().lower()
            in {"0", "false", "no", "off"}
        ):
            external_kg_blocked_by_strict_no_web = True
            allow_kb_augmentation = False
            if backend in {"", "auto", "hybrid", "qianfan", "dbpedia"}:
                backend = "light_rag"

        def _finalize(result: dict[str, Any]) -> dict[str, Any]:
            if isinstance(result, dict):
                la = result.setdefault("linking_artifacts", {})
                if isinstance(la, dict):
                    la.setdefault("effective_backend", backend)
                    la.setdefault("allow_kb_augmentation", bool(allow_kb_augmentation))
                    if external_kg_blocked_by_strict_no_web:
                        la["external_kg_blocked_by_strict_no_web"] = True
                    external_errors = self._consume_external_errors()
                    if external_errors:
                        la["external_errors"] = external_errors
                    external_stats = self._consume_external_stats()
                    if external_stats:
                        la["external_stats"] = external_stats
            return self._annotate_retrieval_result(result)

        def _normalized_query_variants() -> list[str]:
            variants: list[str] = []
            if isinstance(query_variants, list):
                variants.extend([str(x).strip() for x in query_variants if str(x).strip()])

            q = str(question or "").strip()
            if q and q not in variants:
                variants.insert(0, q)

            return _uniq(variants)

        def _merge_qianfan_rows_into_result(
            result: dict[str, Any],
            rows: list[dict[str, Any]],
            *,
            external_backend_name: str,
        ) -> dict[str, Any]:
            if not rows:
                return result

            ge = result.setdefault("graph_evidence", {})
            if not isinstance(ge, dict):
                ge = {}
                result["graph_evidence"] = ge

            ge.setdefault("kb_chunks", [])
            ge["kb_chunks"].extend(rows)

            ge.setdefault("query_results", [])
            ge["query_results"].extend(rows)
            ge["query_results"].sort(
                key=lambda x: float(x.get("rerank_score", 0.0) or 0.0),
                reverse=True,
            )

            la = result.setdefault("linking_artifacts", {})
            if isinstance(la, dict):
                la["external_backend"] = external_backend_name
                la["kb_ids"] = self.qianfan_kb_ids

            kb_conf = _clip_float(
                _mean([float(r.get("rerank_score", 0.0) or 0.0) for r in rows[:3]])
            )
            result["confidence"] = _clip_float(max(float(result.get("confidence", 0.0) or 0.0), kb_conf))
            result["coverage"] = _clip_float(
                max(float(result.get("coverage", 0.0) or 0.0), len(rows) / max(self.qianfan_top_k, 1))
            )
            return result

        # ------------------------------------------------------------------
        # route: proxy_graph / light_rag
        # ------------------------------------------------------------------
        if backend in {"proxy_graph", "light_rag"}:
            proxy_result = self._retrieve_proxy_graph(
                question=question,
                schema=schema,
                graph_context=graph_context,
                documents=documents,
            )

            if allow_kb_augmentation and self._qianfan_enabled():
                kb_rows_all: list[dict[str, Any]] = []
                seen_kb: set[tuple[str, str, str]] = set()

                for q in _normalized_query_variants():
                    for row in self._search_qianfan_kb(q):
                        key = (
                            str(row.get("chunk_id", "")).strip(),
                            str(row.get("doc_id", "")).strip(),
                            str(row.get("text", ""))[:200],
                        )
                        if key in seen_kb:
                            continue
                        seen_kb.add(key)
                        kb_rows_all.append(row)

                if kb_rows_all:
                    kb_rows_all.sort(
                        key=lambda x: float(x.get("rerank_score", 0.0) or 0.0),
                        reverse=True,
                    )
                    kb_rows_all = kb_rows_all[: max(self.qianfan_top_k, 5)]
                    proxy_result = _merge_qianfan_rows_into_result(
                        proxy_result,
                        kb_rows_all,
                        external_backend_name="proxy_graph+qianfan_kb",
                    )

            return _finalize(proxy_result)

        # ------------------------------------------------------------------
        # capabilities
        # ------------------------------------------------------------------
        schema_has_signal = bool(schema) and bool(
            schema.get("relation_types") or schema.get("node_types") or schema.get("summary")
        )
        graph_capable = bool(self.cypher_executor is not None) and schema_has_signal
        kb_capable = self._qianfan_enabled()
        dbpedia_capable = self._dbpedia_enabled()

        # ------------------------------------------------------------------
        # route: dbpedia
        # ------------------------------------------------------------------
        if backend == "dbpedia":
            rows = self._search_dbpedia_multi(
                question=question,
                schema=schema,
                query_variants=query_variants,
                documents=documents,
            )
            return _finalize(self._build_dbpedia_result(question, rows))

        # ------------------------------------------------------------------
        # route: qianfan
        # ------------------------------------------------------------------
        if backend == "qianfan":
            rows = self._search_qianfan_kb_multi(
                question,
                query_variants=query_variants,
            )
            return _finalize(self._build_qianfan_result(question, rows))

        # ------------------------------------------------------------------
        # route: graph
        # ------------------------------------------------------------------
        if backend == "graph":
            if not graph_capable:
                return _finalize(
                    self._empty_result("graph_unavailable", ["graph_unavailable"])
                )
            return _finalize(
                self._retrieve_graph_only(question, schema, graph_context)
            )

        # ------------------------------------------------------------------
        # route: auto = graph -> qianfan -> dbpedia
        # ------------------------------------------------------------------
        if backend == "auto":
            graph_result = self.retrieve_graph_evidence(
                question=question,
                schema=schema,
                graph_context=graph_context,
                documents=documents,
                query_variants=query_variants,
                retrieval_backend="graph",
                allow_kb_augmentation=allow_kb_augmentation,
            )
            if self._result_has_signal(graph_result):
                return graph_result

            if kb_capable:
                qianfan_result = self.retrieve_graph_evidence(
                    question=question,
                    schema=schema,
                    graph_context=graph_context,
                    documents=documents,
                    query_variants=query_variants,
                    retrieval_backend="qianfan",
                    allow_kb_augmentation=allow_kb_augmentation,
                )
                if self._result_has_signal(qianfan_result):
                    return qianfan_result

            if dbpedia_capable:
                dbpedia_result = self.retrieve_graph_evidence(
                    question=question,
                    schema=schema,
                    graph_context=graph_context,
                    documents=documents,
                    query_variants=query_variants,
                    retrieval_backend="dbpedia",
                    allow_kb_augmentation=allow_kb_augmentation,
                )
                if self._result_has_signal(dbpedia_result):
                    return dbpedia_result

            return _finalize(
                self._empty_result(
                    "no_backend_available",
                    ["graph_unavailable", "kb_unavailable", "dbpedia_unavailable"],
                )
            )

        # ------------------------------------------------------------------
        # route: hybrid = graph + qianfan
        # ------------------------------------------------------------------
        if backend == "hybrid":
            graph_result = (
                self._retrieve_graph_only(question, schema, graph_context)
                if graph_capable
                else self._empty_result("graph_unavailable", ["graph_unavailable"])
            )

            if kb_capable:
                rows = self._search_qianfan_kb_multi(
                    question,
                    query_variants=query_variants,
                )
                if rows:
                    graph_result = _merge_qianfan_rows_into_result(
                        graph_result,
                        rows,
                        external_backend_name="qianfan_kb",
                    )

            return _finalize(graph_result)

        # ------------------------------------------------------------------
        # unknown backend
        # ------------------------------------------------------------------
        return _finalize(
            self._empty_result(f"unknown_backend:{backend}", [f"unknown_backend:{backend}"])
        )
