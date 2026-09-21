from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from byokg_rag import BYOKGRAGProvider
from config import settings
from lexical_graph import LexicalGraphRetriever, build_graph_from_corpus
from models import QwenVLClient

def _is_local_image_path(value: str) -> bool:
    v = str(value or "").strip()
    if not v:
        return False
    if v.startswith("file://"):
        return True
    return Path(v).expanduser().is_file()

def _get_allowed_local_media_root() -> Path | None:
    raw = os.getenv("VLLM_ALLOWED_LOCAL_MEDIA_ROOT", "").strip()
    if not raw:
        return None
    try:
        p = Path(raw).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    except Exception:
        return None


def _extract_local_path_from_ref(image_ref: str) -> Path | None:
    raw = str(image_ref or "").strip()
    if not raw:
        return None

    try:
        if raw.startswith("file://"):
            parsed = urlparse(raw)
            path = unquote(parsed.path)
            return Path(path).expanduser().resolve()
        return Path(raw).expanduser().resolve()
    except Exception:
        return None


def _is_under_allowed_local_media_root(path: Path, allowed_root: Path | None) -> bool:
    if allowed_root is None:
        return True
    try:
        resolved_path = path.resolve()
        resolved_root = allowed_root.resolve()
    except Exception:
        return False
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def _normalize_local_image_ref(image_ref: str) -> tuple[str, str]:
    raw = str(image_ref or "").strip()
    if not raw:
        return "", "empty"

    local_path = _extract_local_path_from_ref(raw)
    if local_path is None or not local_path.is_file():
        return "", "file_not_found"

    allowed_root = _get_allowed_local_media_root()
    if not _is_under_allowed_local_media_root(local_path, allowed_root):
        return "", "outside_allowed_local_media_root"

    return f"file://{local_path.as_posix()}", "local_path"


def _prepare_vl_image_ref(image_ref: str, bucket: str = "generic") -> tuple[str, str]:
    raw = str(image_ref or "").strip()
    if not raw:
        return "", "empty"

    # 允许 data URL 原样传给下游 client；
    # 是否可用，由具体后端在 models.py 决定。
    if _is_data_url(raw):
        return raw, "data_url"

    # 本地图 / file://：只做本地规范化，不在 ToolRouter 里提前处理
    if _is_local_image_path(raw):
        local_path = _extract_local_path_from_ref(raw)
        if local_path is None or not local_path.is_file():
            return "", "file_not_found"
        return _to_file_url(local_path), "local_path"

    # 远程图：原样保留，让下游 client 根据后端决定
    if _is_remote_http_url(raw):
        return raw, "remote_http_url"

    return "", "invalid_image_ref"


def _safe_int(value: Any, default: int, low: int | None = None, high: int | None = None) -> int:
    try:
        x = int(value)
    except Exception:
        x = default
    if low is not None:
        x = max(low, x)
    if high is not None:
        x = min(high, x)
    return x

def _env_flag(name: str, default: bool = True) -> bool:
    value = str(getattr(settings, name, "") or os.getenv(name.upper(), "")).strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}

def _env_str(name: str, default: str = "") -> str:
    value = getattr(settings, name, None)
    if value is None or str(value).strip() == "":
        value = os.getenv(name.upper(), default)
    return str(value or "").strip()


def _env_list(name: str, default: str = "") -> list[str]:
    raw = _env_str(name, default=default)
    return [x.strip() for x in raw.split(",") if x.strip()]




def _http_verify_value() -> bool | str:
    ca_bundle = os.getenv("HTTP_CA_BUNDLE", "").strip()
    if ca_bundle:
        return ca_bundle
    return _env_flag("http_verify", default=True)


def make_http_client(timeout: float = 20, follow_redirects: bool = False) -> httpx.Client:
    return httpx.Client(
        timeout=timeout,
        follow_redirects=follow_redirects,
        verify=_http_verify_value(),
        trust_env=True,
        headers={"User-Agent": "DeepResearch-Eval/1.0"},
    )


def _guess_image_mime(url: str, content_type: str | None = None) -> str:
    if content_type and content_type.startswith("image/"):
        return content_type.split(";")[0].strip()
    mime, _ = mimetypes.guess_type(url)
    if mime and mime.startswith("image/"):
        return mime
    return "image/jpeg"

def _to_file_url(path: Path) -> str:
    return f"file://{path.resolve().as_posix()}"


def _stage_local_image_to_allowed_root(
    src_path: Path,
    bucket: str = "generic",
) -> tuple[str, str]:
    allowed_root = _get_allowed_local_media_root()
    if allowed_root is None:
        return "", "allowed_local_media_root_missing"

    try:
        src = src_path.expanduser().resolve()
        if not src.is_file():
            return "", "file_not_found"

        content = src.read_bytes()
        if not content:
            return "", "empty"

        ext = src.suffix.lower().strip() or ".jpg"
        if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
            ext = ".jpg"

        target_dir = allowed_root / bucket / "local_files"
        target_dir.mkdir(parents=True, exist_ok=True)

        digest = hashlib.sha1(content).hexdigest()
        out_path = target_dir / f"{digest}{ext}"
        if not out_path.exists():
            out_path.write_bytes(content)

        return _to_file_url(out_path), "local_path_staged"
    except Exception:
        return "", "local_stage_failed"


def _download_remote_image_to_allowed_root(
    image_url: str,
    bucket: str = "generic",
    timeout: float = 30.0,
    max_bytes: int = 15 * 1024 * 1024,
) -> tuple[str, str]:
    allowed_root = _get_allowed_local_media_root()
    if allowed_root is None:
        return "", "allowed_local_media_root_missing"

    try:
        with make_http_client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(image_url)
            resp.raise_for_status()

        content = resp.content
        if not content:
            return "", "remote_image_empty"
        if len(content) > max_bytes:
            return "", "remote_image_too_large"

        content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
        mime = _guess_image_mime(image_url, content_type)
        if not mime.startswith("image/"):
            return "", "remote_not_image"

        ext_map = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/jpg": ".jpg",
        }
        ext = ext_map.get(mime, ".jpg")

        target_dir = allowed_root / bucket / "remote_downloads"
        target_dir.mkdir(parents=True, exist_ok=True)

        digest = hashlib.sha1(content).hexdigest()
        out_path = target_dir / f"{digest}{ext}"
        if not out_path.exists():
            out_path.write_bytes(content)

        return _to_file_url(out_path), "remote_url_staged"
    except Exception as exc:
        return "", f"remote_download_failed:{type(exc).__name__}"


def _is_data_url(value: str) -> bool:
    return value.strip().lower().startswith("data:image/")


def _is_remote_http_url(value: str) -> bool:
    v = value.strip().lower()
    return v.startswith("http://") or v.startswith("https://")


def clean_text_content(text: str, max_chars: int = 5000) -> str:
    text = " ".join((text or "").split())
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def _filename_from_url(url: str) -> str:
    try:
        tail = url.rstrip("/").split("/")[-1]
        return tail[:200]
    except Exception:
        return ""


@dataclass
class SearchResult:
    title: str
    link: str
    snippet: str


@dataclass
class ImageCandidate:
    image_url: str
    alt_text: str
    title_text: str
    caption_text: str
    nearby_text: str
    page_url: str


@dataclass
class ToolObservation:
    tool_name: str
    input_data: dict
    output_data: str

@dataclass
class StructuredImageAnalysis:
    visible_facts: dict
    ocr_facts: list[dict]
    hypotheses: list[dict]
    ambiguities: list[str]
    suggested_followups: list[str]

@dataclass
class StructuredOCR:
    ocr_lines: list[dict]
    has_text: bool

def build_fallback_image_description(candidate: ImageCandidate | None = None, image_url: str = "", extra_text: str = "") -> str:
    """
    生成代理图片描述（fallback description）。
    """
    parts: list[str] = []

    if candidate is not None:
        if candidate.alt_text:
            parts.append(f"alt: {candidate.alt_text}")
        if candidate.title_text:
            parts.append(f"title: {candidate.title_text}")
        if candidate.caption_text:
            parts.append(f"caption: {candidate.caption_text}")
        if candidate.nearby_text:
            parts.append(f"context: {candidate.nearby_text}")

        if not image_url:
            image_url = candidate.image_url

    if extra_text.strip():
        parts.append(f"hint: {extra_text.strip()[:500]}")

    if not parts:
        filename = _filename_from_url(image_url)
        if filename:
            parts.append(f"filename: {filename}")

    text = " | ".join(parts).strip()
    if not text:
        text = "网页中检测到一张图片，但缺少可用于描述的上下文信息。"

    return f"代理图片描述（基于网页上下文，非视觉识别）: {text[:1000]}"


class SearchTool:
    """Web search via Tavily Search API with hardcoded multi-key rotation."""

    def __init__(self) -> None:
        self.api_keys = [
            ""
        ]

        # 这些参数也直接写死；如果想改成 advanced / news 也可以在这里改
        self.search_depth = "basic"
        self.topic = "general"
        self.include_raw_content = False

        # 进程内临时失效表：key -> fail_count
        self._disabled_keys: dict[str, int] = {}

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query = str(query or "").strip()
        if not query or not self.api_keys:
            return []

        # 优先尝试失败次数更少的 key
        ordered_keys = sorted(
            self.api_keys,
            key=lambda k: self._disabled_keys.get(k, 0),
        )

        for api_key in ordered_keys:
            if not self._key_has_quota(api_key):
                continue

            results, ok = self._search_tavily_with_key(
                api_key=api_key,
                query=query,
                top_k=top_k,
            )
            
            if ok and results:
                filtered = [r for r in results if not self._is_low_value_domain(getattr(r, "link", ""))]
                if filtered:
                    return filtered[:top_k]
                return results[:top_k]

        return []
    
    
    def _is_low_value_domain(self, url: str) -> bool:
        low = str(url or "").lower()
        blocked = [
            "pinterest.", "behance.", "dribbble.", "99designs.",
            "glossary", "apify.com/", "deprecated",
        ]
        return any(x in low for x in blocked)


    def _key_has_quota(self, api_key: str) -> bool:
        """
        没有单独 quota 查询接口时，先依据进程内失败计数做判断。
        如果某个 key 连续失败很多次，则暂时跳过。
        """
        fail_count = self._disabled_keys.get(api_key, 0)
        return fail_count < 3

    def _mark_key_failed(self, api_key: str) -> None:
        self._disabled_keys[api_key] = self._disabled_keys.get(api_key, 0) + 1

    def _mark_key_success(self, api_key: str) -> None:
        if api_key in self._disabled_keys:
            self._disabled_keys[api_key] = 0

    def _search_tavily_with_key(self, api_key: str, query: str, top_k: int) -> tuple[list[SearchResult], bool]:
        url = "https://api.tavily.com/search"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "query": query,
            "max_results": min(max(1, int(top_k)), 6),
            "search_depth": self.search_depth,
            "topic": self.topic,
            "include_images": False,
            "include_raw_content": self.include_raw_content,
        }

        try:
            with make_http_client(timeout=20) as client:
                resp = client.post(url, headers=headers, json=payload)

                # 常见该 key 当前不可用/额度不足/限流的状态
                if resp.status_code in {401, 402, 403, 429}:
                    self._mark_key_failed(api_key)
                    return [], False

                resp.raise_for_status()
                data = resp.json()

        except httpx.HTTPError:
            self._mark_key_failed(api_key)
            return [], False
        except Exception:
            self._mark_key_failed(api_key)
            return [], False

        results: list[SearchResult] = []
        for item in data.get("results", [])[:top_k]:
            if not isinstance(item, dict):
                continue

            title = str(item.get("title", "") or "")
            link = str(item.get("url", "") or "")
            snippet = str(
                item.get("content", "")
                or item.get("snippet", "")
                or item.get("raw_content", "")
                or ""
            )

            if not link:
                continue

            results.append(
                SearchResult(
                    title=title,
                    link=link,
                    snippet=clean_text_content(snippet, max_chars=800),
                )
            )

        self._mark_key_success(api_key)
        return results, True


class WebpageTool:
    def fetch_text(self, url: str, max_chars: int = 5000) -> str:
        if not url:
            return ""
        try:
            with make_http_client(timeout=20, follow_redirects=True) as client:
                resp = client.get(url)
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            return f"WEB_FETCH_ERROR: {type(exc).__name__}: {exc}"
        except Exception as exc:
            return f"WEB_FETCH_ERROR: {type(exc).__name__}: {exc}"

        content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
        if content_type and content_type not in {"text/html", "application/xhtml+xml"}:
            return f"WEB_FETCH_UNSUPPORTED_CONTENT: {content_type}"

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()

        main_blocks = []
        for selector in ["article", "main"]:
            node = soup.select_one(selector)
            if node is not None:
                txt = " ".join(node.get_text(separator=" ").split())
                if txt:
                    main_blocks.append(txt)

        if not main_blocks:
            paragraphs = []
            for p in soup.find_all("p"):
                txt = " ".join(p.get_text(separator=" ").split())
                if len(txt) >= 60:
                    paragraphs.append(txt)
            if paragraphs:
                paragraphs.sort(key=len, reverse=True)
                main_blocks.append(" ".join(paragraphs[:8]))

        if main_blocks:
            text = max(main_blocks, key=len)
        else:
            text = " ".join(soup.get_text(separator=" ").split())

        return clean_text_content(text, max_chars=max_chars)

    def extract_image_candidates(self, url: str, max_images: int = 4) -> list[ImageCandidate]:
        if not url:
            return []
        try:
            with make_http_client(timeout=20, follow_redirects=True) as client:
                resp = client.get(url)
                resp.raise_for_status()
        except httpx.HTTPError:
            return []
        except Exception:
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        out: list[ImageCandidate] = []

        for img in soup.find_all("img"):
            src = img.get("src")
            if not src:
                continue

            absolute = urljoin(url, src)
            alt_text = clean_text_content((img.get("alt") or ""), max_chars=300)
            title_text = clean_text_content((img.get("title") or ""), max_chars=300)

            caption_text = ""
            figure = img.find_parent("figure")
            if figure is not None:
                figcap = figure.find("figcaption")
                if figcap is not None:
                    caption_text = clean_text_content(figcap.get_text(separator=" "), max_chars=500)

            nearby_text = ""
            parent = img.parent
            if parent is not None:
                nearby_text = clean_text_content(parent.get_text(separator=" "), max_chars=300)

            out.append(
                ImageCandidate(
                    image_url=absolute,
                    alt_text=alt_text,
                    title_text=title_text,
                    caption_text=caption_text,
                    nearby_text=nearby_text,
                    page_url=url,
                )
            )
            if len(out) >= max_images:
                break

        return out

    def extract_image_urls(self, url: str, max_images: int = 4) -> list[str]:
        # 保留兼容旧接口
        return [x.image_url for x in self.extract_image_candidates(url=url, max_images=max_images)]

    # def download_image_as_data_url(self, image_url: str, max_bytes: int = 8 * 1024 * 1024) -> tuple[str, str]:
    #     """
    #     Returns:
    #         (data_url, status)
    #         status in {"ok", "download_failed", "not_image", "too_large", "empty"}
    #     """
    #     if not image_url:
    #         return "", "download_failed"

    #     try:
    #         with make_http_client(timeout=30, follow_redirects=True) as client:
    #             resp = client.get(image_url)
    #             resp.raise_for_status()

    #         content = resp.content
    #         if not content:
    #             return "", "empty"
    #         if len(content) > max_bytes:
    #             return "", "too_large"

    #         content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
    #         if not content_type.startswith("image/"):
    #             return "", "not_image"

    #         return _bytes_to_data_url(content, content_type), "ok"

    #     except httpx.HTTPError:
    #         return "", "download_failed"
    #     except Exception:
    #         return "", "download_failed"


def parse_loose_json_object(text: str, fallback: Any = None) -> Any:
    raw = str(text or "").strip()

    fallback_out = dict(fallback or {}) if isinstance(fallback, dict) else fallback
    if isinstance(fallback_out, dict):
        fallback_out["_raw_model_output"] = raw[:4000]

    if not raw:
        return fallback_out

    def _try_load(candidate: str) -> dict[str, Any] | None:
        s = str(candidate or "").strip()
        if not s:
            return None

        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        repaired = s
        repaired = re.sub(r"\bTrue\b", "true", repaired)
        repaired = re.sub(r"\bFalse\b", "false", repaired)
        repaired = re.sub(r"\bNone\b", "null", repaired)
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

        if repaired.startswith("{") and "'" in repaired and '"' not in repaired:
            repaired = repaired.replace("'", '"')

        try:
            obj = json.loads(repaired)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        return None

    obj = _try_load(raw)
    if obj is not None:
        obj.setdefault("_raw_model_output", raw[:4000])
        return obj

    m = re.search(r"```json\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if m:
        obj = _try_load(m.group(1))
        if obj is not None:
            obj.setdefault("_raw_model_output", raw[:4000])
            return obj

    fenced = re.findall(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    for block in fenced:
        obj = _try_load(block)
        if obj is not None:
            obj.setdefault("_raw_model_output", raw[:4000])
            return obj

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        maybe = raw[start:end + 1]
        obj = _try_load(maybe)
        if obj is not None:
            obj.setdefault("_raw_model_output", raw[:4000])
            return obj

    return fallback_out


class VisionTool:
    """统一输出结构化视觉 JSON。"""

    def __init__(self, vl_client: QwenVLClient) -> None:
        self.vl_client = vl_client
        
    
    def _salvage_json_fragment(self, raw: str, key: str) -> Any:
        text = str(raw or "")
        if not text:
            return None

        pat = re.compile(rf'"{re.escape(key)}"\s*:\s*', flags=re.IGNORECASE)
        m = pat.search(text)
        if not m:
            return None

        start = m.end()
        while start < len(text) and text[start].isspace():
            start += 1
        if start >= len(text):
            return None

        opener = text[start]
        if opener not in "{[":
            return None

        closer = "}" if opener == "{" else "]"
        depth = 0
        in_string = False
        escape = False

        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue

            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    frag = text[start:i + 1]
                    try:
                        return json.loads(frag)
                    except Exception:
                        try:
                            repaired = re.sub(r"\bTrue\b", "true", frag)
                            repaired = re.sub(r"\bFalse\b", "false", repaired)
                            repaired = re.sub(r"\bNone\b", "null", repaired)
                            repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
                            return json.loads(repaired)
                        except Exception:
                            return None
        return None

    
    def _salvage_image_analysis(self, text: str, fallback: dict[str, Any]) -> dict[str, Any]:
        out = dict(fallback or {})
        raw = str(text or "")[:4000]
        out["_raw_model_output"] = raw

        visible_facts = self._salvage_json_fragment(text, "visible_facts")
        ocr_facts = self._salvage_json_fragment(text, "ocr_facts")
        hypotheses = self._salvage_json_fragment(text, "hypotheses")
        ambiguities = self._salvage_json_fragment(text, "ambiguities")

        if isinstance(visible_facts, dict) and visible_facts:
            out["visible_facts"] = visible_facts
        if isinstance(ocr_facts, list):
            out["ocr_facts"] = ocr_facts
        if isinstance(hypotheses, list):
            out["hypotheses"] = hypotheses
        if isinstance(ambiguities, list) and ambiguities:
            out["ambiguities"] = ambiguities[:8]

        salvaged = bool(out.get("visible_facts")) or bool(out.get("ocr_facts")) or bool(out.get("hypotheses"))
        out["source_mode"] = "model_parse_salvaged" if salvaged else out.get("source_mode", "model_parse_failed")
        return out
    

    def _extract_json(self, text: str, fallback: dict[str, Any]) -> dict[str, Any]:
        parsed = parse_loose_json_object(text, fallback=None)
        if isinstance(parsed, dict):
            parsed.setdefault("_raw_model_output", str(text or "")[:4000])
            parsed.setdefault("source_mode", parsed.get("source_mode", "model_parse_failed"))
            return parsed

        return self._salvage_image_analysis(text=text, fallback=fallback)

    def _empty_image_analysis(
        self,
        *,
        question: str = "",
        region_hint: str = "",
        fallback_text: str = "",
        source_mode: str = "model_parse_failed",
    ) -> dict[str, Any]:
        ambiguities = []
        if region_hint:
            ambiguities.append(f"局部区域 {region_hint} 未获得稳定视觉结论。")
        else:
            ambiguities.append("未获得稳定视觉结论。")

        hypotheses: list[dict[str, Any]] = []
        if fallback_text.strip():
            hypotheses.append(
                {
                    "label": "proxy_context",
                    "reason": fallback_text.strip()[:500],
                    "confidence": 0.35,
                }
            )
            ambiguities.append("当前结果部分依赖网页/上下文代理描述，而非直接视觉读取。")

        return {
            "visible_facts": {},
            "ocr_facts": [],
            "hypotheses": hypotheses,
            "ambiguities": ambiguities[:6],
            "suggested_followups": [],
            "question": question,
            "region_hint": region_hint,
            "source_mode": source_mode,
        }

    def _empty_ocr(
        self,
        *,
        region_hint: str = "",
        fallback_text: str = "",
        source_mode: str = "model_parse_failed",
    ) -> dict[str, Any]:
        out = {
            "ocr_lines": [],
            "has_text": False,
            "region_hint": region_hint,
            "source_mode": source_mode,
        }
        if fallback_text.strip():
            out["fallback_text"] = fallback_text.strip()[:500]
        return out

    def describe_image(
        self,
        image_ref: str,
        question: str = "",
        region_hint: str = "",
        fallback_text: str = "",
    ) -> dict[str, Any]:
        region_text = f"重点分析区域: {region_hint}\n" if region_hint else ""
        prompt = (
            "假设你是一个专业的多模态视觉证据抽取助手。请只输出一个 JSON 对象，不要输出任何解释性文字。\n"
            "JSON schema:\n"
            "{\n"
            '  "visible_facts": {"key":"value"},\n'
            '  "ocr_facts": [{"text":"", "region":"", "confidence":0.0}],\n'
            '  "hypotheses": [{"label":"", "reason":"", "confidence":0.0}],\n'
            '  "ambiguities": ["..."],\n'
            '  "suggested_followups": ["text","table","chart","diagram","equation","axis","legend","object_region","clock","sign","screen"]\n'
            "}\n"
            "要求：\n"
            "1) visible_facts 只写直接可见且较确定的事实；\n"
            "2) 如果题目涉及比较、计数、相对位置、方向、大小、距离、坐标、图表读取，visible_facts 必须优先写这些关系；\n"
            "3) OCR 内容放到 ocr_facts；\n"
            "4) 不确定推测放到 hypotheses；\n"
            "5) 说不清的地方放到 ambiguities；\n"
            "6) suggested_followups 只给短标签，不超过 3 个；优先返回与题目类型最相关的区域标签；\n"
            "7) 所有 confidence 都在 0 到 1 之间。\n"
            f"问题: {question}\n"
            f"{region_text}"
        )

        raw = self.vl_client.chat_with_image(
            prompt=prompt,
            image_url=image_ref,
            route="vision",
        )

        fallback = self._empty_image_analysis(
            question=question,
            region_hint=region_hint,
            fallback_text=fallback_text,
            source_mode="model_parse_failed",
        )
        fallback["_raw_model_output"] = str(raw or "")[:4000]
        if fallback_text.strip():
            fallback["_fallback_text_used"] = fallback_text.strip()[:800]

        parsed = self._extract_json(raw, fallback=fallback)

        if not isinstance(parsed.get("visible_facts"), dict):
            parsed["visible_facts"] = {}
        if not isinstance(parsed.get("ocr_facts"), list):
            parsed["ocr_facts"] = []
        if not isinstance(parsed.get("hypotheses"), list):
            parsed["hypotheses"] = []
        if not isinstance(parsed.get("ambiguities"), list):
            parsed["ambiguities"] = []
        if not isinstance(parsed.get("suggested_followups"), list):
            parsed["suggested_followups"] = []

        parsed["question"] = question
        parsed["region_hint"] = region_hint
        parsed.setdefault("source_mode", "vision_model")
        return parsed

    def describe_image_or_fallback(
        self,
        image_ref: str,
        question: str = "",
        region_hint: str = "",
        fallback_text: str = "",
    ) -> dict[str, Any]:
        if image_ref:
            return self.describe_image(
                image_ref=image_ref,
                question=question,
                region_hint=region_hint,
                fallback_text=fallback_text,
            )

        return self._empty_image_analysis(
            question=question,
            region_hint=region_hint,
            fallback_text=fallback_text,
            source_mode="fallback_proxy",
        )

    def ocr_image(
        self,
        image_ref: str,
        region_hint: str = "",
        fallback_text: str = "",
    ) -> dict[str, Any]:
        region_text = f"重点 OCR 区域: {region_hint}\n" if region_hint else ""
        prompt = (
            "假设你是一个专业的 OCR 结构化抽取助手。请只输出一个 JSON 对象，不要输出任何解释性文字。\n"
            "JSON schema:\n"
            "{\n"
            '  "ocr_lines": [{"text":"", "region":"", "confidence":0.0}],\n'
            '  "has_text": true\n'
            "}\n"
            "要求：\n"
            "1) 只提取能明确辨认的文字；\n"
            "2) region 用短标签；\n"
            "3) 如果没有文字，返回 ocr_lines=[] 且 has_text=false。\n"
            f"{region_text}"
        )

        raw = self.vl_client.chat_with_image(
            prompt=prompt,
            image_url=image_ref,
            temperature=0.1,
            route="vision",
        )

        fallback = self._empty_ocr(
            region_hint=region_hint,
            fallback_text=fallback_text,
            source_mode="model_parse_failed",
        )
        fallback["_raw_model_output"] = str(raw or "")[:4000]
        if fallback_text.strip():
            fallback["_fallback_text_used"] = fallback_text.strip()[:800]

        parsed = self._extract_json(raw, fallback=fallback)

        if not isinstance(parsed.get("ocr_lines"), list):
            parsed["ocr_lines"] = []
        parsed["has_text"] = bool(parsed.get("has_text", bool(parsed["ocr_lines"])))
        parsed["region_hint"] = region_hint
        parsed.setdefault("source_mode", "vision_model")
        return parsed

    def ocr_image_or_fallback(
        self,
        image_ref: str,
        region_hint: str = "",
        fallback_text: str = "",
    ) -> dict[str, Any]:
        if image_ref:
            return self.ocr_image(
                image_ref=image_ref,
                region_hint=region_hint,
                fallback_text=fallback_text,
            )
        return self._empty_ocr(
            region_hint=region_hint,
            fallback_text=fallback_text,
            source_mode="fallback_proxy",
        )


class LexicalGraphTool:
    """Knowledge-enhanced retrieval for DeepResearch using lexical-graph ideas."""

    def retrieve_from_documents(self, question: str, documents: list[tuple[str, str]], top_k: int = 5) -> list[str]:
        index = build_graph_from_corpus(documents)
        retriever = LexicalGraphRetriever(index)
        results = retriever.retrieve(question=question, top_k=top_k)
        return [
            f"[{i+1}] source={r.source} score={r.score:.3f} reasons={','.join(r.reasons)} text={r.text[:500]}"
            for i, r in enumerate(results)
        ]


class ToolRouter:
    """Unified tool interface to support iterative agent tool-use."""

    def __init__(
        self,
        search_tool: SearchTool,
        webpage_tool: WebpageTool,
        vision_tool: VisionTool,
        lexical_graph_tool: LexicalGraphTool,
        byokg_provider: BYOKGRAGProvider,
        current_image_ref: str | None = None,
        allow_web_search_tool: bool | None = None,
    ) -> None:
        self.search_tool = search_tool
        self.webpage_tool = webpage_tool
        self.vision_tool = vision_tool
        self.lexical_graph_tool = lexical_graph_tool
        self.byokg_provider = byokg_provider
        self.current_image_ref = str(current_image_ref or "").strip() or None
        self.allow_web_search_tool = (
            bool(getattr(settings, "enable_web_search_tool", True))
            if allow_web_search_tool is None
            else bool(allow_web_search_tool)
        )

    @staticmethod
    def _web_tool_names() -> set[str]:
        return {
            "search_web",
            "open_webpage",
            "extract_images",
        }

    def _web_search_tool_enabled(self) -> bool:
        return bool(self.allow_web_search_tool) and bool(getattr(settings, "enable_web_search_tool", True))
    
    def set_current_image_ref(self, image_ref: str | None) -> None:
        self.current_image_ref = str(image_ref or "").strip() or None

    
    @staticmethod
    def _is_runtime_image_placeholder(value: str) -> bool:
        v = str(value or "").strip().lower()
        if v in {
            "",
            "current_image",
            "current_image_ref",
            "image_ref",
            "image_url",
            "this_image",
            "same_image",
            "local_image",
            "resolved_local_image",
            "local_image_file",   
        }:
            return True

        # MMMU / MMMU-Pro / agent 过程中可能生成的图片槽位占位符
        if re.fullmatch(r"<\s*image\s*\d+\s*>", v):
            return True
        if re.fullmatch(r"image[\s_]+\d+", v):
            return True
        if re.fullmatch(r"image_ref[\s_]*\d+", v):   
            return True

        return False

    def _resolve_runtime_image_ref(self, raw_ref: str) -> str:
        ref = str(raw_ref or "").strip()
        if self._is_runtime_image_placeholder(ref):
            return str(self.current_image_ref or "").strip()
        return ref
    
    @staticmethod
    def _empty_visual_tool_output(
        tool_name: str,
        question: str = "",
        region_hint: str = "",
    ) -> str:
        if tool_name in {"analyze_image", "analyze_image_region"}:
            return json.dumps(
                {
                    "visible_facts": {},
                    "ocr_facts": [],
                    "hypotheses": [],
                    "ambiguities": ["no_image_input"],
                    "suggested_followups": [],
                    "question": question,
                    "region_hint": region_hint,
                    "source_mode": "no_image_input",
                    "skipped": True,
                },
                ensure_ascii=False,
            )

        return json.dumps(
            {
                "ocr_lines": [],
                "has_text": False,
                "region_hint": region_hint,
                "source_mode": "no_image_input",
                "skipped": True,
            },
            ensure_ascii=False,
        )

    def _normalize_visual_args(self, args: dict[str, Any]) -> dict[str, Any]:
        out = dict(args or {})

        raw_image_ref = str(out.get("image_ref", "")).strip()
        raw_image_url = str(out.get("image_url", "")).strip()
        raw = raw_image_ref or raw_image_url

        resolved = self._resolve_runtime_image_ref(raw)

        if resolved:
            if raw and raw != resolved:
                out["_raw_image_ref"] = raw
                out["_resolved_from_placeholder"] = True

            out["image_ref"] = resolved

            # 只在原始入参本来就是远程 URL 时才保留 image_url
            if raw_image_url and raw_image_url == resolved and (
                resolved.startswith("http://") or resolved.startswith("https://")
            ):
                out["image_url"] = resolved
            else:
                out.pop("image_url", None)

        return out
    
    def _infer_route_mode(self, args: dict[str, Any] | None = None) -> str:
        args = args or {}

        # 这些 key 后面在 eval_vqa.py 里任选其一传进来都能识别
        candidate_keys = [
            "route_mode",
            "rag_mode",
            "executor_mode",
            "workflow_mode",
            "retrieval_mode",
            "pipeline_mode",
        ]

        for key in candidate_keys:
            value = str(args.get(key, "") or "").strip().lower()
            if not value:
                continue

            if value in {"light_rag", "light", "proxy_graph"}:
                return "light_rag"

            if value in {"full_rag", "full", "agentic_rag"}:
                return "full_rag"

        return ""


    def _normalize_backend_for_route(self, backend: str, args: dict[str, Any] | None = None) -> str:
        args = args or {}
        backend = str(backend or "").strip().lower()
        route_mode = self._infer_route_mode(args)

        alias_map = {
            "default": "auto",
            "kg": "auto",
            "lexical_graph": "light_rag",
        }
        backend = alias_map.get(backend, backend)

        # 显式 light_rag：优先锁到本地代理图谱
        if route_mode == "light_rag":
            if backend in {"", "auto", "graph", "qianfan", "hybrid"}:
                return "light_rag"
            return backend

        # 显式 full_rag：不要再被旧的 light/proxy backend 卡死
        if route_mode in {"full_rag", "agentic_rag"}:
            if backend in {"", "light_rag", "proxy_graph"}:
                return "auto"
            return backend

        if backend in {"proxy_graph", "light_rag", "dbpedia", "graph", "qianfan", "hybrid", "auto"}:
            return backend

        return backend or "auto"
    
    
    def _build_effective_byokg_provider(self, args: dict[str, Any]) -> BYOKGRAGProvider:
        args = dict(args or {})

        backend = str(args.get("retrieval_backend", "") or "").strip().lower()
        qianfan_api_key = str(args.get("qianfan_api_key", "") or "").strip()
        qianfan_kb_ids = args.get("qianfan_kb_ids")
        dbpedia_endpoint = str(args.get("dbpedia_endpoint", "") or "").strip()
        dbpedia_timeout = int(args.get("dbpedia_timeout")) if args.get("dbpedia_timeout") is not None else int(
            getattr(settings, "dbpedia_timeout", 12)
        )
        dbpedia_top_k = int(args.get("dbpedia_top_k")) if args.get("dbpedia_top_k") is not None else int(
            getattr(settings, "dbpedia_top_k", 6)
        )

        backend = self._normalize_backend_for_route(backend=backend, args=args)

        if (
            not self._web_search_tool_enabled()
            and bool(getattr(settings, "strict_no_external_kg_when_web_disabled", False))
            and backend in {"", "auto", "hybrid", "qianfan", "dbpedia"}
        ):
            backend = "light_rag"
            qianfan_api_key = ""
            qianfan_kb_ids = []
            dbpedia_endpoint = ""
            args["allow_kb_augmentation"] = False
            args["_external_kg_blocked_by_strict_no_web"] = True
        strict_external_blocked = bool(args.get("_external_kg_blocked_by_strict_no_web"))
        
        # 没有单次覆盖时，直接复用已有 provider
        no_override = (
            self.byokg_provider is not None
            and not strict_external_blocked
            and not qianfan_api_key
            and not qianfan_kb_ids
            and not dbpedia_endpoint
            and args.get("dbpedia_timeout") is None
            and args.get("dbpedia_top_k") is None
        )

        if no_override:
            base = self.byokg_provider
            if backend and backend != getattr(base, "retrieval_backend", ""):
                return BYOKGRAGProvider(
                    linker=base.linker,
                    strategy_linker=base.strategy_linker,
                    cypher_dry_run=base.validator.dry_run,
                    cypher_executor=base.cypher_executor,
                    max_refine_rounds=base.max_refine_rounds,
                    max_query_results=base.max_query_results,
                    retrieval_backend=backend,
                    qianfan_api_key=base.qianfan_api_key,
                    qianfan_kb_ids=base.qianfan_kb_ids,
                    qianfan_search_url=base.qianfan_search_url,
                    qianfan_timeout=base.qianfan_timeout,
                    qianfan_top_k=base.qianfan_top_k,
                    proxy_top_k=base.proxy_top_k,
                    proxy_max_paths=base.proxy_max_paths,
                    dbpedia_endpoint=base.dbpedia_endpoint,
                    dbpedia_timeout=base.dbpedia_timeout,
                    dbpedia_top_k=base.dbpedia_top_k,
                )
            return base

        if self.byokg_provider is not None:
            base = self.byokg_provider
            return BYOKGRAGProvider(
                linker=base.linker,
                strategy_linker=base.strategy_linker,
                cypher_dry_run=base.validator.dry_run,
                cypher_executor=base.cypher_executor,
                max_refine_rounds=base.max_refine_rounds,
                max_query_results=base.max_query_results,
                retrieval_backend=backend or base.retrieval_backend,
                qianfan_api_key=qianfan_api_key if strict_external_blocked else (qianfan_api_key or base.qianfan_api_key),
                qianfan_kb_ids=qianfan_kb_ids if strict_external_blocked else (qianfan_kb_ids or base.qianfan_kb_ids),
                qianfan_search_url=base.qianfan_search_url,
                qianfan_timeout=base.qianfan_timeout,
                qianfan_top_k=base.qianfan_top_k,
                proxy_top_k=base.proxy_top_k,
                proxy_max_paths=base.proxy_max_paths,
                dbpedia_endpoint=dbpedia_endpoint if strict_external_blocked else (dbpedia_endpoint or base.dbpedia_endpoint),
                dbpedia_timeout=dbpedia_timeout,
                dbpedia_top_k=dbpedia_top_k,
            )

        return BYOKGRAGProvider(
            retrieval_backend=backend or "auto",
            qianfan_api_key=qianfan_api_key,
            qianfan_kb_ids=qianfan_kb_ids,
            dbpedia_endpoint=dbpedia_endpoint,
            dbpedia_timeout=dbpedia_timeout,
            dbpedia_top_k=dbpedia_top_k,
        )
    
    def run(self, tool_name: str, args: dict) -> ToolObservation:
        args = dict(args or {})
        
        if tool_name in self._web_tool_names() and not self._web_search_tool_enabled():
            return ToolObservation(
                tool_name=tool_name,
                input_data=args,
                output_data="SKIPPED_WEB_SEARCH_DISABLED",
            )

        visual_tools = {
            "extract_images",
            "analyze_image",
            "analyze_image_region",
            "ocr_image",
            "ocr_image_region",
        }

        if tool_name in visual_tools - {"extract_images"}:
            incoming_image_ref = str(args.get("image_ref", "") or args.get("image_url", "")).strip()
            args = self._normalize_visual_args(args)

            resolved_image_ref = str(args.get("image_ref", "") or args.get("image_url", "")).strip()
            question = str(args.get("question", "")).strip()
            region_hint = str(args.get("region_hint", "")).strip() or "relevant_region"

            if not resolved_image_ref:
                return ToolObservation(
                    tool_name=tool_name,
                    input_data={**args, "image_ref_kind": "no_image_input"},
                    output_data=self._empty_visual_tool_output(
                        tool_name=tool_name,
                        question=question,
                        region_hint=region_hint,
                    ),
                )

            allow_external_image_ref = bool(args.get("allow_external_image_ref", False))
            has_question_image = bool(str(self.current_image_ref or "").strip())

            if (
                not has_question_image
                and incoming_image_ref
                and not self._is_runtime_image_placeholder(incoming_image_ref)
                and not allow_external_image_ref
            ):
                return ToolObservation(
                    tool_name=tool_name,
                    input_data={**args, "image_ref_kind": "rejected_external_image_ref_no_question_image"},
                    output_data=self._empty_visual_tool_output(
                        tool_name=tool_name,
                        question=question,
                        region_hint=region_hint,
                    ),
                )

        # 无图题直接禁止 extract_images
        if tool_name == "extract_images" and not str(self.current_image_ref or "").strip():
            return ToolObservation(
                tool_name=tool_name,
                input_data=args,
                output_data="SKIPPED_NO_IMAGE_INPUT",
            )

        if tool_name == "search_web":
            query = str(args.get("query", "")).strip()
            top_k = _safe_int(
                args.get("top_k", settings.max_snippets_per_round),
                default=5,
                low=1,
                high=10,
            )
            results = self.search_tool.search(query=query, top_k=top_k)
            compact = "\n".join(
                [f"[{i+1}] {r.title} | {r.link} | {r.snippet}" for i, r in enumerate(results)]
            )
            return ToolObservation(tool_name=tool_name, input_data=args, output_data=compact)

        if tool_name == "open_webpage":
            url = str(args.get("url", "")).strip()
            content = self.webpage_tool.fetch_text(url=url)
            return ToolObservation(tool_name=tool_name, input_data=args, output_data=content)

        if tool_name == "extract_images":
            url = str(args.get("url", "")).strip()
            candidates = self.webpage_tool.extract_image_candidates(url=url)

            compact_rows = []
            for i, c in enumerate(candidates, start=1):
                compact_rows.append(
                    json.dumps(
                        {
                            "rank": i,
                            "image_url": c.image_url,
                            "alt_text": c.alt_text,
                            "title_text": c.title_text,
                            "caption_text": c.caption_text,
                            "nearby_text": c.nearby_text,
                        },
                        ensure_ascii=False,
                    )
                )

            return ToolObservation(
                tool_name=tool_name,
                input_data=args,
                output_data="\n".join(compact_rows),
            )

        if tool_name == "retrieve_lexical_graph":
            question = str(args.get("question", "")).strip()
            docs = args.get("documents", [])
            top_k = _safe_int(
                args.get("top_k", settings.max_snippets_per_round),
                default=5,
                low=1,
                high=10,
            )
            normalized_docs: list[tuple[str, str]] = []
            if isinstance(docs, list):
                for d in docs:
                    if isinstance(d, dict):
                        source = str(d.get("source", "unknown"))
                        text = str(d.get("text", ""))
                        if text:
                            normalized_docs.append((source, text))
            out = self.lexical_graph_tool.retrieve_from_documents(
                question=question,
                documents=normalized_docs,
                top_k=top_k,
            )
            return ToolObservation(tool_name=tool_name, input_data=args, output_data="\n".join(out))

        if tool_name == "analyze_image":
            image_ref = str(args.get("image_ref", "") or args.get("image_url", "")).strip()
            question = str(args.get("question", "")).strip() or "提取图片关键证据"
            fallback_text = str(args.get("fallback_text", "")).strip()

            prepared_ref, ref_kind = _prepare_vl_image_ref(
                image_ref=image_ref,
                bucket=str(args.get("image_bucket", "generic")).strip() or "generic",
            )
            if not prepared_ref:
                return ToolObservation(
                    tool_name=tool_name,
                    input_data={**args, "image_ref_kind": ref_kind or "no_image_input"},
                    output_data=self._empty_visual_tool_output(
                        tool_name=tool_name,
                        question=question,
                        region_hint="",
                    ),
                )

            result = self.vision_tool.describe_image(
                image_ref=prepared_ref,
                question=question,
                fallback_text=fallback_text,
            )
            return ToolObservation(
                tool_name=tool_name,
                input_data={**args, "image_ref_kind": ref_kind},
                output_data=json.dumps(result, ensure_ascii=False),
            )

        if tool_name == "analyze_image_region":
            image_ref = str(args.get("image_ref", "") or args.get("image_url", "")).strip()
            question = str(args.get("question", "")).strip() or "提取图片局部关键证据"
            region_hint = str(args.get("region_hint", "")).strip() or "relevant_region"
            fallback_text = str(args.get("fallback_text", "")).strip()

            prepared_ref, ref_kind = _prepare_vl_image_ref(
                image_ref=image_ref,
                bucket=str(args.get("image_bucket", "generic")).strip() or "generic",
            )
            if not prepared_ref:
                return ToolObservation(
                    tool_name=tool_name,
                    input_data={**args, "image_ref_kind": ref_kind or "no_image_input"},
                    output_data=self._empty_visual_tool_output(
                        tool_name=tool_name,
                        question=question,
                        region_hint=region_hint,
                    ),
                )

            result = self.vision_tool.describe_image(
                image_ref=prepared_ref,
                question=question,
                region_hint=region_hint,
                fallback_text=fallback_text,
            )
            return ToolObservation(
                tool_name=tool_name,
                input_data={**args, "image_ref_kind": ref_kind},
                output_data=json.dumps(result, ensure_ascii=False),
            )

        if tool_name == "ocr_image":
            image_ref = str(args.get("image_ref", "") or args.get("image_url", "")).strip()
            fallback_text = str(args.get("fallback_text", "")).strip()

            prepared_ref, ref_kind = _prepare_vl_image_ref(
                image_ref=image_ref,
                bucket=str(args.get("image_bucket", "generic")).strip() or "generic",
            )
            if not prepared_ref:
                return ToolObservation(
                    tool_name=tool_name,
                    input_data={**args, "image_ref_kind": ref_kind or "no_image_input"},
                    output_data=self._empty_visual_tool_output(
                        tool_name=tool_name,
                        question="",
                        region_hint="",
                    ),
                )

            result = self.vision_tool.ocr_image(
                image_ref=prepared_ref,
                fallback_text=fallback_text,
            )
            return ToolObservation(
                tool_name=tool_name,
                input_data={**args, "image_ref_kind": ref_kind},
                output_data=json.dumps(result, ensure_ascii=False),
            )

        if tool_name == "ocr_image_region":
            image_ref = str(args.get("image_ref", "") or args.get("image_url", "")).strip()
            region_hint = str(args.get("region_hint", "")).strip() or "relevant_region"
            fallback_text = str(args.get("fallback_text", "")).strip()

            prepared_ref, ref_kind = _prepare_vl_image_ref(
                image_ref=image_ref,
                bucket=str(args.get("image_bucket", "generic")).strip() or "generic",
            )
            if not prepared_ref:
                return ToolObservation(
                    tool_name=tool_name,
                    input_data={**args, "image_ref_kind": ref_kind or "no_image_input"},
                    output_data=self._empty_visual_tool_output(
                        tool_name=tool_name,
                        question="",
                        region_hint=region_hint,
                    ),
                )

            result = self.vision_tool.ocr_image(
                image_ref=prepared_ref,
                region_hint=region_hint,
                fallback_text=fallback_text,
            )
            return ToolObservation(
                tool_name=tool_name,
                input_data={**args, "image_ref_kind": ref_kind},
                output_data=json.dumps(result, ensure_ascii=False),
            )

        if tool_name == "retrieve_graph_evidence":
            question = str(args.get("question", "")).strip()
            schema = args.get("schema", {})
            graph_context = args.get("graph_context")
            documents = args.get("documents", [])

            if not isinstance(schema, dict):
                schema = {}

            normalized_documents: list[dict[str, str]] = []
            if isinstance(documents, list):
                for d in documents:
                    if not isinstance(d, dict):
                        continue
                    source = str(d.get("source", "")).strip() or "proxy_doc"
                    text = str(d.get("text", "")).strip()
                    if text:
                        normalized_documents.append({"source": source, "text": text})

            requested_backend = self._normalize_backend_for_route(
                backend=str(args.get("retrieval_backend", "") or ""),
                args=args,
            )
            if (
                not self._web_search_tool_enabled()
                and bool(getattr(settings, "strict_no_external_kg_when_web_disabled", False))
                and requested_backend in {"", "auto", "hybrid", "qianfan", "dbpedia"}
            ):
                args["retrieval_backend"] = "light_rag"
                args["allow_kb_augmentation"] = False
                args["_external_kg_blocked_by_strict_no_web"] = True

            provider = self._build_effective_byokg_provider(args)

            query_variants = args.get("query_variants", [])
            if not isinstance(query_variants, list):
                query_variants = []

            allow_kb_augmentation = bool(args.get("allow_kb_augmentation", False))
            if args.get("_external_kg_blocked_by_strict_no_web"):
                allow_kb_augmentation = False

            try:
                result = provider.retrieve_graph_evidence(
                    question=question,
                    schema=schema,
                    graph_context=str(graph_context) if graph_context is not None else None,
                    documents=normalized_documents or None,
                    query_variants=[str(x).strip() for x in query_variants if str(x).strip()],
                    allow_kb_augmentation=allow_kb_augmentation,
                )
            except Exception as exc:
                result = {
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
                        "selected_strategy": "kg_exception",
                        "refinement_history": [
                            {
                                "round": 1,
                                "ok": False,
                                "errors": ["kg_exception", type(exc).__name__, str(exc)],
                            }
                        ],
                        "effective_backend": getattr(provider, "retrieval_backend", ""),
                        "allow_kb_augmentation": allow_kb_augmentation,
                    },
                    "confidence": 0.0,
                    "coverage": 0.0,
                    "confidence_type": "retrieval",
                    "confidence_label": "retrieval_confidence_not_final_answer",
                    "tool_error": {
                        "code": "TOOL_ERROR",
                        "stage": "retrieve_graph_evidence.provider",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                }

            if isinstance(result, dict):
                result.setdefault("confidence_type", "retrieval")
                result.setdefault("confidence_label", "retrieval_confidence_not_final_answer")

                linking_artifacts = result.get("linking_artifacts", {})
                if isinstance(linking_artifacts, dict):
                    linking_artifacts.setdefault("confidence_type", "retrieval")
                    linking_artifacts.setdefault("confidence_label", "retrieval_confidence_not_final_answer")
                    linking_artifacts.setdefault("effective_backend", getattr(provider, "retrieval_backend", ""))
                    linking_artifacts.setdefault("allow_kb_augmentation", allow_kb_augmentation)
                    if args.get("_external_kg_blocked_by_strict_no_web"):
                        linking_artifacts["external_kg_blocked_by_strict_no_web"] = True

            debug_input = {
                **args,
                "_resolved_backend": getattr(provider, "retrieval_backend", ""),
                "_route_mode": self._infer_route_mode(args),
                "_confidence_type": "retrieval",
                "_document_count": len(normalized_documents),
                "_allow_kb_augmentation": allow_kb_augmentation,
                "_external_kg_blocked_by_strict_no_web": bool(args.get("_external_kg_blocked_by_strict_no_web")),
            }

            return ToolObservation(
                tool_name=tool_name,
                input_data=debug_input,
                output_data=json.dumps(result, ensure_ascii=False),
            )

        raise ValueError(f"Unsupported tool: {tool_name}")
