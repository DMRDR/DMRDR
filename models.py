from __future__ import annotations

import base64
import hashlib
import json
import re
import mimetypes
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


import httpx
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from config import settings


def _env_bool(name: str, default: bool = False) -> bool:
    value = str(os.getenv(name, "")).strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}

def _http_verify_value() -> bool | str:
    ca_bundle = os.getenv("HTTP_CA_BUNDLE", "").strip()
    if ca_bundle:
        return ca_bundle

    raw = os.getenv("HTTP_VERIFY", "").strip().lower()
    if not raw:
        return True
    return raw in {"1", "true", "yes", "on"}


def _make_http_client(timeout: float = 30.0, follow_redirects: bool = True) -> httpx.Client:
    return httpx.Client(
        timeout=timeout,
        follow_redirects=follow_redirects,
        verify=_http_verify_value(),
        trust_env=True,
        headers={"User-Agent": "DeepResearch-VLClient/1.0"},
    )


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


def _to_file_url(path: Path) -> str:
    return f"file://{path.resolve().as_posix()}"


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


def _guess_image_mime(url: str, content_type: str | None = None) -> str:
    if content_type and content_type.startswith("image/"):
        return content_type.split(";")[0].strip()
    mime, _ = mimetypes.guess_type(url)
    if mime and mime.startswith("image/"):
        return mime
    return "image/jpeg"


def _stage_local_image_to_allowed_root(src_path: Path, bucket: str = "generic") -> str:
    allowed_root = _get_allowed_local_media_root()
    if allowed_root is None:
        return _to_file_url(src_path)

    src = src_path.expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Local image not found: {src}")

    content = src.read_bytes()
    if not content:
        raise ValueError(f"Empty image file: {src}")

    ext = src.suffix.lower().strip() or ".jpg"
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        ext = ".jpg"

    target_dir = allowed_root / bucket / "local_files"
    target_dir.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha1(content).hexdigest()
    out_path = target_dir / f"{digest}{ext}"
    if not out_path.exists():
        out_path.write_bytes(content)

    return _to_file_url(out_path)


def _download_remote_image_to_allowed_root(
    image_url: str,
    bucket: str = "generic",
    timeout: float = 30.0,
    max_bytes: int = 15 * 1024 * 1024,
) -> str:
    allowed_root = _get_allowed_local_media_root()
    if allowed_root is None:
        raise ValueError(
            "Remote image requires VLLM_ALLOWED_LOCAL_MEDIA_ROOT to be set "
            "and allowed by vLLM --allowed-local-media-path."
        )

    with _make_http_client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(image_url)
        resp.raise_for_status()

    content = resp.content
    if not content:
        raise ValueError("Remote image is empty")
    if len(content) > max_bytes:
        raise ValueError("Remote image is too large")

    content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
    mime = _guess_image_mime(image_url, content_type)
    if not mime.startswith("image/"):
        raise ValueError(f"Remote URL is not an image: {mime}")

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

    return _to_file_url(out_path)

def _bytes_to_data_url(content: bytes, mime: str = "image/jpeg") -> str:
    mime = str(mime or "image/jpeg").strip().lower()
    if not mime.startswith("image/"):
        mime = "image/jpeg"
    b64 = base64.b64encode(content).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _guess_local_image_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if mime and mime.startswith("image/"):
        return mime
    return "image/jpeg"


class QwenVLClient:
    """Client: local/vLLM + DashScope"""

    def __init__(self) -> None:
        self.client = OpenAI(
            base_url=settings.api_base,
            api_key=settings.api_key,
            timeout=settings.api_timeout_s,
        )
        self.model = self._resolve_model_name(settings.model_name)

        self.ds_client: OpenAI | None = None
        self.ds_text_model = str(getattr(settings, "dashscope_text_model", "")).strip()
        self.ds_vision_model = str(getattr(settings, "dashscope_vision_model", "")).strip()

        ds_base = str(getattr(settings, "dashscope_api_base", "") or "").strip()
        ds_key = str(getattr(settings, "dashscope_api_key", "") or "").strip()
        if ds_base and ds_key:
            self.ds_client = OpenAI(
                base_url=ds_base,
                api_key=ds_key,
                timeout=settings.api_timeout_s,
            )

        self.xa_client: OpenAI | None = None
        self.xa_text_model = str(getattr(settings, "xiaoai_text_model", "")).strip()
        self.xa_vision_model = str(getattr(settings, "xiaoai_vision_model", "")).strip()

        xa_base = str(getattr(settings, "xiaoai_api_base", "") or "").strip()
        xa_key = str(getattr(settings, "xiaoai_api_key", "") or "").strip()
        if xa_base and xa_key:
            self.xa_client = OpenAI(
                base_url=xa_base,
                api_key=xa_key,
                timeout=settings.api_timeout_s,
            )


    def _resolve_model_name(self, preferred: str) -> str:
        preferred = str(preferred or "").strip()
        if not preferred:
            return ""

        try:
            model_list = self.client.models.list()
            available = [m.id for m in model_list.data if getattr(m, "id", None)]
        except Exception:
            return preferred

        if not available:
            return preferred
        if preferred in available:
            return preferred

        preferred_tail = preferred.split("/")[-1].lower()
        candidates = [
            m
            for m in available
            if preferred_tail in m.lower() or m.lower() in preferred_tail or "qwen" in m.lower()
        ]
        chosen = candidates[0] if candidates else available[0]
        print(
            f"[QwenVLClient] configured model '{preferred}' not found; fallback to '{chosen}'. "
            f"available={available}"
        )
        return chosen
    
    @staticmethod
    def _normalize_backend_name(value: Any, default: str = "local") -> str:
        raw = str(value or "").strip().lower().replace("-", "_")
        if not raw:
            return default

        alias = {
            "none": "local",
            "local": "local",
            "vllm": "local",
            "local_vllm": "local",
            "dash": "dashscope",
            "ds": "dashscope",
            "dashscope": "dashscope",
            "aliyun": "dashscope",
            "xiaoai": "xiaoai",
            "xiaoai_plus": "xiaoai",
            "xiaoai.plus": "xiaoai",
            "gpt": "xiaoai",
            "gpt_5_1": "xiaoai",
        }
        return alias.get(raw, default)

    def _route_backend_name(self, *, route: str, with_image: bool = False) -> str:
        route = str(route or "default").strip().lower()

        if with_image or route == "vision":
            explicit = str(getattr(settings, "vision_backend", "") or "").strip()
            if explicit:
                return self._normalize_backend_name(explicit, default="local")
            if bool(getattr(settings, "vision_use_xiaoai", False)):
                return "xiaoai"
            if bool(getattr(settings, "vision_use_dashscope", False)):
                return "dashscope"
            return "local"

        if route == "planner":
            explicit = str(getattr(settings, "planner_backend", "") or "").strip()
            if explicit:
                return self._normalize_backend_name(explicit, default="local")
            if bool(getattr(settings, "planner_use_xiaoai", False)):
                return "xiaoai"
            if bool(getattr(settings, "planner_use_dashscope", False)):
                return "dashscope"
            return "local"

        if route == "reasoning":
            explicit = str(getattr(settings, "reasoning_backend", "") or "").strip()
            if explicit:
                return self._normalize_backend_name(explicit, default="local")
            if bool(getattr(settings, "reasoning_use_xiaoai", False)):
                return "xiaoai"
            if bool(getattr(settings, "reasoning_use_dashscope", False)):
                return "dashscope"
            return "local"

        return "local"

    def _select_backend(self, *, route: str = "default", with_image: bool = False) -> tuple[OpenAI, str]:
        backend = self._route_backend_name(route=route, with_image=with_image)

        if backend == "xiaoai":
            model = self.xa_vision_model if (with_image or route == "vision") else self.xa_text_model
            if self.xa_client is not None and model:
                return self.xa_client, model
            print(
                f"[QwenVLClient] XIAOAI.PLUS backend requested for route={route}, "
                "but client/model is missing; fallback to local vLLM.",
                flush=True,
            )
            return self.client, self.model

        if backend == "dashscope":
            model = self.ds_vision_model if (with_image or route == "vision") else self.ds_text_model
            if self.ds_client is not None and model:
                return self.ds_client, model
            print(
                f"[QwenVLClient] DashScope backend requested for route={route}, "
                "but client/model is missing; fallback to local vLLM.",
                flush=True,
            )
            return self.client, self.model

        return self.client, self.model

    def _local_image_ref_to_data_url(self, image_ref: str) -> str:
        local_path = _extract_local_path_from_ref(image_ref)
        if local_path is None or not local_path.is_file():
            raise ValueError(f"invalid_local_image_ref: {image_ref}")

        content = local_path.read_bytes()
        if not content:
            raise ValueError(f"empty_local_image_file: {local_path}")

        mime = _guess_local_image_mime(local_path)
        return _bytes_to_data_url(content, mime=mime)
    
    
    def image_ref_to_data_url(self, image_ref: str) -> str:
        raw = str(image_ref or "").strip()
        if not raw:
            raise ValueError("empty_image_ref")
        if raw.startswith("data:image/"):
            return raw
        return self._local_image_ref_to_data_url(raw)
    
    
    def _prepare_image_ref_for_backend(
        self,
        image_ref: str,
        *,
        backend_name: str,
        bucket: str = "generic",
    ) -> tuple[str, str]:
        raw = str(image_ref or "").strip()
        if not raw:
            raise ValueError("empty_image_ref")

        # 外部 OpenAI 兼容接口，当前包括 DashScope / XIAOAI.PLUS：
        # - 远程图：直接传 URL
        # - 本地图 / file://：转 Base64 data URL
        # - 已经是 data URL：原样传
        if backend_name in {"dashscope", "xiaoai"}:
            if raw.startswith("data:image/"):
                return raw, "data_url"

            if raw.startswith("http://") or raw.startswith("https://"):
                return raw, "remote_http_url"

            return self._local_image_ref_to_data_url(raw), "local_file_as_data_url"

        # 本地 vLLM：
        # 继续沿用现有 allowed root + file:// staging 逻辑
        prepared = self._normalize_image_ref_for_vllm(
            image_ref=raw,
            bucket=bucket,
        )
        return prepared, "vllm_file_url"
    
    
    def _route_max_tokens(self, route: str, with_image: bool = False) -> int:
        route = str(route or "default").strip().lower()

        route_caps = {
            "default": 320,
            "planner": int(getattr(settings, "planner_api_max_tokens", 384) or 384),
            "reasoning": int(getattr(settings, "reasoning_api_max_tokens", 640) or 640),
            "vision": int(getattr(settings, "vision_api_max_tokens", 768) or 768),
            "verify": int(getattr(settings, "verify_api_max_tokens", 384) or 384),
            "synthesize": int(getattr(settings, "synthesize_api_max_tokens", 640) or 640),
        }

        cap = int(route_caps.get(route, 320) or 320)

        try:
            configured = int(getattr(settings, "api_max_tokens", 256) or 256)
        except Exception:
            configured = 256

        if with_image and route != "vision":
            cap = min(cap, max(384, configured))

        return max(1, min(max(configured, cap), cap))


    def _clip_text_by_chars(self, text: Any, max_chars: int) -> str:
        s = str(text or "")
        if max_chars <= 0:
            return ""
        return s if len(s) <= max_chars else s[:max_chars]


    def _char_budget_for_route(self, route: str, with_image: bool = False) -> int:
        route = str(route or "default").strip().lower()

        budgets = {
            "default": 18000,
            "planner": 12000,
            "reasoning": 14000,
            "vision": 6000,
            "verify": 12000,
            "synthesize": 16000,
        }

        budget = budgets.get(route, 12000)
        if with_image:
            budget = min(budget, 6000)
        return budget


    def _shrink_messages_for_budget(
        self,
        messages: list[dict[str, Any]],
        *,
        route: str,
        with_image: bool = False,
    ) -> list[dict[str, Any]]:
        total_budget = self._char_budget_for_route(route=route, with_image=with_image)
        used = 0
        compact: list[dict[str, Any]] = []

        for msg in messages:
            msg2 = dict(msg)
            content = msg2.get("content")

            if isinstance(content, str):
                remain = max(0, total_budget - used)
                clipped = self._clip_text_by_chars(content, remain)
                msg2["content"] = clipped
                used += len(clipped)

            elif isinstance(content, list):
                new_parts: list[dict[str, Any]] = []
                for part in content:
                    if not isinstance(part, dict):
                        continue

                    if part.get("type") == "text":
                        remain = max(0, total_budget - used)
                        clipped = self._clip_text_by_chars(part.get("text", ""), remain)
                        new_parts.append({"type": "text", "text": clipped})
                        used += len(clipped)
                    else:
                        new_parts.append(part)

                msg2["content"] = new_parts

            compact.append(msg2)

        return compact
    
    
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

        wait_s = base * (2 ** attempt_idx)
        return max(0.5, min(wait_s, cap, 120.0))


    @staticmethod
    def _is_non_retryable_openai_error(exc: Exception) -> bool:
        return isinstance(
            exc,
            (
                NotFoundError,
                AuthenticationError,
                PermissionDeniedError,
                BadRequestError,
            ),
        )


    @staticmethod
    def _is_retryable_openai_error(exc: Exception) -> bool:
        return isinstance(
            exc,
            (
                APIConnectionError,
                APITimeoutError,
                RateLimitError,
                InternalServerError,
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
            ),
        )


    def _create_chat_completion_with_retry(
        self,
        *,
        client: OpenAI,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        debug_stage_prefix: str,
    ):
        last_error: Exception | None = None
        attempts = self._retry_attempts()

        for i in range(attempts):
            try:
                return client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                last_error = exc
                self._debug_print_vllm_io(
                    stage=f"{debug_stage_prefix}_ERROR attempt={i+1}",
                    messages=messages,
                    error=f"{type(exc).__name__}: {exc}",
                    temperature=temperature,
                )

                if self._is_non_retryable_openai_error(exc):
                    raise

                if i < attempts - 1:
                    time.sleep(self._retry_sleep_s(i))
                    continue
                raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("unexpected_retry_exit")
    
    
    def chat(self, messages: list[dict[str, Any]], temperature: float = 0.1, route: str = "default") -> str:
        shrunk_messages = self._shrink_messages_for_budget(
            messages,
            route=route,
            with_image=False,
        )
        max_tokens = self._route_max_tokens(route=route, with_image=False)
        client, model = self._select_backend(route=route, with_image=False)

        self._debug_print_vllm_io(
            stage="REQUEST",
            messages=shrunk_messages,
            temperature=temperature,
        )

        completion = self._create_chat_completion_with_retry(
            client=client,
            model=model,
            messages=shrunk_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            debug_stage_prefix="REQUEST",
        )
        text = completion.choices[0].message.content or ""

        self._debug_print_vllm_io(
            stage="RESPONSE",
            response_text=text,
            temperature=temperature,
        )
        return text
    
    
    def chat_json(
        self,
        messages: list[dict[str, Any]],
        fallback: Any,
        temperature: float = 0.1,
        route: str = "default",
    ) -> Any:
        raw = self.chat(messages=messages, temperature=temperature, route=route)
        text = str(raw or "").strip()

        def _try_load_json(candidate: str) -> Any:
            candidate = str(candidate or "").strip()
            if not candidate:
                return None

            try:
                return json.loads(candidate)
            except Exception:
                pass

            decoder = json.JSONDecoder()

            # 从任意可能的 JSON 起点尝试 raw_decode
            for i, ch in enumerate(candidate):
                if ch not in "{[":
                    continue
                try:
                    obj, _ = decoder.raw_decode(candidate[i:])
                    return obj
                except Exception:
                    continue

            return None

        if not text:
            return fallback

        # 直接整段解析
        obj = _try_load_json(text)
        if obj is not None:
            return obj

        # 提取 ```json ... ``` / ``` ... ``` 代码块
        fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        for block in fenced_blocks:
            obj = _try_load_json(block)
            if obj is not None:
                return obj

        # 提取最外层对象 {...}
        first_obj = text.find("{")
        last_obj = text.rfind("}")
        if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
            obj = _try_load_json(text[first_obj:last_obj + 1])
            if obj is not None:
                return obj

        # 提取最外层数组 [...]
        first_arr = text.find("[")
        last_arr = text.rfind("]")
        if first_arr != -1 and last_arr != -1 and last_arr > first_arr:
            obj = _try_load_json(text[first_arr:last_arr + 1])
            if obj is not None:
                return obj

        # 最终失败，返回 fallback
        return fallback
    
    def _normalize_image_ref_for_vllm(self, image_ref: str, bucket: str = "generic") -> str:
        raw = str(image_ref or "").strip()
        if not raw:
            raise ValueError("empty_image_ref")

        if raw.startswith("data:image/"):
            raise ValueError("inline_data_url_blocked")

        local_path = _extract_local_path_from_ref(raw)
        if local_path is not None and local_path.is_file():
            return _stage_local_image_to_allowed_root(local_path, bucket=bucket)

        if raw.startswith("http://") or raw.startswith("https://"):
            return _download_remote_image_to_allowed_root(raw, bucket=bucket)

        raise ValueError(f"invalid_image_ref: {raw}")


    def _debug_vllm_enabled(self) -> bool:
        return bool(getattr(settings, "debug_vllm_io", False))


    def _debug_clip(self, text: Any) -> str:
        s = str(text or "")
        limit = int(getattr(settings, "debug_vllm_max_chars", 4000) or 4000)
        return s if len(s) <= limit else s[:limit] + "...<truncated>"


    def _debug_print_vllm_io(
        self,
        *,
        stage: str,
        messages: list[dict[str, Any]] | None = None,
        response_text: str | None = None,
        error: str | None = None,
        temperature: float | None = None,
    ) -> None:
        if not self._debug_vllm_enabled():
            return

        print(f"\n[QwenVLClient::{stage}]")
        if temperature is not None:
            print(f"temperature={temperature}")

        if messages is not None:
            full = bool(getattr(settings, "debug_vllm_full_messages", False))
            if full:
                try:
                    print(json.dumps(messages, ensure_ascii=False, indent=2))
                except Exception:
                    print(self._debug_clip(messages))
            else:
                try:
                    compact = json.dumps(messages, ensure_ascii=False)
                except Exception:
                    compact = str(messages)
                print(self._debug_clip(compact))

        if response_text is not None:
            print("response=" + self._debug_clip(response_text))

        if error is not None:
            print("error=" + self._debug_clip(error))
    
    
    def _backend_name(self, client: Any) -> str:
        if self.ds_client is not None and client is self.ds_client:
            return "dashscope"
        if self.xa_client is not None and client is self.xa_client:
            return "xiaoai"
        if client is self.client:
            return "local_vllm"
        return "unknown_backend"
    

    def chat_with_image(
        self,
        prompt: str,
        image_url: str,
        temperature: float = 0.1,
        image_bucket: str = "generic",
        route: str = "vision",
    ) -> str:
        raw_image_ref = str(image_url or "").strip()
        prepared_image_ref = ""
        prepared_image_ref_kind = ""

        try:
            client, model = self._select_backend(route=route, with_image=True)
            backend_name = self._backend_name(client)

            prepared_image_ref, prepared_image_ref_kind = self._prepare_image_ref_for_backend(
                image_ref=raw_image_ref,
                backend_name=backend_name,
                bucket=image_bucket,
            )

            messages: list[dict[str, Any]] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": prepared_image_ref}},
                    ],
                }
            ]

            shrunk_messages = self._shrink_messages_for_budget(
                messages,
                route=route,
                with_image=True,
            )
            max_tokens = self._route_max_tokens(route=route, with_image=True)

            if self._debug_vllm_enabled():
                print(
                    "[QwenVLClient::IMAGE_REQUEST_META] "
                    f"route={route} "
                    f"chosen_backend={backend_name} "
                    f"chosen_model={model} "
                    f"raw_image_ref={self._debug_clip(raw_image_ref)} "
                    f"prepared_image_ref_kind={prepared_image_ref_kind} "
                    f"prepared_image_ref={self._debug_clip(prepared_image_ref)} "
                    f"max_tokens={max_tokens}"
                )

            self._debug_print_vllm_io(
                stage="IMAGE_REQUEST",
                messages=shrunk_messages,
                temperature=temperature,
            )

            completion = self._create_chat_completion_with_retry(
                client=client,
                model=model,
                messages=shrunk_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                debug_stage_prefix="IMAGE_REQUEST",
            )
            text = completion.choices[0].message.content or ""

            self._debug_print_vllm_io(
                stage="IMAGE_RESPONSE",
                response_text=text,
                temperature=temperature,
            )
            return text

        except Exception as exc:
            self._debug_print_vllm_io(
                stage="IMAGE_ERROR",
                error=(
                    f"route={route} | "
                    f"raw_image_ref={raw_image_ref} | "
                    f"prepared_image_ref_kind={prepared_image_ref_kind} | "
                    f"prepared_image_ref={self._debug_clip(prepared_image_ref)} | "
                    f"{type(exc).__name__}: {exc}"
                ),
                temperature=temperature,
            )
            raise