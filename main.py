from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path

from agents import CriticAgent, PlannerAgent, ResearcherAgent, SynthesizerAgent, VerifierAgent
from byokg_rag import BYOKGRAGProvider
from models import QwenVLClient
from tools import LexicalGraphTool, SearchTool, ToolRouter, VisionTool, WebpageTool
from workflow import DeepResearchWorkflow


def _to_file_url(path: Path) -> str:
    return f"file://{path.resolve().as_posix()}"


def _stage_cli_local_image(image_path: str, media_root_override: str = "") -> str:
    p = Path(image_path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Local image not found: {p}")

    raw_root = str(media_root_override or "").strip() or os.getenv("VLLM_ALLOWED_LOCAL_MEDIA_ROOT", "").strip()
    if not raw_root:
        # 没有限定 root 时，直接返回本地绝对路径；tools.py / models.py 后面会再规范化为 file://
        return str(p)

    media_root = Path(raw_root).expanduser().resolve()
    media_root.mkdir(parents=True, exist_ok=True)

    target_dir = media_root / "cli_images"
    target_dir.mkdir(parents=True, exist_ok=True)

    content = p.read_bytes()
    digest = hashlib.sha1(content).hexdigest()
    ext = p.suffix.lower() or ".jpg"
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        ext = ".jpg"

    dst = target_dir / f"{digest}{ext}"
    if not dst.exists():
        shutil.copy2(p, dst)

    return _to_file_url(dst)


def _prepare_cli_image_ref(
    image_path: str,
    image_url: str,
    vllm_local_media_root: str = "",
) -> str | None:
    local = str(image_path or "").strip()
    remote = str(image_url or "").strip()

    # 本地图优先
    if local:
        return _stage_cli_local_image(local, media_root_override=vllm_local_media_root)

    if remote:
        return remote

    return None


def _assert_runtime_contract() -> None:
    required_qwenvl = [
        "_select_backend",
        "_normalize_image_ref_for_vllm",
        "chat_json",
        "chat_with_image",
    ]
    missing_qwenvl = [name for name in required_qwenvl if not hasattr(QwenVLClient, name)]
    if missing_qwenvl:
        raise RuntimeError(
            "Loaded models.QwenVLClient is not the patched version. "
            f"Missing methods: {missing_qwenvl}"
        )

    required_vision = [
        "describe_image",
        "ocr_image",
    ]
    missing_vision = [name for name in required_vision if not hasattr(VisionTool, name)]
    if missing_vision:
        raise RuntimeError(
            "Loaded tools.VisionTool is not the patched version. "
            f"Missing methods: {missing_vision}"
        )


def _backend_healthcheck(llm: QwenVLClient, image_ref: str | None = None) -> None:
    # 本地默认路由
    local_resp = llm.chat(
        [{"role": "user", "content": "Reply with OK."}],
        temperature=0.0,
        route="default",
    )
    if not str(local_resp or "").strip():
        raise RuntimeError("Local vLLM healthcheck failed: empty response")

    # Planner / Reasoning -> DashScope text
    planner_obj = llm.chat_json(
        [{"role": "user", "content": 'Return {"ok": true}'}],
        fallback={"ok": False},
        temperature=0.0,
        route="planner",
    )
    if not isinstance(planner_obj, dict) or not planner_obj.get("ok", False):
        raise RuntimeError("DashScope planner healthcheck failed")

    reasoning_obj = llm.chat_json(
        [{"role": "user", "content": 'Return {"ok": true}'}],
        fallback={"ok": False},
        temperature=0.0,
        route="reasoning",
    )
    if not isinstance(reasoning_obj, dict) or not reasoning_obj.get("ok", False):
        raise RuntimeError("DashScope reasoning healthcheck failed")

    # Vision -> DashScope vision（有图时才做）
    if image_ref:
        vision_resp = llm.chat_with_image(
            prompt="Describe the image in several short sentences.",
            image_url=image_ref,
            temperature=0.0,
            route="vision",
        )
        if not str(vision_resp or "").strip():
            raise RuntimeError("DashScope vision healthcheck failed: empty response")
        

def build_workflow(retrieval_backend: str = "") -> DeepResearchWorkflow:
    llm = QwenVLClient()

    search_tool = SearchTool()
    webpage_tool = WebpageTool()
    vision_tool = VisionTool(vl_client=llm)
    lexical_graph_tool = LexicalGraphTool()

    backend = str(
        retrieval_backend or getattr(__import__("config").settings, "kg_retrieval_backend", "auto") or "auto"
    ).strip().lower() or "auto"
    byokg_provider = BYOKGRAGProvider(retrieval_backend=backend)

    tool_router = ToolRouter(
        search_tool=search_tool,
        webpage_tool=webpage_tool,
        vision_tool=vision_tool,
        lexical_graph_tool=lexical_graph_tool,
        byokg_provider=byokg_provider,
    )

    planner = PlannerAgent(llm=llm)
    researcher = ResearcherAgent(llm=llm, tool_router=tool_router)
    critic = CriticAgent(llm=llm)
    verifier = VerifierAgent(llm=llm)
    synthesizer = SynthesizerAgent(llm=llm)

    return DeepResearchWorkflow(
        planner=planner,
        researcher=researcher,
        critic=critic,
        verifier=verifier,
        synthesizer=synthesizer,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepResearch with Qwen3-VL-8B-Instruct")
    parser.add_argument("question", type=str, help="research question")
    parser.add_argument("--image-path", type=str, default="", help="local image path")
    parser.add_argument("--image-url", type=str, default="", help="remote image url")
    parser.add_argument(
        "--route-mode",
        type=str,
        default="auto",
        choices=["auto", "direct_inference", "light_rag", "full_rag", "vision_recheck", "agentic_rag"], help="workflow route mode"
    )
    parser.add_argument(
        "--retrieval-backend",
        type=str,
        default="",
        help="optional KG backend override: auto/qianfan/dbpedia/graph/proxy_graph/light_rag/hybrid",
    )
    parser.add_argument(
        "--debug-vllm-io",
        action="store_true",
        help="print request/response payloads sent to vLLM",
    )
    parser.add_argument(
        "--healthcheck",
        action="store_true",
        help="run local/DashScope backend healthcheck before workflow",
    )
    parser.add_argument(
        "--vllm-local-media-root",
        type=str,
        default="",
        help="Local media root for staging images before sending to vLLM; overrides VLLM_ALLOWED_LOCAL_MEDIA_ROOT.",
    )
    args = parser.parse_args()

    if args.debug_vllm_io:
        from config import settings
        settings.debug_vllm_io = True

    if args.vllm_local_media_root:
        os.environ["VLLM_ALLOWED_LOCAL_MEDIA_ROOT"] = str(
            Path(args.vllm_local_media_root).expanduser().resolve()
        )

    _assert_runtime_contract()

    image_ref = _prepare_cli_image_ref(
        args.image_path,
        args.image_url,
        vllm_local_media_root=args.vllm_local_media_root,
    )

    workflow = build_workflow(retrieval_backend=args.retrieval_backend)

    if args.healthcheck:
        _backend_healthcheck(workflow.planner.llm, image_ref=image_ref)

    if image_ref:
        workflow.researcher.tool_router.set_current_image_ref(image_ref)

    report = workflow.run(
        question=args.question,
        image_ref=image_ref,
        route_mode=args.route_mode,
    )

    print("\n=== Research Report ===\n")
    print(report)


if __name__ == "__main__":
    main()
