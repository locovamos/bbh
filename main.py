import os
import uuid
import json
import asyncio
import time
import base64
import urllib.error
import urllib.request
from urllib.parse import urlparse
from pathlib import Path
from typing import Literal, Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from compose_transparent_overlay import ForegroundOverlay, overlay_multiple_non_transparent_parts
from remove_background import remove_background_video
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(
    title="Video Explanation Gap Analyzer",
    description="Analyze a YouTube link or MP4 file and find video segments with concepts not fully explained.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔥 allow all (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MEDIA_DIR = Path("media")
UPLOADS_DIR = MEDIA_DIR / "uploads"
OUTPUTS_DIR = MEDIA_DIR / "outputs"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")

HERA_PROMPT_TEMPLATE_PATH = Path(__file__).with_name("PROMPT_HERA.md")
PIPELINE_JOBS: dict[str, dict[str, Any]] = {}

class GapSegment(BaseModel):
    title: str = Field(..., description="Short segment title")
    content: str = Field(..., description="What is not explained in this segment")
    start_timestamp: str = Field(..., description="Timestamp where the gap starts. Format HH:MM:SS")
    end_timestamp: str = Field(..., description="Timestamp where the gap ends. Format HH:MM:SS")


class AnalysisResponse(BaseModel):
    video_title: str
    gaps: list[GapSegment]


class BackgroundRemovalResponse(BaseModel):
    output_video_url: str
    output_video_path: str


class HeraVideoCreateResponse(BaseModel):
    video_id: str
    project_url: str | None = None


class HeraVideoStatusResponse(BaseModel):
    status: str
    video_id: str | None = None
    video_url: str | None = None
    response: dict | None = None


class ComposeIntervalItem(BaseModel):
    foreground_video_url: str
    start_timestamp: str = Field(..., description="Start timestamp in HH:MM:SS")
    end_timestamp: str | None = Field(default=None, description="Optional end timestamp in HH:MM:SS")


class ComposeOverlayRequest(BaseModel):
    background_video_url: str
    overlays: list[ComposeIntervalItem]


class ComposeOverlayResponse(BaseModel):
    output_video_url: str
    output_video_path: str


class TavilyImageResponse(BaseModel):
    query: str
    image_url: str


class PipelineGapResult(BaseModel):
    title: str
    content: str
    reference_type: str
    start_timestamp: str
    end_timestamp: str
    image_url: str
    hera_video_id: str
    generated_video_url: str
    transparent_video_url: str
    clip_duration_seconds: int
    poll_rounds: int


class PipelinePrototypeResponse(BaseModel):
    source_video_url: str
    final_video_url: str
    final_video_path: str
    analysis: AnalysisResponse
    processed_gaps: list[PipelineGapResult]
    details: dict[str, Any] | None = None


class PipelineJobStartResponse(BaseModel):
    job_id: str
    status: str


class PipelineJobStatusResponse(BaseModel):
    job_id: str
    status: str
    stage: str
    message: str | None = None
    progress: float
    result: PipelinePrototypeResponse | None = None
    error: str | None = None
    updated_at: float


ViewerProfile = Literal["informed", "curious", "newcomer"]
CaptionDensity = Literal["subtle", "immersive"]


def _profile_instruction(viewer_profile: ViewerProfile) -> str:
    if viewer_profile == "informed":
        return (
            "Knowledge baseline: follows major U.S. political actors/institutions and headline events.\n"
            "Include only high-value gaps: niche people, acronyms, process details, or historical references that block deep understanding.\n"
            "Do NOT include obvious basics (president, congress, Democrats vs Republicans)."
        )
    if viewer_profile == "curious":
        return (
            "Knowledge baseline: recognizes common U.S. political terms but misses insider context.\n"
            "Include medium/high-impact gaps: policy acronyms, non-obvious public figures, institutions, and events assumed by speakers.\n"
            "Skip very basic civics unless the reference is used in a specialized way."
        )
    return (
        "Knowledge baseline: little U.S. political context; likely international newcomer.\n"
        "Include any reference that is required to follow the conversation: people, acronyms, media brands, institutions, election mechanics, key events.\n"
        "Prefer clear plain-language explanations over insider jargon."
    )


def _density_instruction(caption_density: CaptionDensity) -> str:
    if caption_density == "subtle":
        return (
            "Selection policy: high precision only.\n"
            "Return only the most important gaps (roughly 1 meaningful gap per 60-120s of discussion).\n"
            "Merge repeated references into one interval and avoid near-duplicates."
        )
    return (
        "Selection policy: high recall with quality control.\n"
        "Return most meaningful gaps (roughly 1 meaningful gap per 20-45s when references are dense).\n"
        "Include secondary references if they improve comprehension, but still avoid trivial/name-only mentions."
    )


def _global_quality_criteria() -> str:
    return """
Quality criteria for identifying a "not explained" gap:
1) Dependency test: Without this explanation, an international viewer would miss the point or implication.
2) Explanation test: The speaker does not define it clearly in the nearby context.
3) Specificity test: The missing context can be explained concretely in 1-2 sentences.
4) Impact test: Prioritize references that affect interpretation, argument strength, or factual understanding.

Timestamp and dedup rules:
- Use precise intervals where the unexplained reference is introduced/discussed.
- If a reference repeats throughout the video, keep the earliest high-signal interval unless a later segment adds a new unexplained angle.
- Avoid overlapping intervals unless two distinct gaps truly co-occur.

Content writing rules:
- title: short noun phrase (who/what is missing).
- content: explain what it is + why that context matters in this discussion.
- Do not invent facts; if uncertain, keep wording cautious.
""".strip()


def _build_analysis_prompt(viewer_profile: ViewerProfile, caption_density: CaptionDensity) -> str:
    return f"""
You are analyzing a video for explanation gaps.
Find intervals where a concept, idea, person, or reference appears but is not clearly explained.

Rules:
- Keep title concise and specific.
- content must be short (max ~90 characters), clear, and directly useful.
- Timestamps must match the real interval.
- If no gaps, return "gaps": [].
- Viewer profile: {viewer_profile}
- Profile guidance: {_profile_instruction(viewer_profile)}
- Caption density: {caption_density}
- Density guidance: {_density_instruction(caption_density)}
- Apply this global rubric exactly:
{_global_quality_criteria()}
""".strip()


def _download_url_to_path(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"Failed to download URL '{url}': {exc.reason}") from exc

    if not data:
        raise HTTPException(status_code=400, detail=f"Downloaded file is empty: {url}")
    destination.write_bytes(data)
    return destination


def _guess_extension_from_url(url: str, fallback: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".mp4", ".webm", ".mov", ".mkv"}:
        return suffix
    return fallback


def _timestamp_to_seconds(value: str) -> int:
    parts = value.strip().split(":")
    if len(parts) != 3:
        raise HTTPException(status_code=400, detail=f"Invalid timestamp format: {value}")
    try:
        h = int(parts[0])
        m = int(parts[1])
        s = int(parts[2])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid timestamp format: {value}") from exc
    if h < 0 or m < 0 or s < 0 or m > 59 or s > 59:
        raise HTTPException(status_code=400, detail=f"Invalid timestamp value: {value}")
    return h * 3600 + m * 60 + s


def _safe_duration_from_gap(start_timestamp: str, end_timestamp: str, fallback_seconds: int, min_seconds: int = 6) -> int:
    try:
        duration = _timestamp_to_seconds(end_timestamp) - _timestamp_to_seconds(start_timestamp)
        return max(min_seconds, min(60, duration))
    except HTTPException:
        return max(min_seconds, min(60, fallback_seconds))


def _prepare_yt_cookiefile() -> str | None:
    """
    Optional cookie sources for yt-dlp on cloud runners:
    - YTDLP_COOKIES_TXT_B64: base64-encoded cookies.txt content
    - YTDLP_COOKIES_TXT: raw cookies.txt content
    """
    cookies_b64 = os.getenv("YTDLP_COOKIES_TXT_B64")
    cookies_raw = os.getenv("YTDLP_COOKIES_TXT")
    content: str | None = None

    if cookies_b64:
        try:
            content = base64.b64decode(cookies_b64).decode("utf-8")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Invalid YTDLP_COOKIES_TXT_B64 value: {exc}") from exc
    elif cookies_raw:
        content = cookies_raw

    if not content:
        return None

    cookie_path = UPLOADS_DIR / "yt_cookies.txt"
    cookie_path.write_text(content, encoding="utf-8")
    return str(cookie_path)


def _download_youtube_video(youtube_url: str, destination: Path) -> Path:
    try:
        import yt_dlp
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="yt-dlp is not installed. Install dependencies with `pip install -r requirements.txt`.",
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    template = str(destination.with_suffix(".%(ext)s"))
    options = {
        "format": "mp4/bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "outtmpl": template,
        "quiet": True,
        "no_warnings": True,
        "retries": 5,
        "fragment_retries": 5,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        },
        "extractor_args": {
            "youtube": {"player_client": ["android", "web"]},
        },
    }
    cookiefile = _prepare_yt_cookiefile()
    if cookiefile:
        options["cookiefile"] = cookiefile

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.extract_info(youtube_url, download=True)
    except Exception as exc:
        message = str(exc)
        if "Sign in to confirm you’re not a bot" in message or "Sign in to confirm you're not a bot" in message:
            raise HTTPException(
                status_code=502,
                detail=(
                    "YouTube blocked anonymous download. Configure YTDLP_COOKIES_TXT_B64 "
                    "(preferred) or YTDLP_COOKIES_TXT in Railway service variables using an "
                    "exported YouTube cookies.txt, then retry."
                ),
            ) from exc
        raise HTTPException(status_code=502, detail=f"Failed to download YouTube video: {exc}") from exc

    mp4_files = list(destination.parent.glob(f"{destination.stem}*.mp4"))
    if not mp4_files:
        raise HTTPException(status_code=502, detail="YouTube download did not produce an MP4 file.")
    if mp4_files[0] != destination:
        destination.write_bytes(mp4_files[0].read_bytes())
    return destination


def _build_hera_prompt(
    title: str,
    body_text: str,
    seconds: int,
    banner_color: str,
    title_text_color: str,
    body_text_color: str,
) -> str:
    if not HERA_PROMPT_TEMPLATE_PATH.exists():
        raise HTTPException(status_code=500, detail="Missing PROMPT_HERA.md template file.")

    template = HERA_PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    required_placeholders = {
        "{{TITLE}}",
        "{{BODY_TEXT}}",
        "{{SECONDS}}",
        "{{BANNER_COLOR}}",
        "{{TITLE_TEXT_COLOR}}",
        "{{BODY_TEXT_COLOR}}",
    }
    missing = [token for token in required_placeholders if token not in template]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"PROMPT_HERA.md is missing placeholders: {', '.join(missing)}",
        )

    prompt = (
        template.replace("{{TITLE}}", title)
        .replace("{{BODY_TEXT}}", body_text)
        .replace("{{SECONDS}}", str(seconds))
        .replace("{{BANNER_COLOR}}", banner_color)
        .replace("{{TITLE_TEXT_COLOR}}", title_text_color)
        .replace("{{BODY_TEXT_COLOR}}", body_text_color)
    )
    return prompt


def _normalize_color_descriptor(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    descriptor = value.strip()
    if not descriptor:
        return None
    if len(descriptor) > 80:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} is too long (max 80 characters).",
        )
    if "{" in descriptor or "}" in descriptor:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} contains invalid characters.",
        )
    return descriptor


def _resolve_optional_color(value: str | None, default_value: str, field_name: str) -> str:
    normalized = _normalize_color_descriptor(value, field_name)
    return normalized or default_value


def _hera_create_video(
    *,
    title: str,
    body_text: str,
    seconds: int,
    asset_image_url: str,
    banner_background_color: str | None,
    title_text_color: str | None,
    body_text_color: str | None,
) -> HeraVideoCreateResponse:
    hera_api_key = os.getenv("HERA_API_KEY")
    if not hera_api_key:
        raise HTTPException(status_code=500, detail="Missing HERA_API_KEY environment variable.")
    if seconds < 1 or seconds > 60:
        raise HTTPException(status_code=400, detail="seconds must be between 1 and 60.")

    safe_banner_color = _resolve_optional_color(
        banner_background_color,
        "dark charcoal to near-black",
        "banner_background_color",
    )
    safe_title_color = _resolve_optional_color(
        title_text_color,
        "white or off-white",
        "title_text_color",
    )
    safe_body_color = _resolve_optional_color(
        body_text_color,
        "soft off-white",
        "body_text_color",
    )
    prompt = _build_hera_prompt(
        title=title,
        body_text=body_text,
        seconds=seconds,
        banner_color=safe_banner_color,
        title_text_color=safe_title_color,
        body_text_color=safe_body_color,
    )

    payload = {
        "prompt": prompt,
        "assets": [{"type": "image", "url": asset_image_url}],
        "duration_seconds": seconds,
        "outputs": [{"format": "mp4", "aspect_ratio": "16:9", "fps": "24", "resolution": "1080p"}],
    }
    request = urllib.request.Request(
        url="https://api.hera.video/v1/videos",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-api-key": hera_api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore")
        raise HTTPException(status_code=502, detail=f"Hera API HTTP error {exc.code}: {error_body or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"Hera API connection error: {exc.reason}") from exc

    try:
        parsed = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail=f"Invalid JSON from Hera API: {exc.msg}") from exc

    video_id = parsed.get("video_id")
    if not video_id:
        raise HTTPException(status_code=502, detail="Hera API response missing video_id.")
    return HeraVideoCreateResponse(video_id=video_id, project_url=parsed.get("project_url"))


def _hera_get_video_status(video_id: str) -> HeraVideoStatusResponse:
    hera_api_key = os.getenv("HERA_API_KEY")
    if not hera_api_key:
        raise HTTPException(status_code=500, detail="Missing HERA_API_KEY environment variable.")

    request = urllib.request.Request(
        url=f"https://api.hera.video/v1/videos/{video_id}",
        headers={"x-api-key": hera_api_key},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore")
        try:
            parsed_error = json.loads(error_body) if error_body else {"error": exc.reason}
        except json.JSONDecodeError:
            parsed_error = {"error": error_body or exc.reason}
        return HeraVideoStatusResponse(status="error", video_id=video_id, response=parsed_error)
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"Hera API connection error: {exc.reason}") from exc

    try:
        parsed = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail=f"Invalid JSON from Hera API: {exc.msg}") from exc

    status = parsed.get("status")
    if status == "success":
        outputs = parsed.get("outputs", [])
        for output in outputs:
            if output.get("status") == "success" and output.get("file_url"):
                return HeraVideoStatusResponse(status="success", video_id=parsed.get("video_id", video_id), video_url=output["file_url"])
        return HeraVideoStatusResponse(status="error", video_id=parsed.get("video_id", video_id), response=parsed)
    if status == "in-progress":
        return HeraVideoStatusResponse(status="in-progress", video_id=parsed.get("video_id", video_id))
    return HeraVideoStatusResponse(status="error", video_id=parsed.get("video_id", video_id), response=parsed)


def _tavily_first_image(query: str) -> str:
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise HTTPException(status_code=500, detail="Missing TAVILY_API_KEY environment variable.")
    if not query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty.")

    tavily_payload = {
        "api_key": tavily_api_key,
        "query": query + "filetype:png",
        "search_depth": "advanced",
        "include_images": True,
    }
    tavily_request = urllib.request.Request(
        url="https://api.tavily.com/search",
        data=json.dumps(tavily_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(tavily_request, timeout=60) as tavily_response:
            response = json.loads(tavily_response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore")
        raise HTTPException(status_code=502, detail=f"Tavily API HTTP error {exc.code}: {error_body or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"Tavily API connection error: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail=f"Invalid JSON from Tavily API: {exc.msg}") from exc

    images = response.get("images") or []
    if not images:
        raise HTTPException(status_code=404, detail="No images found for this query.")
    return images[0]


def _run_gemini_analysis(
    video_part: types.Part,
    fallback_video_title: str,
    viewer_profile: ViewerProfile,
    caption_density: CaptionDensity,
) -> AnalysisResponse:
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Missing GEMINI_API_KEY environment variable.",
        )

    client = genai.Client(api_key=api_key)

    prompt = _build_analysis_prompt(viewer_profile=viewer_profile, caption_density=caption_density)
    response = client.models.generate_content(
        model=model_name,
        contents=[prompt, video_part],
        config={
            "response_mime_type": "application/json",
            "response_json_schema": AnalysisResponse.model_json_schema(),
        },
    )
    if not getattr(response, "text", None):
        raise HTTPException(status_code=502, detail="Gemini returned an empty response.")

    try:
        result = AnalysisResponse.model_validate_json(response.text)
        if not result.video_title:
            result.video_title = fallback_video_title
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini JSON schema validation failed: {exc}",
        ) from exc


def _classify_reference_type(title: str, content: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    if not api_key:
        return "entity"

    client = genai.Client(api_key=api_key)
    prompt = (
        "Classify this reference into exactly one label: person, place, entity, ideology.\n"
        f"Title: {title}\n"
        f"Context: {content}\n"
        "Return JSON: {\"reference_type\":\"person|place|entity|ideology\"}"
    )
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": {
                    "type": "object",
                    "properties": {
                        "reference_type": {
                            "type": "string",
                            "enum": ["person", "place", "entity", "ideology"],
                        }
                    },
                    "required": ["reference_type"],
                },
            },
        )
        parsed = json.loads(response.text or "{}")
        ref_type = parsed.get("reference_type")
        if ref_type in {"person", "place", "entity", "ideology"}:
            return ref_type
    except Exception:
        pass
    return "entity"


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Video analyzer is running"}


@app.post("/analyze-video", response_model=AnalysisResponse)
async def analyze_video(
    youtube_url: str | None = Form(default=None),
    video_url: str | None = Form(default=None),
    video_file: UploadFile | None = File(default=None),
    viewer_profile: ViewerProfile = Form(default="newcomer"),
    caption_density: CaptionDensity = Form(default="subtle"),
) -> AnalysisResponse:
    provided_urls = [url for url in [youtube_url, video_url] if url]
    if len(provided_urls) > 1:
        raise HTTPException(
            status_code=400,
            detail="Provide only one URL field: youtube_url or video_url.",
        )
    source_url = provided_urls[0] if provided_urls else None

    if bool(source_url) == bool(video_file):
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one video input: URL (youtube_url/video_url) or video_file.",
        )

    if source_url:
        video_part = types.Part(
            file_data=types.FileData(file_uri=source_url),
        )
        return _run_gemini_analysis(
            video_part=video_part,
            fallback_video_title="youtube-video",
            viewer_profile=viewer_profile,
            caption_density=caption_density,
        )

    if video_file is None:
        raise HTTPException(status_code=400, detail="video_file is required.")
    if video_file.content_type != "video/mp4":
        raise HTTPException(status_code=400, detail="Only MP4 file uploads are supported.")

    content = await video_file.read()
    max_mb = int(os.getenv("MAX_VIDEO_MB", "50"))
    max_bytes = max_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file too large ({len(content) // (1024 * 1024)} MB). Maximum is {max_mb} MB.",
        )

    video_part = types.Part.from_bytes(data=content, mime_type="video/mp4")
    video_title = Path(video_file.filename or "uploaded-video.mp4").stem
    return _run_gemini_analysis(
        video_part=video_part,
        fallback_video_title=video_title,
        viewer_profile=viewer_profile,
        caption_density=caption_density,
    )
@app.post("/remove-background", response_model=BackgroundRemovalResponse)
async def remove_background_endpoint(
    request: Request,
    video_file: UploadFile = File(...),
) -> BackgroundRemovalResponse:
    if video_file.content_type not in {"video/mp4", "video/webm", "video/quicktime"}:
        raise HTTPException(status_code=400, detail="Supported formats: mp4, webm, mov.")

    suffix = Path(video_file.filename or "video.mp4").suffix or ".mp4"
    input_name = f"{uuid.uuid4().hex}{suffix}"
    output_name = f"{Path(input_name).stem}_transparent.webm"
    input_path = UPLOADS_DIR / input_name
    output_path = OUTPUTS_DIR / output_name

    content = await video_file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    input_path.write_bytes(content)

    try:
        await run_in_threadpool(
            remove_background_video,
            str(input_path),
            str(output_path),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Background removal failed: {exc}") from exc

    output_url = str(request.base_url).rstrip("/") + f"/media/outputs/{output_name}"
    return BackgroundRemovalResponse(
        output_video_url=output_url,
        output_video_path=str(output_path.resolve()),
    )


@app.post("/generate-video", response_model=HeraVideoCreateResponse)
async def generate_video_with_hera(
    title: str = Form(...),
    body_text: str = Form(...),
    seconds: int = Form(...),
    asset_image_url: str = Form(...),
    banner_background_color: str | None = Form(default=None),
    title_text_color: str | None = Form(default=None),
    body_text_color: str | None = Form(default=None),
) -> HeraVideoCreateResponse:
    return _hera_create_video(
        title=title,
        body_text=body_text,
        seconds=seconds,
        asset_image_url=asset_image_url,
        banner_background_color=banner_background_color,
        title_text_color=title_text_color,
        body_text_color=body_text_color,
    )


@app.get("/generate-video/{video_id}", response_model=HeraVideoStatusResponse)
async def get_generated_video_status(video_id: str) -> HeraVideoStatusResponse:
    return _hera_get_video_status(video_id)


@app.get("/search-image", response_model=TavilyImageResponse)
async def search_image_with_tavily(query: str) -> TavilyImageResponse:
    return TavilyImageResponse(query=query, image_url=_tavily_first_image(query))


def _update_pipeline_job(
    job_id: str,
    *,
    status: str | None = None,
    stage: str | None = None,
    message: str | None = None,
    progress: float | None = None,
    result: PipelinePrototypeResponse | None = None,
    error: str | None = None,
) -> None:
    job = PIPELINE_JOBS.get(job_id)
    if not job:
        return
    if status is not None:
        job["status"] = status
    if stage is not None:
        job["stage"] = stage
    if message is not None:
        job["message"] = message
    if progress is not None:
        job["progress"] = max(0.0, min(1.0, progress))
    if result is not None:
        job["result"] = result
    if error is not None:
        job["error"] = error
    job["updated_at"] = time.time()


async def _run_pipeline_prototype_core(
    *,
    base_url: str,
    youtube_url: str,
    viewer_profile: ViewerProfile,
    caption_density: CaptionDensity,
    max_gaps: int,
    fallback_gap_seconds: int,
    min_caption_seconds: int,
    poll_interval_seconds: int,
    max_poll_rounds: int,
    banner_background_color: str | None,
    title_text_color: str | None,
    body_text_color: str | None,
    job_id: str | None = None,
) -> PipelinePrototypeResponse:
    t0 = time.perf_counter()
    if max_gaps < 1 or max_gaps > 20:
        raise HTTPException(status_code=400, detail="max_gaps must be between 1 and 20.")
    if fallback_gap_seconds < 1 or fallback_gap_seconds > 60:
        raise HTTPException(status_code=400, detail="fallback_gap_seconds must be between 1 and 60.")
    if min_caption_seconds < 1 or min_caption_seconds > 60:
        raise HTTPException(status_code=400, detail="min_caption_seconds must be between 1 and 60.")
    if poll_interval_seconds < 2 or poll_interval_seconds > 60:
        raise HTTPException(status_code=400, detail="poll_interval_seconds must be between 2 and 60.")
    if max_poll_rounds < 1 or max_poll_rounds > 300:
        raise HTTPException(status_code=400, detail="max_poll_rounds must be between 1 and 300.")

    if job_id:
        _update_pipeline_job(job_id, status="running", stage="analyzing", message="Analyzing YouTube video", progress=0.05)

    video_part = types.Part(file_data=types.FileData(file_uri=youtube_url))
    analysis = await asyncio.to_thread(
        _run_gemini_analysis,
        video_part,
        "youtube-video",
        viewer_profile,
        caption_density,
    )
    selected_gaps = analysis.gaps[:max_gaps]
    if not selected_gaps:
        raise HTTPException(status_code=400, detail="Gemini analysis returned no gaps to process.")

    if job_id:
        _update_pipeline_job(job_id, stage="download_background", message="Downloading source YouTube video", progress=0.12)

    pipeline_id = uuid.uuid4().hex
    background_path = UPLOADS_DIR / f"{pipeline_id}_background.mp4"
    await asyncio.to_thread(_download_youtube_video, youtube_url, background_path)

    async def _process_gap(index: int, gap: GapSegment) -> tuple[int, str | None, dict | None]:
        reference_type = await asyncio.to_thread(_classify_reference_type, gap.title, gap.content)
        image_query = gap.title
        try:
            image_url = await asyncio.to_thread(_tavily_first_image, image_query)
        except HTTPException as exc:
            if exc.status_code == 404:
                return index, None, None
            raise
        gap_seconds = _safe_duration_from_gap(
            gap.start_timestamp,
            gap.end_timestamp,
            fallback_gap_seconds,
            min_seconds=min_caption_seconds,
        )
        create_result = await asyncio.to_thread(
            _hera_create_video,
            title=gap.title,
            body_text=gap.content,
            seconds=gap_seconds,
            asset_image_url=image_url,
            banner_background_color=banner_background_color,
            title_text_color=title_text_color,
            body_text_color=body_text_color,
        )
        return (
            index,
            create_result.video_id,
            {
                "gap": gap,
                "reference_type": reference_type,
                "image_url": image_url,
                "image_query": image_query,
                "clip_duration_seconds": gap_seconds,
                "generated_video_url": None,
                "transparent_video_url": None,
                "transparent_video_path": None,
                "poll_rounds": 0,
            },
        )

    if job_id:
        _update_pipeline_job(job_id, stage="search_generate", message="Classifying references, searching images, generating clips", progress=0.22)
    created = await asyncio.gather(*[_process_gap(i, g) for i, g in enumerate(selected_gaps)])
    jobs_by_id: dict[str, dict] = {}
    gap_index_to_video_id: dict[int, str] = {}
    skipped_gap_indices: list[int] = []
    for idx, video_id, payload in created:
        if video_id is None or payload is None:
            skipped_gap_indices.append(idx)
            continue
        jobs_by_id[video_id] = payload
        gap_index_to_video_id[idx] = video_id

    if not jobs_by_id:
        raise HTTPException(status_code=400, detail="No processable gaps: image search returned no results for all gaps.")

    if job_id:
        _update_pipeline_job(job_id, stage="poll_generation", message="Waiting for generated clips", progress=0.4)

    pending_ids = set(jobs_by_id.keys())
    for round_idx in range(max_poll_rounds):
        if not pending_ids:
            break
        status_results = await asyncio.gather(
            *[asyncio.to_thread(_hera_get_video_status, video_id) for video_id in list(pending_ids)]
        )
        completed_ids: list[str] = []
        for status in status_results:
            video_id = status.video_id or ""
            if not video_id or video_id not in jobs_by_id:
                continue
            item = jobs_by_id[video_id]
            item["poll_rounds"] += 1
            if status.status == "success" and status.video_url:
                item["generated_video_url"] = status.video_url
                completed_ids.append(video_id)
            elif status.status == "error":
                raise HTTPException(
                    status_code=502,
                    detail=f"Hera generation failed for {video_id}: {status.response}",
                )
        for done_id in completed_ids:
            pending_ids.discard(done_id)
        if pending_ids:
            if job_id:
                done = len(jobs_by_id) - len(pending_ids)
                frac = 0.4 + (done / max(1, len(jobs_by_id))) * 0.25
                _update_pipeline_job(
                    job_id,
                    stage="poll_generation",
                    message=f"Poll round {round_idx + 1}: {done}/{len(jobs_by_id)} ready",
                    progress=frac,
                )
            await asyncio.sleep(poll_interval_seconds)

    if pending_ids:
        raise HTTPException(
            status_code=504,
            detail=f"Timeout waiting for Hera videos: {', '.join(sorted(pending_ids))}",
        )

    if job_id:
        _update_pipeline_job(job_id, stage="remove_background", message="Removing green backgrounds", progress=0.68)

    async def _materialize_clip(video_id: str, item: dict) -> None:
        generated_video_url = item["generated_video_url"]
        if not generated_video_url:
            raise HTTPException(status_code=500, detail=f"Missing generated URL for video_id {video_id}.")
        generated_ext = _guess_extension_from_url(generated_video_url, ".mp4")
        generated_local_path = UPLOADS_DIR / f"{pipeline_id}_{video_id}{generated_ext}"
        await asyncio.to_thread(_download_url_to_path, generated_video_url, generated_local_path)

        transparent_name = f"{pipeline_id}_{video_id}_transparent.webm"
        transparent_path = OUTPUTS_DIR / transparent_name
        await asyncio.to_thread(remove_background_video, str(generated_local_path), str(transparent_path))

        item["transparent_video_url"] = base_url + f"/media/outputs/{transparent_name}"
        item["transparent_video_path"] = str(transparent_path.resolve())

    await asyncio.gather(*[_materialize_clip(video_id, item) for video_id, item in jobs_by_id.items()])

    # Preserve Gemini gap ordering in final composition/output.
    overlays: list[ForegroundOverlay] = []
    processed_gaps: list[PipelineGapResult] = []
    for idx, gap in enumerate(selected_gaps):
        matched_video_id = gap_index_to_video_id.get(idx)
        matched_item = jobs_by_id.get(matched_video_id) if matched_video_id else None
        if matched_video_id is None or matched_item is None:
            continue

        overlays.append(
            ForegroundOverlay(
                foreground_video_path=str(matched_item["transparent_video_path"]),
                start_time=gap.start_timestamp,
                end_time=gap.end_timestamp,
            )
        )
        processed_gaps.append(
            PipelineGapResult(
                title=gap.title,
                content=gap.content,
                reference_type=str(matched_item["reference_type"]),
                start_timestamp=gap.start_timestamp,
                end_timestamp=gap.end_timestamp,
                image_url=str(matched_item["image_url"]),
                hera_video_id=matched_video_id,
                generated_video_url=str(matched_item["generated_video_url"]),
                transparent_video_url=str(matched_item["transparent_video_url"]),
                clip_duration_seconds=int(matched_item["clip_duration_seconds"]),
                poll_rounds=int(matched_item["poll_rounds"]),
            )
        )

    if job_id:
        _update_pipeline_job(job_id, stage="compose", message="Compositing overlays over source video", progress=0.9)

    final_name = f"{pipeline_id}_final_composed.mp4"
    final_path = OUTPUTS_DIR / final_name
    await run_in_threadpool(
        overlay_multiple_non_transparent_parts,
        str(background_path),
        overlays,
        str(final_path),
    )

    source_video_url = base_url + f"/media/uploads/{background_path.name}"
    final_video_url = base_url + f"/media/outputs/{final_name}"
    elapsed = time.perf_counter() - t0
    result = PipelinePrototypeResponse(
        source_video_url=source_video_url,
        final_video_url=final_video_url,
        final_video_path=str(final_path.resolve()),
        analysis=analysis,
        processed_gaps=processed_gaps,
        details={
            "selected_gaps_count": len(selected_gaps),
            "processed_gaps_count": len(processed_gaps),
            "skipped_gaps_count": len(skipped_gap_indices),
            "skipped_gap_indices": skipped_gap_indices,
            "min_caption_seconds": min_caption_seconds,
            "fallback_gap_seconds": fallback_gap_seconds,
            "poll_interval_seconds": poll_interval_seconds,
            "max_poll_rounds": max_poll_rounds,
            "elapsed_seconds": round(elapsed, 2),
        },
    )
    if job_id:
        _update_pipeline_job(job_id, status="success", stage="done", message="Pipeline completed", progress=1.0, result=result)
    return result


@app.post("/pipeline-prototype", response_model=PipelinePrototypeResponse)
async def run_pipeline_prototype(
    request: Request,
    youtube_url: str = Form(...),
    viewer_profile: ViewerProfile = Form(default="newcomer"),
    caption_density: CaptionDensity = Form(default="subtle"),
    max_gaps: int = Form(default=5),
    fallback_gap_seconds: int = Form(default=6),
    min_caption_seconds: int = Form(default=6),
    poll_interval_seconds: int = Form(default=8),
    max_poll_rounds: int = Form(default=45),
    banner_background_color: str | None = Form(default=None),
    title_text_color: str | None = Form(default=None),
    body_text_color: str | None = Form(default=None),
) -> PipelinePrototypeResponse:
    base_url = str(request.base_url).rstrip("/")
    return await _run_pipeline_prototype_core(
        base_url=base_url,
        youtube_url=youtube_url,
        viewer_profile=viewer_profile,
        caption_density=caption_density,
        max_gaps=max_gaps,
        fallback_gap_seconds=fallback_gap_seconds,
        min_caption_seconds=min_caption_seconds,
        poll_interval_seconds=poll_interval_seconds,
        max_poll_rounds=max_poll_rounds,
        banner_background_color=banner_background_color,
        title_text_color=title_text_color,
        body_text_color=body_text_color,
    )


@app.post("/pipeline-jobs", response_model=PipelineJobStartResponse)
async def start_pipeline_job(
    request: Request,
    youtube_url: str = Form(...),
    viewer_profile: ViewerProfile = Form(default="newcomer"),
    caption_density: CaptionDensity = Form(default="subtle"),
    max_gaps: int = Form(default=5),
    fallback_gap_seconds: int = Form(default=6),
    min_caption_seconds: int = Form(default=6),
    poll_interval_seconds: int = Form(default=8),
    max_poll_rounds: int = Form(default=45),
    banner_background_color: str | None = Form(default=None),
    title_text_color: str | None = Form(default=None),
    body_text_color: str | None = Form(default=None),
) -> PipelineJobStartResponse:
    job_id = uuid.uuid4().hex
    PIPELINE_JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "stage": "queued",
        "message": "Job queued",
        "progress": 0.0,
        "result": None,
        "error": None,
        "updated_at": time.time(),
    }
    base_url = str(request.base_url).rstrip("/")

    async def _runner() -> None:
        try:
            _update_pipeline_job(job_id, status="running", stage="starting", message="Starting pipeline", progress=0.01)
            await _run_pipeline_prototype_core(
                base_url=base_url,
                youtube_url=youtube_url,
                viewer_profile=viewer_profile,
                caption_density=caption_density,
                max_gaps=max_gaps,
                fallback_gap_seconds=fallback_gap_seconds,
                min_caption_seconds=min_caption_seconds,
                poll_interval_seconds=poll_interval_seconds,
                max_poll_rounds=max_poll_rounds,
                banner_background_color=banner_background_color,
                title_text_color=title_text_color,
                body_text_color=body_text_color,
                job_id=job_id,
            )
        except Exception as exc:
            _update_pipeline_job(job_id, status="error", stage="failed", message="Pipeline failed", error=str(exc), progress=1.0)

    asyncio.create_task(_runner())
    return PipelineJobStartResponse(job_id=job_id, status="queued")


@app.get("/pipeline-jobs/{job_id}", response_model=PipelineJobStatusResponse)
async def get_pipeline_job_status(job_id: str) -> PipelineJobStatusResponse:
    job = PIPELINE_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Pipeline job not found.")
    return PipelineJobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        stage=job["stage"],
        message=job.get("message"),
        progress=float(job.get("progress", 0.0)),
        result=job.get("result"),
        error=job.get("error"),
        updated_at=float(job.get("updated_at", time.time())),
    )


@app.post("/compose-overlay", response_model=ComposeOverlayResponse)
async def compose_overlay_video(
    request: Request,
    payload: ComposeOverlayRequest,
) -> ComposeOverlayResponse:
    if not payload.overlays:
        raise HTTPException(status_code=400, detail="At least one overlay item is required.")

    compose_id = uuid.uuid4().hex
    bg_ext = _guess_extension_from_url(payload.background_video_url, ".mp4")
    background_path = UPLOADS_DIR / f"{compose_id}_background{bg_ext}"
    output_name = f"{compose_id}_composed.mp4"
    output_path = OUTPUTS_DIR / output_name

    _download_url_to_path(payload.background_video_url, background_path)

    overlays: list[ForegroundOverlay] = []
    for index, item in enumerate(payload.overlays):
        fg_ext = _guess_extension_from_url(item.foreground_video_url, ".webm")
        foreground_path = UPLOADS_DIR / f"{compose_id}_fg_{index}{fg_ext}"
        _download_url_to_path(item.foreground_video_url, foreground_path)
        overlays.append(
            ForegroundOverlay(
                foreground_video_path=str(foreground_path),
                start_time=item.start_timestamp,
                end_time=item.end_timestamp,
            )
        )

    try:
        await run_in_threadpool(
            overlay_multiple_non_transparent_parts,
            str(background_path),
            overlays,
            str(output_path),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Overlay composition failed: {exc}") from exc

    output_url = str(request.base_url).rstrip("/") + f"/media/outputs/{output_name}"
    return ComposeOverlayResponse(
        output_video_url=output_url,
        output_video_path=str(output_path.resolve()),
    )
