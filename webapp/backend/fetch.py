"""Download a video from a link (YouTube, TikTok, Instagram, Facebook, ...) for dubbing.

yt-dlp does the site-specific work. Two limits keep a pasted link from turning into an hour-long
job: the same size cap as a direct upload, and a duration cap checked from the page's metadata
*before* anything is downloaded, so an over-long video is rejected in seconds rather than after a
multi-gigabyte transfer.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

MAX_DURATION_S = 10 * 60
# Dubbing only needs the speech and a watchable picture; 720p keeps downloads small and the
# final remux fast without visibly hurting a phone-sized result.
FORMAT = "bv*[height<=720]+ba/b[height<=720]/bv*+ba/b"

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


class LinkError(RuntimeError):
    """The link could not be turned into a local video file; the message is user-facing."""


def valid_url(url: str) -> bool:
    parsed = urlparse(url.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _friendly(message: str) -> str:
    text = _ANSI.sub("", message).removeprefix("ERROR: ").strip()
    lowered = text.lower()
    if any(k in lowered for k in ("login", "log in", "sign in", "cookies", "private")):
        return (
            "This video needs a login or is private, so it can't be fetched automatically. "
            "Download it yourself and upload the file instead."
        )
    if "unsupported url" in lowered:
        return "This website isn't supported. Download the video and upload the file instead."
    if "unavailable" in lowered or "not available" in lowered or "404" in lowered:
        return "This video is unavailable (deleted, region-locked or the link is wrong)."
    return f"Could not download this link: {text[:200]}"


def download(url: str, job_dir: Path, max_bytes: int, progress=None) -> tuple[Path, str]:
    """Fetch `url` into `job_dir`; return (video path, title). Raises LinkError on failure."""
    import yt_dlp

    def hook(d: dict) -> None:
        if progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                progress(round(100 * d.get("downloaded_bytes", 0) / total))

    options = {
        "format": FORMAT,
        "merge_output_format": "mp4",
        "outtmpl": str(job_dir / "input.%(ext)s"),
        "noplaylist": True,
        "max_filesize": max_bytes,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [hook],
        # YouTube now requires solving a JavaScript challenge. yt-dlp only looks for deno by
        # default; Node is already installed for the frontend, so use that.
        "js_runtimes": {"node": {}},
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
            if info.get("_type") == "playlist":
                raise LinkError("This link is a playlist. Paste a link to a single video.")
            duration = info.get("duration")
            if duration and duration > MAX_DURATION_S:
                raise LinkError(
                    f"This video is {duration / 60:.0f} min long; links are limited to "
                    f"{MAX_DURATION_S // 60} min. Trim it and upload the file instead."
                )
            info = ydl.process_ie_result(info, download=True)
    except yt_dlp.utils.DownloadError as exc:
        raise LinkError(_friendly(str(exc))) from exc

    # max_filesize makes yt-dlp skip an oversized file silently rather than raise.
    files = [p for p in job_dir.glob("input.*") if p.suffix not in {".part", ".ytdl"}]
    if not files:
        raise LinkError(
            f"The video is larger than {max_bytes // (1024 * 1024)} MB, or no downloadable "
            "video was found at this link."
        )
    return files[0], (info.get("title") or "video")
