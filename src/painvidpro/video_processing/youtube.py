"""File to download and process videos from YouTube."""

from typing import Any, Dict, List

import yt_dlp


def _extract_meta(result: Dict[str, Any], key_list: List[str], default: Any = "") -> Dict[str, Any]:
    """Extracts the keys from the results."""
    ret: Dict[str, Any] = {}
    for key in key_list:
        ret[key] = result.get(key, default)
    return ret


def download_video(url: str, output_path: str, format: str = "bestvideo[height<=360]") -> int:
    """Downloads a YouTube video.

    Args:
        url: The YouTube URL.
        output_path: The output path on disk.
        format: The ydl video format.

    Returns:
        yt-dlp retcode, 0 for success.
    """
    ydl_opts = {"format": format, "outtmpl": output_path}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        return ydl.download([url])


def get_info_from_yt_url(url: str) -> List[Dict[str, Any]]:
    """Get all Video Ids related to the YouTube url.

    If the URL corresponds to a playlist, the information of all videos get returned.
    If the URL corresponds to a single video, only the information of the video gets returned.

    For a playlist URL make sure that the format is: "https://www.youtube.com/playlist?list=..."
    otherwise the function will only retrieve the first video if any. Bad playlsit url example:
    "https://www.youtube.com/watch?v=...&list=..."

    Args:
        url: The YouTube URL.

    Returns:
        A list of YouTube metadata as a dict.
    """
    key_list = ["id", "title", "license", "channel", "channel_id"]
    ret: List[Dict[str, Any]] = []
    ydl_opts = {
        "outtmpl": "%(id)s%(title)s%(license)s%(channel_id)s%(channel)s",
        "quiet": True,
        "ignoreerrors": True,
        "extract_flat": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url, download=False)
        if not result:
            return ret

        if "entries" in result:
            # A playlist
            for item in result["entries"]:
                ret.append(_extract_meta(item, key_list))
        else:
            # A single video
            ret.append(_extract_meta(result, key_list))
        return ret


def get_metadata_from_vido(video_id: str) -> Dict[str, Any]:
    """Extracts some metadata from the video.

    Args:
        video_id: The video ID or URL.

    Returns:
        A dict containing the metadata.
    """
    ydl_opts = {
        "outtmpl": "%(id)s%(title)s%(license)s%(channel_id)s%(channel)s",
        "quiet": True,
        "ignoreerrors": True,
    }
    key_list = ["id", "title", "license", "channel", "channel_id"]

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        res = ydl.extract_info(video_id, download=False)
        return _extract_meta(res, key_list)
