"""File to download and process videos from YouTube."""

from typing import List

import yt_dlp


def download_video(url: str, output_path: str):
    """Downloads a YouTube video.

    Args:
        url: The YouTube URL.
        output_path: The output path on disk.
    """
    ydl_opts = {"format": "bestvideo[height<=360]", "outtmpl": output_path}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def get_ids_from_playlist(playlist_url: str) -> List[str]:
    """Get all Video Ids that are contained in a YouTube playlist.

    Args:
        playlist_url: The playlist URL.

    Returns:
        A list of YouTube video IDs.
    """
    ydl_opts = {
        "outtmpl": "%(id)s",
        "quiet": True,
        "ignoreerrors": True,
    }
    ret: List[str] = []

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(playlist_url, download=False)

        if "entries" in result:
            for item in result["entries"]:
                ret.append(item["id"])
    return ret
