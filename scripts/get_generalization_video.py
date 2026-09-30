#!/usr/bin/env python3
"""Download a public YouTube match/event video for held-out generalization review."""
import argparse
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.frcvision.roster import validate_alliance_teams

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("url", help="Public FRC event/match YouTube URL (choose season/event absent from training data)")
ap.add_argument("--name", default="heldout_match")
ap.add_argument("--max-height", type=int, choices=(360, 480, 720, 1080, 1440, 2160),
                default=720, help="Maximum source height; prefer 1080p or higher for bumper OCR")
ap.add_argument("--out", default=os.getenv("FRC_DATA_ROOT", "/home/brian/Projects/RoboPerception-data") + "/videos")
ap.add_argument("--event", required=True, help="Event and season, e.g. 2025 FIRST Championship")
ap.add_argument("--match", required=True, help="Official match label or key")
ap.add_argument("--red-teams", nargs=3, required=True, metavar=("R1", "R2", "R3"))
ap.add_argument("--blue-teams", nargs=3, required=True, metavar=("B1", "B2", "B3"))
ap.add_argument("--roster-source", required=True,
                help="The Blue Alliance match page used to verify all six team numbers")
ap.add_argument("--first-results", default="",
                help="Optional official FIRST match-results URL for cross-checking")
ap.add_argument("--publisher", default="FIRST Robotics Competition")
ap.add_argument("--rights", default="Copyright belongs to the publisher; local evaluation only. Check the source terms before redistribution.")
args = ap.parse_args()
out = Path(args.out).resolve()
out.mkdir(parents=True, exist_ok=True)
alliance_teams = validate_alliance_teams({
    "red": args.red_teams,
    "blue": args.blue_teams,
})
format_selector = (
    f"bv*[height<={args.max_height}][ext=mp4][vcodec^=avc1]/"
    f"b[height<={args.max_height}][ext=mp4][vcodec^=avc1]"
)
subprocess.run(
    [sys.executable, "-m", "yt_dlp", "--no-playlist", "--force-overwrites",
     "-f", format_selector,
     "-o", str(out / f"{args.name}.%(ext)s"), args.url],
    check=True,
)
video_extensions = {".mp4", ".mkv", ".mov", ".webm"}
downloads = [
    path for path in out.glob(f"{args.name}.*")
    if path.suffix.lower() in video_extensions and path.is_file()
]
if not downloads:
    raise FileNotFoundError(f"yt-dlp finished without a recognized video for {args.name}")
video = max(downloads, key=lambda path: path.stat().st_mtime)
roster_path = video.with_suffix(".roster.json")
roster = {
    "schema_version": 1,
    "event": args.event,
    "match": args.match,
    "youtube_url": args.url,
    "publisher": args.publisher,
    "roster_source": args.roster_source,
    "first_match_results": args.first_results or None,
    "recorded_at": datetime.now(timezone.utc).isoformat(),
    "rights": args.rights,
    "alliance_teams": alliance_teams,
}
roster_path.write_text(json.dumps(roster, indent=2) + "\n", encoding="utf-8")
print(f"Saved video: {video}")
print(f"Saved verified alliance roster: {roster_path}")
