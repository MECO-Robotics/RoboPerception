"""Load and validate the alliance lineup recorded beside a match video."""
from __future__ import annotations

import json
import math
from pathlib import Path


def validate_alliance_teams(alliance_teams):
    if not isinstance(alliance_teams, dict):
        raise ValueError("alliance_teams must contain red and blue team lists")
    normalized = {}
    for alliance in ("red", "blue"):
        teams = alliance_teams.get(alliance)
        if not isinstance(teams, list) or len(teams) != 3:
            raise ValueError(f"{alliance} alliance must list exactly three teams")
        values = [str(team).strip() for team in teams]
        if any(not value.isdigit() or int(value) <= 0 for value in values):
            raise ValueError(f"{alliance} team numbers must be positive integers")
        if len(set(values)) != 3:
            raise ValueError(f"{alliance} alliance team numbers must be unique")
        normalized[alliance] = values
    if set(normalized["red"]) & set(normalized["blue"]):
        raise ValueError("a team cannot appear on both alliances in one match")
    return normalized


def load_video_roster(video_path):
    """Read the matching .roster.json sidecar, or return None."""
    path = Path(video_path).with_suffix(".roster.json")
    if not path.is_file():
        return None
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["alliance_teams"] = validate_alliance_teams(
        metadata.get("alliance_teams"),
    )
    metadata["roster_file"] = str(path)
    return metadata


def match_team_number(raw_number, alliance, alliance_teams, max_distance=1):
    """Return the unique exact/near roster match, retaining ambiguity as rejection.

    One missing, extra, or misread digit can be recovered from at least three
    observed digits when the match is uniquely closest across both alliances.
    The reader still requires repeated matching samples to confirm.
    """
    raw = str(raw_number or "").strip()
    teams = validate_alliance_teams(alliance_teams)
    if alliance not in teams:
        return None, "rejected_unknown_alliance"
    if raw in teams[alliance]:
        return raw, "exact"
    if len(raw) < 2:
        return None, "rejected_short"

    def distance(left, right):
        previous = list(range(len(right) + 1))
        for i, char_left in enumerate(left, 1):
            current = [i]
            for j, char_right in enumerate(right, 1):
                current.append(min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (char_left != char_right),
                ))
            previous = current
        return previous[-1]

    other_alliance = "blue" if alliance == "red" else "red"
    if raw in teams[other_alliance]:
        return None, "rejected_wrong_alliance"

    def eligible(team):
        # Short fuzzy reads are too ambiguous (e.g. "18" can mean 180 or
        # an OCR error for 16). Exact two-digit reads were accepted above.
        minimum_digits = max(3, math.ceil(len(team) * 2 / 3))
        if len(raw) < minimum_digits:
            return False
        return True

    scored = sorted(
        (distance(raw, team), team) for team in teams[alliance]
        if eligible(team)
    )
    other_scored = sorted(
        (distance(raw, team), team) for team in teams[other_alliance]
        if eligible(team)
    )
    if not scored and not other_scored:
        return None, "rejected_short"
    best_distance = scored[0][0] if scored else math.inf
    best = [team for score, team in scored if score == best_distance]
    other_distance = other_scored[0][0] if other_scored else math.inf
    if other_distance <= max_distance and other_distance <= best_distance:
        return None, (
            "rejected_ambiguous" if other_distance == best_distance
            else "rejected_wrong_alliance"
        )
    if best_distance <= max_distance and len(best) == 1:
        return best[0], "partial"
    if len(best) > 1 and best_distance <= max_distance:
        return None, "rejected_ambiguous"
    return None, "rejected_not_in_roster"
