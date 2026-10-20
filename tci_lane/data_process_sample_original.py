"""Riot match/timeline JSON -> four lane snapshots.

The attached TOP-only example is generalized here. All features are cumulative
blue-minus-red gaps at minute 10; the target is the minute-15 gold gap.
"""

from __future__ import annotations

from dataclasses import dataclass

LANES = ("TOP", "MIDDLE", "JUNGLE", "BOTTOM")
FEATURES = (
    "gold_gap_10",
    "xp_gap_10",
    "cs_gap_10",
    "jungle_cs_gap_10",
    "champion_damage_gap_10",
    "kills_gap_10",
)
INPUT_MS = 10 * 60_000
TARGET_MS = 15 * 60_000


@dataclass(frozen=True)
class LaneSample:
    match_id: str
    lane: str
    features: tuple[float, ...]
    target: float


def _participants(match: dict) -> dict[str, dict[str, tuple[int, ...]]]:
    positions: dict[str, dict[str, list[int]]] = {
        lane: {"blue": [], "red": []} for lane in LANES
    }
    for p in match["info"]["participants"]:
        position = p["teamPosition"]
        if position == "UTILITY":
            position = "BOTTOM"
        if position not in positions:
            continue
        team = p["teamId"]
        if team not in (100, 200):
            raise ValueError(f"Unexpected teamId: {team}")
        positions[position]["blue" if team == 100 else "red"].append(
            int(p["participantId"])
        )
    result = {
        lane: {side: tuple(sorted(ids)) for side, ids in teams.items()}
        for lane, teams in positions.items()
    }
    for lane, teams in result.items():
        expected = 2 if lane == "BOTTOM" else 1
        if any(len(ids) != expected for ids in teams.values()):
            raise ValueError(f"Missing/duplicate position for {lane}: {teams}")
    return result


def _frame_at(frames: list[dict], timestamp: int) -> dict:
    # Riot minute frames commonly land a few milliseconds AFTER the nominal
    # boundary (e.g. 600027ms). Select the nearest snapshot within 30 seconds
    # and use its actual timestamp as the input event cutoff below.
    if not frames:
        raise ValueError(f"No frames for {timestamp}ms")
    frame = min(frames, key=lambda f: abs(int(f["timestamp"]) - timestamp))
    if abs(int(frame["timestamp"]) - timestamp) > 30_000:
        raise ValueError(
            f"No frame close to {timestamp}ms; nearest={frame['timestamp']}ms"
        )
    return frame


def _sum_stat(frame: dict, ids: tuple[int, ...], key: str) -> float:
    return float(sum(frame["participantFrames"][str(pid)][key] for pid in ids))


def _sum_damage(frame: dict, ids: tuple[int, ...]) -> float:
    return float(
        sum(
            frame["participantFrames"][str(pid)]["damageStats"][
                "totalDamageDoneToChampions"
            ]
            for pid in ids
        )
    )


def extract_match(match: dict, timeline: dict) -> list[LaneSample]:
    match_id = str(match["metadata"]["matchId"])
    timeline_id = str(timeline["metadata"]["matchId"])
    if match_id != timeline_id:
        raise ValueError(f"Mismatched match and timeline IDs: {match_id}, {timeline_id}")
    frames = timeline["info"]["frames"]
    at_10 = _frame_at(frames, INPUT_MS)
    at_15 = _frame_at(frames, TARGET_MS)
    positions = _participants(match)

    # Use event timestamps, not containing-frame index: frame event boundaries
    # are not necessarily identical to the snapshot timestamps.
    kills: dict[int, int] = {}
    valid_ids = {
        pid for sides in positions.values() for ids in sides.values() for pid in ids
    }
    for frame in frames:
        for event in frame.get("events", []):
            if event.get("type") != "CHAMPION_KILL":
                continue
            if int(event["timestamp"]) > int(at_10["timestamp"]):
                continue
            killer = int(event.get("killerId", 0))
            if killer in valid_ids:
                kills[killer] = kills.get(killer, 0) + 1

    rows: list[LaneSample] = []
    for lane in LANES:
        blue, red = positions[lane]["blue"], positions[lane]["red"]

        def gap(frame: dict, key: str) -> float:
            return _sum_stat(frame, blue, key) - _sum_stat(frame, red, key)

        x = (
            gap(at_10, "totalGold"),
            gap(at_10, "xp"),
            gap(at_10, "minionsKilled"),
            gap(at_10, "jungleMinionsKilled"),
            _sum_damage(at_10, blue) - _sum_damage(at_10, red),
            float(sum(kills.get(pid, 0) for pid in blue) - sum(kills.get(pid, 0) for pid in red)),
        )
        rows.append(LaneSample(match_id, lane, x, gap(at_15, "totalGold")))
    return rows