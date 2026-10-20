"""Riot match/timeline JSON -> 라인별 '분 단위 누적 격차' 테이블.

data_process_sample.py(10분 스냅숏 1개 → 15분 골드 격차)를 다음과 같이 변형했다.
  1) 스냅숏 1개가 아니라 1분 ~ 경기 종료 직전까지 '모든 분'을 샘플로 사용 (TCI(x, t))
  2) 모든 값은 '경기 시작부터 t분까지의 누적치'를 블루(100) - 레드(200)로 뺀 '격차'로 통일
  3) 프레임 스탯(CS, 피해량 등)과 이벤트(킬, 와드, 포탑 방패, 오브젝트) 모두
     프레임의 실제 timestamp 이하만 집계하여 미래 정보 누출을 막음
  4) BOTTOM = ADC(BOTTOM) + 서포터(UTILITY) 합산

출력 컬럼
  match_id, lane, minute, blue_win, game_minutes,
  gold_gap, xp_gap, dmg_gap         (타겟 후보: 원단위 격차)
  피처 풀 (FEATURE_POOL)            (입력 후보)
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import pandas as pd

LANES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM")
LANE_OF_TURRET = {"TOP_LANE": "TOP", "MID_LANE": "MIDDLE", "BOT_LANE": "BOTTOM"}
MIN_GAME_SEC = 20 * 60  # 1.1 분석 데이터 기준: 20분 이상 경기만 사용
START_MINUTE = 3        # 1~2분은 첫 미니언 웨이브 전후로 격차가 사실상 0 → 정규화가 불안정하여 제외

FEATURE_POOL = (
    "minute",           # 경과 시간 t
    "cs_gap",           # 라인 미니언 처치 격차
    "jungle_cs_gap",    # 정글 몬스터 처치 격차
    "dmg_gap",          # 챔피언에게 가한 피해량 격차 (타겟이 gold+xp 이므로 피처로 사용 가능)
    "kills_gap",
    "deaths_gap",
    "assists_gap",
    "solo_kills_gap",   # 어시스트 없는 킬
    "dmg_taken_gap",    # 받은 피해량 격차
    "cc_time_gap",      # 상대에게 건 CC 시간 격차 (timeEnemySpentControlled, 초)
    "wards_placed_gap",
    "wards_killed_gap",
    "plates_gap",       # 해당 라인 포탑 방패 획득 격차
    "turrets_gap",      # 해당 라인 포탑 파괴 격차
    "dragons_gap",      # 팀 드래곤 처치 격차
    "grubs_herald_gap", # 팀 공허 유충 + 전령 처치 격차
    "barons_gap",       # 팀 바론 처치 격차
)


def load_pairs(data_dir: str):
    """data_dir 안의 match_*.json / timeline_*.json 을 matchId 로 짝지어 반환."""
    matches, timelines = {}, {}
    for path in sorted(glob.glob(os.path.join(data_dir, "match_*.json"))):
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        for m in obj if isinstance(obj, list) else [obj]:
            matches[m["metadata"]["matchId"]] = m
    for path in sorted(glob.glob(os.path.join(data_dir, "timeline_*.json"))):
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        for t in obj if isinstance(obj, list) else [obj]:
            timelines[t["metadata"]["matchId"]] = t
    ids = sorted(set(matches) & set(timelines))
    return [(matches[i], timelines[i]) for i in ids]


def _positions(match: dict) -> dict[str, dict[int, tuple[int, ...]]]:
    pos = {lane: {100: [], 200: []} for lane in LANES}
    for p in match["info"]["participants"]:
        lane = "BOTTOM" if p["teamPosition"] == "UTILITY" else p["teamPosition"]
        if lane in pos:
            pos[lane][p["teamId"]].append(int(p["participantId"]))
    for lane, sides in pos.items():
        need = 2 if lane == "BOTTOM" else 1
        if any(len(v) != need for v in sides.values()):
            raise ValueError(f"position error {lane}: {sides}")
    return {lane: {s: tuple(v) for s, v in sides.items()} for lane, sides in pos.items()}


def _blue_win(match: dict) -> int:
    for team in match["info"]["teams"]:
        if team["teamId"] == 100:
            return int(team["win"])
    raise ValueError("no blue team")


def extract_match(match: dict, timeline: dict) -> list[dict]:
    """한 경기 -> (라인 4개 × 분) 행 목록."""
    if match["info"]["gameDuration"] < MIN_GAME_SEC:
        return []
    mid = match["metadata"]["matchId"]
    pos = _positions(match)
    frames = timeline["info"]["frames"]
    blue_win = _blue_win(match)
    game_min = match["info"]["gameDuration"] / 60

    # 모든 이벤트를 시간순으로 한 줄에 펼친다.
    events = sorted(
        (e for f in frames for e in f.get("events", [])), key=lambda e: e["timestamp"]
    )

    # 참가자별 누적 카운터 / 라인·팀별 누적 카운터
    per_p = defaultdict(lambda: defaultdict(int))     # per_p[pid][stat]
    per_lane = defaultdict(lambda: defaultdict(int))  # per_lane[(lane, team)][stat]
    per_team = defaultdict(lambda: defaultdict(int))  # per_team[team][stat]

    def team_of(pid: int) -> int:
        return 100 if 1 <= pid <= 5 else 200

    def apply(e: dict):
        ty = e["type"]
        if ty == "CHAMPION_KILL":
            k, v = int(e.get("killerId", 0)), int(e["victimId"])
            assists = e.get("assistingParticipantIds", []) or []
            if k > 0:
                per_p[k]["kills"] += 1
                if not assists:
                    per_p[k]["solo_kills"] += 1
            per_p[v]["deaths"] += 1
            for a in assists:
                per_p[int(a)]["assists"] += 1
        elif ty == "WARD_PLACED":
            if e.get("wardType") not in (None, "UNDEFINED") and e.get("creatorId", 0) > 0:
                per_p[int(e["creatorId"])]["wards_placed"] += 1
        elif ty == "WARD_KILL":
            if e.get("killerId", 0) > 0:
                per_p[int(e["killerId"])]["wards_killed"] += 1
        elif ty == "TURRET_PLATE_DESTROYED":
            lane = LANE_OF_TURRET.get(e.get("laneType"))
            if lane:  # teamId = 방패를 잃은 팀 -> 상대 팀이 획득
                taker = 200 if e["teamId"] == 100 else 100
                per_lane[(lane, taker)]["plates"] += 1
        elif ty == "BUILDING_KILL" and e.get("buildingType") == "TOWER_BUILDING":
            lane = LANE_OF_TURRET.get(e.get("laneType"))
            if lane:
                taker = 200 if e["teamId"] == 100 else 100
                per_lane[(lane, taker)]["turrets"] += 1
        elif ty == "ELITE_MONSTER_KILL":
            team = e.get("killerTeamId") or team_of(int(e.get("killerId", 0)))
            mt = e.get("monsterType")
            if mt == "DRAGON":
                per_team[team]["dragons"] += 1
            elif mt in ("HORDE", "RIFTHERALD"):
                per_team[team]["grubs_herald"] += 1
            elif mt == "BARON_NASHOR":
                per_team[team]["barons"] += 1

    rows: list[dict] = []
    ei = 0
    end_minute = int(match["info"]["gameDuration"] // 60)
    for minute in range(1, min(len(frames), end_minute + 1)):
        fr = frames[minute]
        ts = int(fr["timestamp"])
        while ei < len(events) and events[ei]["timestamp"] <= ts:
            apply(events[ei])
            ei += 1
        if minute < START_MINUTE:
            continue
        pf = fr["participantFrames"]

        def fsum(ids, getter):
            return float(sum(getter(pf[str(i)]) for i in ids))

        for lane in LANES:
            b, r = pos[lane][100], pos[lane][200]

            def gap(getter):
                return fsum(b, getter) - fsum(r, getter)

            def egap(stat):
                return float(sum(per_p[i][stat] for i in b) - sum(per_p[i][stat] for i in r))

            row = {
                "match_id": mid,
                "lane": lane,
                "minute": minute,
                "blue_win": blue_win,
                "game_minutes": game_min,
                # ---- 타겟 후보 (원단위 격차)
                "gold_gap": gap(lambda p: p["totalGold"]),
                "xp_gap": gap(lambda p: p["xp"]),
                "dmg_gap": gap(lambda p: p["damageStats"]["totalDamageDoneToChampions"]),
                # ---- 피처 풀
                "cs_gap": gap(lambda p: p["minionsKilled"]),
                "jungle_cs_gap": gap(lambda p: p["jungleMinionsKilled"]),
                "dmg_taken_gap": gap(lambda p: p["damageStats"]["totalDamageTaken"]),
                "cc_time_gap": gap(lambda p: p["timeEnemySpentControlled"]) / 1000.0,
                "kills_gap": egap("kills"),
                "deaths_gap": egap("deaths"),
                "assists_gap": egap("assists"),
                "solo_kills_gap": egap("solo_kills"),
                "wards_placed_gap": egap("wards_placed"),
                "wards_killed_gap": egap("wards_killed"),
                "plates_gap": float(per_lane[(lane, 100)]["plates"] - per_lane[(lane, 200)]["plates"]),
                "turrets_gap": float(per_lane[(lane, 100)]["turrets"] - per_lane[(lane, 200)]["turrets"]),
                "dragons_gap": float(per_team[100]["dragons"] - per_team[200]["dragons"]),
                "grubs_herald_gap": float(per_team[100]["grubs_herald"] - per_team[200]["grubs_herald"]),
                "barons_gap": float(per_team[100]["barons"] - per_team[200]["barons"]),
            }
            rows.append(row)
    return rows


def end_of_game_table(pairs) -> pd.DataFrame:
    """경기 종료 시점(match 데이터)의 라인별/팀 전체 격차 — 타겟 후보 검증용."""
    rows = []
    for match, _ in pairs:
        if match["info"]["gameDuration"] < MIN_GAME_SEC:
            continue
        pos = _positions(match)
        bw = _blue_win(match)
        by_id = {p["participantId"]: p for p in match["info"]["participants"]}
        groups = {lane: pos[lane] for lane in LANES}
        groups["TEAM"] = {100: tuple(range(1, 6)), 200: tuple(range(6, 11))}
        for name, sides in groups.items():
            def g(key):
                return float(sum(by_id[i][key] for i in sides[100]) - sum(by_id[i][key] for i in sides[200]))
            rows.append({
                "match_id": match["metadata"]["matchId"], "lane": name, "blue_win": bw,
                "gold_gap": g("goldEarned"), "xp_gap": g("champExperience"),
                "dmg_gap": g("totalDamageDealtToChampions"),
            })
    return pd.DataFrame(rows)


def build_minute_table(pairs) -> pd.DataFrame:
    rows = []
    for match, timeline in pairs:
        rows.extend(extract_match(match, timeline))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    pairs = load_pairs(data_dir)
    df = build_minute_table(pairs)
    print(f"matches={df.match_id.nunique()} rows={len(df)}")
    print(df.groupby("lane").size())
    print(df[df.minute == 15].groupby("lane")[["gold_gap", "xp_gap", "cs_gap"]].describe().T)
