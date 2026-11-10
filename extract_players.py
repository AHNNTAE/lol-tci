"""match_N.json 들에서 '누가 어느 경기·라인·팀이었는지'만 뽑는다 (소환사별 TCI 분석용).

타임라인은 읽지 않으므로 extract_features.py 보다 훨씬 빠르다.
결과: results/players_full.csv.gz
  match_id, participant_id, puuid, team_id, lane(TOP/JUNGLE/MIDDLE/BOTTOM), role(UTILITY 구분), champion, win

사용법 (lol_tci_lane 폴더에서):
    python extract_players.py --data match_data --workers 2
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from multiprocessing import Pool

import pandas as pd

MIN_GAME_SEC = 20 * 60


def process(path: str) -> list[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as exc:
        print(f"  [skip file] {path}: {exc}")
        return []
    rows = []
    for m in obj if isinstance(obj, list) else [obj]:
        if not m or m["info"].get("gameDuration", 0) < MIN_GAME_SEC:
            continue
        mid = m["metadata"]["matchId"]
        for p in m["info"]["participants"]:
            pos = p.get("teamPosition", "")
            rows.append({
                "match_id": mid,
                "participant_id": int(p["participantId"]),
                "puuid": p.get("puuid", ""),
                "team_id": int(p["teamId"]),
                "lane": "BOTTOM" if pos == "UTILITY" else pos,
                "role": pos,
                "champion": p.get("championName", ""),
                "win": int(bool(p.get("win"))),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="match_data")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.data, "match_*.json")),
                   key=lambda p: int(re.findall(r"(\d+)\.json$", p)[0]))
    print(f"매치 파일 {len(paths)}개 처리 중...")
    rows = []
    with Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(process, paths), 1):
            rows.extend(r)
            if k % 100 == 0 or k == len(paths):
                print(f"  {k}/{len(paths)}")
    df = pd.DataFrame(rows).drop_duplicates(["match_id", "participant_id"])
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "players_full.csv.gz")
    df.to_csv(out, index=False, compression="gzip")
    print(f"완료: 경기 {df.match_id.nunique():,}개, 행 {len(df):,}개 → {out} "
          f"({os.path.getsize(out) / 1e6:.0f} MB)")


if __name__ == "__main__":  # Windows 멀티프로세싱에 필요
    main()
