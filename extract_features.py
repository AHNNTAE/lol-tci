"""대용량 원본(match_N.json / timeline_N.json 수천 개) → 압축된 분 단위 피처 파일.

원본 JSON 전체(수십 GB)를 한 번에 읽지 않고 파일 쌍(match_N, timeline_N)을 하나씩 처리한다.
  - 파일 쌍마다 결과를 features_cache/part_N.csv.gz 로 저장 → 중간에 끊겨도 이어서 실행 가능
  - 여러 CPU 코어로 병렬 처리 (--workers)
  - 마지막에 하나로 합쳐 아래 두 파일을 만든다
      results/minute_features_full.csv.gz   (라인 × 분 단위 피처, 학습 입력)
      results/end_of_game_full.csv.gz       (경기 종료 시점 격차, 타겟 검증용)

사용법 (lol_tci_lane 폴더에서):
    python extract_features.py --data match_data --workers 4
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from multiprocessing import Pool

import pandas as pd

from tci_lane.data_process import end_of_game_table, extract_match

CACHE = "features_cache"


def _load(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, list) else [obj]


def process_pair(args) -> tuple[str, int, int, int]:
    """한 파일 쌍 처리. (index, 사용 경기 수, 제외 경기 수, 짝 없는 경기 수) 반환."""
    idx, match_path, timeline_path = args
    out_min = os.path.join(CACHE, f"part_{idx}.csv.gz")
    out_end = os.path.join(CACHE, f"end_{idx}.csv.gz")
    if os.path.exists(out_min) and os.path.exists(out_end):
        return idx, -1, 0, 0  # 이미 처리됨
    try:
        matches = {m["metadata"]["matchId"]: m for m in _load(match_path) if m}
        timelines = {t["metadata"]["matchId"]: t for t in _load(timeline_path) if t}
    except Exception as exc:  # 깨진 파일
        print(f"  [skip file] {idx}: {exc}")
        return idx, 0, 0, 0
    ids = sorted(set(matches) & set(timelines))
    unpaired = len(set(matches) ^ set(timelines))
    rows, pairs, skipped = [], [], 0
    for mid in ids:
        try:
            r = extract_match(matches[mid], timelines[mid])  # 20분 미만이면 빈 리스트
        except Exception:
            r = []  # 포지션 누락(다시하기 등), 프레임 결손 경기
        if r:
            rows.extend(r)
            pairs.append((matches[mid], timelines[mid]))
        else:
            skipped += 1
    pd.DataFrame(rows).to_csv(out_min + ".tmp", index=False, compression="gzip")
    end_of_game_table(pairs).to_csv(out_end + ".tmp", index=False, compression="gzip")
    os.replace(out_min + ".tmp", out_min)  # 다 쓴 뒤에만 완료 처리
    os.replace(out_end + ".tmp", out_end)
    return idx, len(pairs), skipped, unpaired


def _read_parts(paths):
    for p in paths:
        try:
            yield pd.read_csv(p)
        except pd.errors.EmptyDataError:  # 빈 파일 쌍(경기 0개)
            continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="match_data")
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)

    jobs = []
    for mp in glob.glob(os.path.join(a.data, "match_*.json")):
        idx = re.findall(r"match_(\d+)\.json$", mp)[0]
        tp = os.path.join(a.data, f"timeline_{idx}.json")
        if os.path.exists(tp):
            jobs.append((idx, mp, tp))
    jobs.sort(key=lambda j: int(j[0]))
    print(f"파일 쌍 {len(jobs)}개, 프로세스 {a.workers}개로 처리합니다.")

    t0, used, skipped, unpaired = time.time(), 0, 0, 0
    with Pool(a.workers) as pool:
        for k, (idx, n, s, u) in enumerate(pool.imap_unordered(process_pair, jobs), 1):
            used += max(n, 0); skipped += s; unpaired += u
            if k % 20 == 0 or k == len(jobs):
                el = time.time() - t0
                print(f"  {k}/{len(jobs)} 완료 · 경과 {el/60:.1f}분 · 남은 예상 {el/k*(len(jobs)-k)/60:.1f}분")

    print("합치는 중...")
    parts = sorted(glob.glob(os.path.join(CACHE, "part_*.csv.gz")))
    ends = sorted(glob.glob(os.path.join(CACHE, "end_*.csv.gz")))
    md = pd.concat(_read_parts(parts), ignore_index=True)
    ed = pd.concat(_read_parts(ends), ignore_index=True)
    md = md.drop_duplicates(["match_id", "lane", "minute"])  # 여러 소환사 기록에 같은 경기 중복 가능
    ed = ed.drop_duplicates(["match_id", "lane"])
    md.to_csv(os.path.join(a.out, "minute_features_full.csv.gz"), index=False, compression="gzip")
    ed.to_csv(os.path.join(a.out, "end_of_game_full.csv.gz"), index=False, compression="gzip")
    size = os.path.getsize(os.path.join(a.out, "minute_features_full.csv.gz")) / 1e6
    print(f"완료: 경기 {md.match_id.nunique():,}개 (이번 실행 제외 {skipped:,}, 짝 없음 {unpaired:,}), "
          f"행 {len(md):,}개 → results/minute_features_full.csv.gz ({size:.0f} MB)")


if __name__ == "__main__":  # Windows 멀티프로세싱에 필요
    main()
