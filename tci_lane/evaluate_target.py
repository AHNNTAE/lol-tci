"""타겟 변수(y) 후보 평가 — evaluate_y.py 를 라인·시점·복합지표로 확장.

후보: gold, xp, dmg, gold+xp, gold+dmg, gold+xp+dmg
  - 단일 지표: 원단위 격차 그대로
  - 복합 지표: 각 격차를 '해당 분·해당 라인'의 RMS(제곱평균제곱근)로 나눈 뒤 평균
      z_k(t) = gap_k(t) / RMS_k(t),  y = mean_k z_k(t)
    평균을 빼지 않고 RMS로만 나누는 이유: 0 = '대등'이라는 부호 의미를 보존하기 위함.
평가: 블루팀 기준 y>0 이면 블루 승리로 예측했을 때의 정확도(accuracy)와
      부호 대신 연속값으로 본 ROC-AUC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CANDIDATES = {
    "gold": ("gold_gap",),
    "xp": ("xp_gap",),
    "dmg": ("dmg_gap",),
    "gold+xp": ("gold_gap", "xp_gap"),
    "gold+dmg": ("gold_gap", "dmg_gap"),
    "gold+xp+dmg": ("gold_gap", "xp_gap", "dmg_gap"),
}


def rms(x: pd.Series) -> float:
    v = float(np.sqrt(np.mean(np.square(x))))
    return v if v > 0 else 1.0


def composite(df: pd.DataFrame, cols: tuple[str, ...], scale: dict | None = None) -> pd.Series:
    """cols 격차들의 RMS-정규화 평균. scale 이 주어지면 (minute, col)->RMS 사전을 사용."""
    if len(cols) == 1:
        return df[cols[0]].astype(float)
    parts = []
    for c in cols:
        if scale is None:
            s = df.groupby("minute")[c].transform(rms)
        else:
            s = df["minute"].map(lambda m: scale[c].get(m, scale[c]["_last"]))
        parts.append(df[c] / s)
    return sum(parts) / len(parts)


def add_team_rows(minute_df: pd.DataFrame) -> pd.DataFrame:
    keys = ["match_id", "minute", "blue_win"]
    team = minute_df.groupby(keys)[["gold_gap", "xp_gap", "dmg_gap"]].sum().reset_index()
    team["lane"] = "TEAM"
    return pd.concat([minute_df, team], ignore_index=True)


def _score(y: pd.Series, win: pd.Series) -> tuple[float, float]:
    acc = float(((y > 0).astype(int) == win).mean())
    auc = float(roc_auc_score(win, y)) if win.nunique() == 2 else float("nan")
    return acc, auc


def evaluate(minute_df: pd.DataFrame, end_df: pd.DataFrame, minutes=(10, 15, 20)) -> pd.DataFrame:
    df = add_team_rows(minute_df)
    out = []
    for lane in ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "TEAM"]:
        for name, cols in CANDIDATES.items():
            for m in minutes:
                d = df[(df.lane == lane) & (df.minute == m)]
                acc, auc = _score(composite(d, cols), d.blue_win)
                out.append({"lane": lane, "target": name, "time": f"{m}min", "acc": acc, "auc": auc, "n": len(d)})
            d = end_df[end_df.lane == lane]
            # 종료 시점은 경기당 한 행 → 라인 내 RMS 로 정규화
            parts = [d[c] / rms(d[c]) for c in cols]
            y = d[cols[0]] if len(cols) == 1 else sum(parts) / len(parts)
            acc, auc = _score(y, d.blue_win)
            out.append({"lane": lane, "target": name, "time": "end", "acc": acc, "auc": auc, "n": len(d)})
    return pd.DataFrame(out)


def per_minute_curve(minute_df: pd.DataFrame, cols: tuple[str, ...], max_minute=25) -> pd.DataFrame:
    df = add_team_rows(minute_df)
    out = []
    for lane in ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "TEAM"]:
        for m in range(1, max_minute + 1):
            d = df[(df.lane == lane) & (df.minute == m)]
            if d.blue_win.nunique() < 2:
                continue
            acc, auc = _score(composite(d, cols), d.blue_win)
            out.append({"lane": lane, "minute": m, "acc": acc, "auc": auc})
    return pd.DataFrame(out)
