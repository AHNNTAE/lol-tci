"""TCI로 챌린저 소환사의 실력을 평가한다 (보고서 4장, 발표 결론).

입력
  results/minute_features_full.csv.gz   (extract_features.py)
  results/players_full.csv.gz           (extract_players.py)
  results_full/models/TCI_<LANE>.joblib (run_all.py)
  puuid_challengers.txt                 (분석 대상 챌린저 300명)
출력 (results_players/)
  player_summary.csv      소환사별 경기 수, 주 라인, 승률, 라인전 TCI, 행동별 기여
  reliability.json        결과 1: 승률 vs TCI 의 '같은 소환사 일관성' (홀짝 경기 나눠 상관)
  figures/fig7_reliability.png, fig8_tci_vs_winrate.png, fig9_player_profiles.png

소환사의 TCI = 그 소환사 쪽에서 본 라인 TCI (블루면 그대로, 레드면 부호 반대).
라인전 TCI = 경기마다 10~20분 TCI 평균 → 소환사의 경기들에 대해 다시 평균.
소환사 이름은 공개하지 않고 '소환사 A, B' 처럼 익명으로 표시한다.
"""

from __future__ import annotations

import argparse
import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_all import COLOR, INK, MUTED, save  # 그림 스타일 공유
from tci_lane.data_process import LANES

GROUPS = {  # 행동별 기여를 묶어서 보여 줄 범주
    "파밍": ["cs_gap", "jungle_cs_gap"],
    "교전": ["kills_gap", "assists_gap", "solo_kills_gap", "dmg_gap", "cc_time_gap"],
    "생존": ["deaths_gap", "dmg_taken_gap"],
    "운영": ["plates_gap", "turrets_gap", "dragons_gap", "grubs_herald_gap", "barons_gap",
           "wards_placed_gap", "wards_killed_gap", "minute"],
}
EARLY, LATE = 10, 20  # 라인전 구간 (분)


def lane_tci(md: pd.DataFrame, model_dir: str) -> pd.DataFrame:
    """각 (경기, 라인, 분)의 TCI 와 행동 범주별 기여(블루 기준)를 계산."""
    out = []
    for lane in LANES:
        b = joblib.load(os.path.join(model_dir, f"TCI_{lane}.joblib"))
        pipe, feats = b["model"], b["features"]
        d = md[(md.lane == lane) & md.minute.between(EARLY, LATE)].copy()
        X = d[feats].to_numpy()
        tn = pipe.steps[0][1]
        scaler, lasso = [s[1] for s in pipe.steps[1][1].steps]
        z = scaler.transform(tn.transform(X))
        contrib = z * lasso.coef_
        d["tci"] = contrib.sum(axis=1) + lasso.intercept_
        for g, cols in GROUPS.items():
            idx = [feats.index(c) for c in cols if c in feats]
            d[f"c_{g}"] = contrib[:, idx].sum(axis=1) if idx else 0.0
        out.append(d[["match_id", "lane", "minute", "tci"] + [f"c_{g}" for g in GROUPS]])
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="results/minute_features_full.csv.gz")
    ap.add_argument("--players", default="results/players_full.csv.gz")
    ap.add_argument("--models", default="results_full/models")
    ap.add_argument("--challengers", default="puuid_challengers.txt")
    ap.add_argument("--out", default="results_players")
    ap.add_argument("--min-games", type=int, default=30)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)

    md = pd.read_csv(a.features)
    pl = pd.read_csv(a.players)
    ch = set(open(a.challengers, encoding="utf-8").read().split())

    # 1) 경기별 라인전 TCI (블루 기준) ----------------------------------------
    t = lane_tci(md, a.models)
    gcols = ["tci"] + [f"c_{g}" for g in GROUPS]
    game = t.groupby(["match_id", "lane"])[gcols].mean().reset_index()

    # 2) 소환사 시점으로 변환 (레드면 부호 반대) -------------------------------
    p = pl[pl.puuid.isin(ch)].merge(game, on=["match_id", "lane"], how="inner")
    sign = np.where(p.team_id == 100, 1.0, -1.0)
    for c in gcols:
        p[c] = p[c] * sign
    order = pl[["match_id"]].drop_duplicates().reset_index(drop=True)
    order["game_no"] = np.arange(len(order))  # 경기 순서(파일 순서 ≈ 시간 순)
    p = p.merge(order, on="match_id")

    # 3) 소환사별 요약 ----------------------------------------------------------
    main_lane = p.groupby("puuid").lane.agg(lambda s: s.value_counts().index[0])
    p = p.merge(main_lane.rename("main_lane").reset_index(), on="puuid")
    pm = p[p.lane == p.main_lane]  # 주 라인 경기만 사용 (라인마다 TCI 척도가 같도록)
    summ = pm.groupby("puuid").agg(
        lane=("main_lane", "first"), games=("win", "size"), win_rate=("win", "mean"),
        tci=("tci", "mean"), **{f"c_{g}": (f"c_{g}", "mean") for g in GROUPS}).reset_index()
    summ["lane_share"] = summ.games / p.groupby("puuid").size().reindex(summ.puuid).to_numpy()
    summ = summ[summ.games >= a.min_games].sort_values("tci", ascending=False).reset_index(drop=True)
    summ["tci_rank"] = np.arange(1, len(summ) + 1)
    summ["win_rank"] = summ.win_rate.rank(ascending=False, method="min").astype(int)
    summ["alias"] = [f"소환사 {i:03d}" for i in range(1, len(summ) + 1)]
    summ.drop(columns="puuid").to_csv(os.path.join(a.out, "player_summary.csv"), index=False)

    # 4) 결과 1: 일관성 (홀수 번째 경기 vs 짝수 번째 경기) -----------------------
    pm2 = pm[pm.puuid.isin(summ.puuid)].sort_values(["puuid", "game_no"]).copy()
    pm2["half"] = pm2.groupby("puuid").cumcount() % 2
    halves = pm2.groupby(["puuid", "half"])[["win", "tci"]].mean().unstack("half")
    r_win = float(np.corrcoef(halves[("win", 0)], halves[("win", 1)])[0, 1])
    r_tci = float(np.corrcoef(halves[("tci", 0)], halves[("tci", 1)])[0, 1])
    r_tw = float(np.corrcoef(summ.tci, summ.win_rate)[0, 1])
    rel = {"n_players": int(len(summ)), "min_games": a.min_games,
           "median_games": float(summ.games.median()),
           "split_half_r_winrate": r_win, "split_half_r_tci": r_tci,
           "corr_tci_winrate": r_tw,
           "lanes": summ.lane.value_counts().to_dict()}
    json.dump(rel, open(os.path.join(a.out, "reliability.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(rel, ensure_ascii=False, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, key, name, col, r in [(axes[0], "win", "승률", "#8a8984", r_win),
                                  (axes[1], "tci", "라인전 TCI", "#2a78d6", r_tci)]:
        x, y = halves[(key, 0)], halves[(key, 1)]
        ax.scatter(x, y, s=22, alpha=0.7, color=col, edgecolors="none")
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1)
        ax.set_xlabel(f"홀수 번째 경기들의 {name}")
        ax.set_ylabel(f"짝수 번째 경기들의 {name}")
        ax.set_title(f"{name}: 같은 소환사 일관성 r = {r:.2f}")
    fig.suptitle(f"챌린저 {len(summ)}명 · 경기를 둘로 나눠도 같은 값이 나오는가", fontweight="bold")
    save(fig, a.out, "fig7_reliability.png")

    # 5) 결과 2: TCI vs 승률 지도 --------------------------------------------
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for lane in LANES:
        s = summ[summ.lane == lane]
        ax.scatter(s.tci, s.win_rate, s=28, alpha=0.8, color=COLOR[lane], label=lane,
                   edgecolors="white", linewidths=0.6)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.axhline(0.5, color=MUTED, lw=0.8)
    ax.set_xlabel("라인전 TCI (10~20분 평균, 0 = 상대와 대등)")
    ax.set_ylabel("승률")
    ax.set_title(f"TCI와 승률 (r = {r_tw:.2f})")
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    save(fig, a.out, "fig8_tci_vs_winrate.png")

    # 6) 같은 라인 두 소환사의 '스타일' 비교 (행동 범주별 기여) ------------------
    # 같은 라인·150경기 이상·승률 차이 1%p 이내인 쌍 중 TCI 차이가 가장 큰 두 명
    best, pick = -1.0, summ.iloc[:2]
    for _, g in summ[summ.games >= 150].groupby("lane"):
        g = g.reset_index(drop=True)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                if abs(g.win_rate[i] - g.win_rate[j]) < 0.01 and abs(g.tci[i] - g.tci[j]) > best:
                    best, pick = abs(g.tci[i] - g.tci[j]), g.iloc[[i, j]]
    quad = {"tci+_win+": int(((summ.tci > 0) & (summ.win_rate > .5)).sum()),
            "tci+_win-": int(((summ.tci > 0) & (summ.win_rate < .5)).sum()),
            "tci-_win+": int(((summ.tci < 0) & (summ.win_rate > .5)).sum()),
            "tci-_win-": int(((summ.tci < 0) & (summ.win_rate < .5)).sum())}
    rel.update({"quadrants": quad, "pair": pick[["alias", "lane", "games", "win_rate", "tci",
                                                 "tci_rank", "win_rank"]].to_dict("records")})
    json.dump(rel, open(os.path.join(a.out, "reliability.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=float)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    gs = list(GROUPS)
    w = 0.38
    for i, (_, r) in enumerate(pick.iterrows()):
        vals = [r[f"c_{g}"] for g in gs]
        ax.bar(np.arange(len(gs)) + (i - 0.5) * w, vals, width=w,
               color=["#2a78d6", "#eb6834"][i],
               label=f"{r.alias} · 승률 {r.win_rate:.1%} · TCI {r.tci:+.2f} ({r.tci_rank}위)")
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(range(len(gs)), gs)
    ax.set_ylabel("TCI 기여 (평균)")
    ax.set_title(f"승률이 같은 두 {pick.lane.iloc[0]} 소환사: TCI는 정반대")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="x", visible=False)
    save(fig, a.out, "fig9_player_profiles.png")
    print(pick[["alias", "lane", "games", "win_rate", "tci"] + [f"c_{g}" for g in gs]].to_string())


if __name__ == "__main__":
    main()
