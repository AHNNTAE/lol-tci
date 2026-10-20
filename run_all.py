"""전체 파이프라인: 데이터 → 타겟 평가 → 라인별 모델 비교 → 최종 모델 → 그림/표.

사용법
  (소규모) python run_all.py --data data --out results
           data/ 폴더의 match_*.json, timeline_*.json 을 직접 읽는다.
  (대규모) python extract_features.py --data match_data          # 먼저 피처 파일 생성
           python run_all.py --features results/minute_features_full.csv.gz \
                             --end results/end_of_game_full.csv.gz --out results_full
           모델 비교는 무작위 --compare-matches 경기로 하고(SVR 제외),
           선택된 최종 모델은 전체 경기로 다시 교차검증·학습한다.
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
from matplotlib import font_manager

from tci_lane.data_process import LANES, build_minute_table, end_of_game_table, load_pairs
from tci_lane.evaluate_target import CANDIDATES, evaluate, per_minute_curve
from tci_lane.train import LANE_FEATURES, cross_validate_lane, fit_final, summarize

# ---------------------------------------------------------------- 그림 스타일
for name in ("Noto Sans CJK KR", "NanumGothic", "Malgun Gothic", "AppleGothic"):
    if any(name in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams.update({"axes.unicode_minus": False, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
                     "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984", "figure.dpi": 130,
                     "axes.titleweight": "bold"})
COLOR = {"TOP": "#2a78d6", "JUNGLE": "#eb6834", "MIDDLE": "#1baf7a", "BOTTOM": "#eda100",
         "TEAM": "#52514e"}
INK, MUTED = "#0b0b0b", "#8a8984"


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", name), bbox_inches="tight")
    plt.close(fig)


def main(data_dir: str, out: str, features: str | None = None, end: str | None = None,
         compare_matches: int = 2000, reuse_compare: bool = False):
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    os.makedirs(os.path.join(out, "models"), exist_ok=True)

    # 1) 데이터 -------------------------------------------------------------
    if features:
        md, ed = pd.read_csv(features), pd.read_csv(end)
    else:
        pairs = load_pairs(data_dir)
        md = build_minute_table(pairs)
        ed = end_of_game_table(pairs)
        md.to_csv(os.path.join(out, "minute_features.csv"), index=False)
    n_match = md.match_id.nunique()
    gm = md.groupby("match_id").game_minutes.first()
    data_info = {"n_matches": int(n_match), "n_rows": int(len(md)),
                 "rows_per_lane": md.groupby("lane").size().astype(int).to_dict(),
                 "blue_win_rate": float(md.groupby("match_id").blue_win.first().mean()),
                 "game_minutes": {k: float(v) for k, v in gm.describe().items()}}
    print(f"[data] matches={n_match}, minute-rows={len(md)}")

    # 2) 타겟 후보 평가 -----------------------------------------------------
    tgt = evaluate(md, ed)
    tgt.to_csv(os.path.join(out, "target_candidates.csv"), index=False)
    acc = tgt.pivot_table(index=["target"], columns=["lane", "time"], values="acc")
    lane_mean = tgt[tgt.lane != "TEAM"].groupby(["target", "time"]).acc.mean().unstack()
    lane_mean = lane_mean[["10min", "15min", "20min", "end"]]
    lane_mean.to_csv(os.path.join(out, "target_lane_mean_acc.csv"))
    print("[target] 4개 라인 평균 승패 예측 정확도\n", lane_mean.round(3))

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    order = list(CANDIDATES)
    im = ax.imshow(lane_mean.loc[order].to_numpy(), cmap="Blues", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(4), ["10분", "15분", "20분", "종료"])
    ax.set_yticks(range(len(order)), order)
    ax.grid(False)
    for i, t in enumerate(order):
        for j, c in enumerate(lane_mean.columns):
            v = lane_mean.loc[t, c]
            ax.text(j, i, f"{v:.1%}", ha="center", va="center", fontsize=9,
                    color="white" if v > 0.8 else INK)
    ax.set_title("타겟 후보별 승패 예측 정확도 (4개 라인 평균)")
    fig.colorbar(im, ax=ax, fraction=0.04)
    save(fig, out, "fig1_target_candidates.png")

    curve = per_minute_curve(md, CANDIDATES["gold+xp"])
    curve.to_csv(os.path.join(out, "target_goldxp_curve.csv"), index=False)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for lane in ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "TEAM"]:
        c = curve[curve.lane == lane]
        ax.plot(c.minute, c.acc, color=COLOR[lane], lw=2.4 if lane == "TEAM" else 1.8,
                ls="--" if lane == "TEAM" else "-", label=lane)
    ax.axhline(0.5, color=MUTED, lw=0.8)
    ax.set_xlabel("경기 시간 (분)")
    ax.set_ylabel("승패 예측 정확도")
    ax.set_ylim(0.3, 1.02)
    ax.set_title("정규화 gold+xp 격차의 분 단위 승패 예측 정확도")
    ax.legend(ncol=5, frameon=False, loc="lower right", fontsize=8)
    save(fig, out, "fig2_goldxp_accuracy_by_minute.png")

    # 3) 라인별 모델 비교 ----------------------------------------------------
    large = n_match > compare_matches
    if large:  # 비교는 무작위 표본 경기로 (속도), 최종 평가는 4)에서 전체 경기로
        rng = np.random.default_rng(0)
        keep = rng.choice(md.match_id.unique(), compare_matches, replace=False)
        md_cmp = md[md.match_id.isin(keep)]
    else:
        md_cmp = md
    seeds = (0, 1, 2) if md_cmp.match_id.nunique() <= 300 else (0,)
    exclude = ("SVR(RBF)",) if len(md_cmp) / 4 > 30000 else ()
    cmp_desc = (f"{md_cmp.match_id.nunique():,}경기 · 경기 단위 5-fold"
                + (f" × {len(seeds)}회" if len(seeds) > 1 else ""))
    print(f"[cv] 모델 비교: {cmp_desc}, 제외 모델: {exclude or '없음'}")
    cvs, store = [], {}
    cached = os.path.join(out, "cv_all_folds.csv")
    if reuse_compare and os.path.exists(cached):  # 오래 걸리는 비교 단계 재사용
        cv_all = pd.read_csv(cached)
        print("[cv] 이전 비교 결과 재사용:", cached)
    else:
        for lane in LANES:
            cv, d, oof = cross_validate_lane(md_cmp, lane, seeds=seeds, exclude=exclude)
            cvs.append(cv)
            store[lane] = (d, oof)
            print(f"[cv] {lane} done")
        cv_all = pd.concat(cvs)
    cv_all.to_csv(os.path.join(out, "cv_all_folds.csv"), index=False)
    summ = summarize(cv_all)
    summ.to_csv(os.path.join(out, "cv_summary.csv"), index=False)

    # 최종 모델 선택: 기준선 제외, OOF RMSE 최소 모델과 1% 이내로 차이 나는 후보는 '동률'로 보고
    # 더 단순·해석 가능한 모델을 우선한다 (PREFER 순서). 표본이 클수록 선형 모델끼리 차이가 0에 수렴하기 때문.
    PREFER = ["Lasso", "Ridge", "LinearRegression", "HistGradientBoosting", "RandomForest",
              "MLP(64-32)", "KNN(k=15)", "SVR(RBF)"]
    cand = summ[~summ.model.str.startswith("Mean")].copy()
    cand["base"] = cand.model.str.split(" [", regex=False).str[0]
    cand["rank"] = cand.base.map(PREFER.index)
    picks = []
    for lane, g in cand.groupby("lane"):
        tied = g[g.RMSE <= g.RMSE.min() * 1.01]
        picks.append(tied.sort_values(["rank", "RMSE"]).iloc[0])
    best = pd.DataFrame(picks).set_index("lane")
    print("[select]\n", best[["model", "RMSE", "R2"]])

    # 그림: 라인별 모델 RMSE (time-norm vs raw)
    ORDER = ["Mean(baseline)", "LinearRegression", "Ridge", "Lasso", "KNN(k=15)", "SVR(RBF)",
             "RandomForest", "HistGradientBoosting", "MLP(64-32)"]
    present = {m.split(" [")[0] for m in summ.model.unique()}
    models = [m for m in ORDER if m in present]
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.2), sharey=True)
    for ax, lane in zip(axes, LANES):
        s = summ[summ.lane == lane].set_index("model")
        y = np.arange(len(models))
        raw = [s.RMSE.get(f"{m} [raw]", np.nan) for m in models]
        tn = [s.RMSE.get(f"{m} [time-norm]", np.nan) for m in models]
        ax.barh(y + 0.2, raw, height=0.38, color="#c3c2b7", label="원 격차 피처")
        ax.barh(y - 0.2, tn, height=0.38, color=COLOR[lane], label="시점 정규화 피처")
        ax.set_yticks(y, models)
        ax.invert_yaxis()
        ax.set_title(lane)
        ax.set_xlabel("OOF RMSE (정규화 단위)")
        ax.set_xlim(0, 1.1)
        ax.grid(axis="y", visible=False)
        ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle(f"라인별 모델 비교 — {cmp_desc} 교차검증", fontweight="bold")
    save(fig, out, "fig3_model_comparison.png")

    # 4) 최종 모델 해석 ------------------------------------------------------
    # 대규모일 때: 선택된 모델과 기준선을 '전체 경기'로 다시 교차검증
    if large:
        fcv = []
        for lane in LANES:
            only = [best.loc[lane, "model"], "Mean(baseline) [time-norm]"]
            cv, d, oof = cross_validate_lane(md, lane, seeds=(0,), only=only)
            fcv.append(cv)
            store[lane] = (d, oof)
            print(f"[full-cv] {lane} done")
        final_summ = summarize(pd.concat(fcv))
        final_summ.to_csv(os.path.join(out, "cv_summary_final_full.csv"), index=False)
    else:
        final_summ = summ
    fbest = final_summ.set_index(["lane", "model"])

    metrics = {**data_info, "compare": cmp_desc, "final_eval": "전체 경기" if large else cmp_desc,
               "lanes": {}}
    coef_rows = []
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    for ax, lane in zip(axes, LANES):
        full = best.loc[lane, "model"]
        name, tag = full.split(" [")
        tnorm = tag.startswith("time")
        model, ts, d, y = fit_final(md, lane, name, tnorm)
        joblib.dump({"model": model, "target_scaler": ts, "features": LANE_FEATURES[lane]},
                    os.path.join(out, "models", f"TCI_{lane}.joblib"))
        feats = LANE_FEATURES[lane]
        # 계수 (선형 모델일 때)
        last = model.steps[-1][1] if hasattr(model, "steps") else model
        inner = last.steps[-1][1] if hasattr(last, "steps") else last
        if hasattr(inner, "coef_"):
            if hasattr(inner, "alpha_"):
                metrics.setdefault("alpha", {})[lane] = float(inner.alpha_)
                metrics.setdefault("intercept", {})[lane] = float(inner.intercept_)
            for f, w in zip(feats, inner.coef_):
                coef_rows.append({"lane": lane, "feature": f, "coef": float(w)})

        # OOF 예측 산점도 (seed 0)
        y_true, oof = store[lane][1][0]
        pred = oof[full]
        if len(pred) > 6000:  # 점이 너무 많으면 표본만 그림
            pick = np.random.default_rng(0).choice(len(pred), 6000, replace=False)
            y_true, pred = y_true[pick], pred[pick]
        ax.scatter(y_true, pred, s=6, alpha=0.35, color=COLOR[lane], edgecolors="none")
        lim = [-3.5, 3.5]
        ax.plot(lim, lim, color=MUTED, lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_title(f"{lane} · {name}")
        ax.set_xlabel("실제 y (정규화 gold+xp 격차)")
        if lane == "TOP":
            ax.set_ylabel("예측 TCI")
        b = fbest.loc[(lane, full)]
        ax.text(-3.3, 3.0, f"RMSE {b.RMSE:.3f}\nR² {b.R2:.3f}", fontsize=8, va="top")
        metrics["lanes"][lane] = {
            "model": full, **{k: float(b[k]) for k in
                              ["MSE", "RMSE", "MAE", "R2", "RMSE_3-10", "RMSE_11-20", "RMSE_21+",
                               "AUC_win@15", "RMSE_std"]},
            "baseline_RMSE": float(fbest.loc[(lane, "Mean(baseline) [time-norm]"), "RMSE"]),
            "features": feats,
        }
    save(fig, out, "fig4_pred_vs_true.png")
    pd.DataFrame(coef_rows).to_csv(os.path.join(out, "final_coefficients.csv"), index=False)

    coef = pd.DataFrame(coef_rows)
    if len(coef):
        fig, axes = plt.subplots(1, 4, figsize=(13, 4.2))
        for ax, lane in zip(axes, LANES):
            s = coef[(coef.lane == lane) & (coef.feature != "minute")].sort_values("coef")
            colors = [COLOR[lane] if v >= 0 else "#8a8984" for v in s.coef]
            ax.barh(s.feature, s.coef, color=colors, height=0.6)
            ax.axvline(0, color=INK, lw=0.6)
            ax.set_title(lane)
            ax.set_xlabel("표준화 계수 w (TCI 기여)")
            ax.grid(axis="y", visible=False)
        fig.suptitle("최종 Lasso 모델의 표준화 계수 (0 = Lasso가 제거한 피처)", fontweight="bold")
        save(fig, out, "fig5_coefficients.png")

    # 5) 활용 예시: 한 경기의 라인별 TCI 곡선 --------------------------------
    ex_id = md.match_id.unique()[0]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for lane in LANES:
        bundle = joblib.load(os.path.join(out, "models", f"TCI_{lane}.joblib"))
        d = md[(md.lane == lane) & (md.match_id == ex_id)]
        tci = bundle["model"].predict(d[bundle["features"]].to_numpy())
        ax.plot(d.minute, tci, color=COLOR[lane], lw=2, label=lane)
    ax.axhline(0, color=MUTED, lw=0.8)
    win = "블루 승" if md[md.match_id == ex_id].blue_win.iloc[0] else "레드 승"
    ax.set_title(f"라인별 TCI 추이 예시 ({ex_id}, {win})")
    ax.set_xlabel("경기 시간 (분)")
    ax.set_ylabel("TCI (블루 우위 > 0)")
    ax.legend(ncol=4, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    save(fig, out, "fig6_tci_example.png")
    metrics["example_match"] = ex_id

    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("[done]", json.dumps({k: v["model"] for k, v in metrics["lanes"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--features", default=None, help="extract_features.py 결과 (minute_features_full.csv.gz)")
    ap.add_argument("--end", default=None, help="extract_features.py 결과 (end_of_game_full.csv.gz)")
    ap.add_argument("--compare-matches", type=int, default=2000)
    ap.add_argument("--reuse-compare", action="store_true", help="out 폴더의 기존 모델 비교 결과 재사용")
    a = ap.parse_args()
    main(a.data, a.out, a.features, a.end, a.compare_matches, a.reuse_compare)
