"""라인별 TCI 모델 학습·비교.

TCI_LANE(x, t; θ) = f_θ(x_t)  ≈  y_t
  x_t : t분까지의 누적 격차 피처 (골드·경험치 자체는 제외 → 누출 방지)
  y_t : 0.5 * ( Δgold_t / RMS_gold(t) + Δxp_t / RMS_xp(t) )   (라인별, 분별 RMS 정규화)

검증: 경기(match_id) 단위 GroupKFold(5) × 3회 반복(셔플 시드 상이)
      → 같은 경기의 다른 분 샘플이 학습/검증에 동시에 들어가는 누출 차단
      RMS 정규화 상수도 각 fold 의 학습 데이터에서만 추정
지표: MSE, RMSE, MAE, R² (정규화 단위), 시간 구간별 RMSE, TCI@15분의 승패 AUC
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

TARGET_COLS = ("gold_gap", "xp_gap")

_COMMON = ["minute", "cs_gap", "dmg_gap", "kills_gap", "deaths_gap", "assists_gap",
           "cc_time_gap", "wards_placed_gap", "wards_killed_gap"]
LANE_FEATURES = {
    "TOP": _COMMON + ["solo_kills_gap", "dmg_taken_gap", "plates_gap", "turrets_gap", "grubs_herald_gap"],
    "JUNGLE": ["minute", "jungle_cs_gap", "cs_gap", "dmg_gap", "kills_gap", "deaths_gap", "assists_gap",
               "cc_time_gap", "wards_placed_gap", "wards_killed_gap",
               "dragons_gap", "grubs_herald_gap", "barons_gap"],
    "MIDDLE": _COMMON + ["solo_kills_gap", "dmg_taken_gap", "plates_gap", "turrets_gap"],
    "BOTTOM": _COMMON + ["dmg_taken_gap", "plates_gap", "turrets_gap", "dragons_gap"],
}


def make_models(seed: int = 0, time_norm: bool = True, exclude=()) -> dict:
    """time_norm=True 이면 모든 모델 앞에 TimeNormalizer 를 붙인다. exclude 에 든 모델은 뺀다."""
    sc = StandardScaler
    base = {
        "Mean(baseline)": DummyRegressor(strategy="mean"),
        "LinearRegression": make_pipeline(sc(), LinearRegression()),
        "Ridge": make_pipeline(sc(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
        "Lasso": make_pipeline(sc(), LassoCV(cv=5, random_state=seed)),
        "KNN(k=15)": make_pipeline(sc(), KNeighborsRegressor(n_neighbors=15, weights="distance")),
        "SVR(RBF)": make_pipeline(sc(), SVR(C=1.0, epsilon=0.1)),
        "RandomForest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                              n_jobs=-1, random_state=seed),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=20,
            l2_regularization=1.0, random_state=seed),
        "MLP(64-32)": make_pipeline(sc(), MLPRegressor(
            hidden_layer_sizes=(64, 32), alpha=1e-2, learning_rate_init=1e-3, max_iter=2000,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=seed)),
    }
    base = {k: v for k, v in base.items() if k not in exclude}
    if not time_norm:
        return base
    return {k: make_pipeline(TimeNormalizer(), v) for k, v in base.items()}


# ---------------------------------------------------------------- 타겟 정규화
class TargetScaler:
    """분(minute)별 RMS 로 gold/xp 격차를 나눠 평균한다. 학습 데이터로만 fit.

    표본이 적은 분의 RMS 가 흔들리지 않도록 t±window 분의 행을 모아 RMS 를 계산하고,
    표본이 min_count 미만인 후반 분은 마지막 유효값을 사용한다.
    """

    def __init__(self, cols=TARGET_COLS, window: int = 2, min_count: int = 30):
        self.cols, self.window, self.min_count = cols, window, min_count

    def fit(self, df: pd.DataFrame):
        self.scale_ = {}
        minutes = np.arange(int(df.minute.min()), int(df.minute.max()) + 1)
        for c in self.cols:
            table, last = {}, None
            for m in minutes:
                x = df.loc[(df.minute - m).abs() <= self.window, c].to_numpy()
                if len(x) >= self.min_count:
                    last = float(np.sqrt(np.mean(np.square(x)))) or 1.0
                if last is not None:
                    table[int(m)] = last
            first = table[min(table)]
            for m in minutes:
                table.setdefault(int(m), first)
            table["_last"] = last
            self.scale_[c] = table
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts = []
        minute = df["minute"].to_numpy().astype(int)
        for c in self.cols:
            tab = self.scale_[c]
            keys = np.array(sorted(k for k in tab if k != "_last"))
            vals = np.array([tab[k] for k in keys])
            pos = np.clip(minute - keys[0], 0, len(keys) - 1)
            scale = np.where(minute > keys[-1], tab["_last"], vals[pos])
            parts.append(df[c].to_numpy() / scale)
        return np.mean(parts, axis=0)


class TimeNormalizer(BaseEstimator, TransformerMixin):
    """피처 행렬의 0번 열(minute)을 기준으로 나머지 열을 '분별 RMS'로 나눈다.

    누적 격차는 시간이 지날수록 분산이 커지므로(예: 5분 CS 10개 차이 ≠ 25분 CS 10개 차이),
    타겟과 같은 방식으로 '그 시점의 전형적인 격차 크기' 단위로 바꿔 준다.
    minute 열은 그대로 둔다. 통계는 fit 에 들어온 학습 데이터에서만 계산한다.
    """

    def __init__(self, window: int = 2, min_count: int = 30):
        self.window, self.min_count = window, min_count

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        minute = X[:, 0]
        self.minutes_ = np.arange(int(minute.min()), int(minute.max()) + 1)
        table = np.full((len(self.minutes_), X.shape[1] - 1), np.nan)
        for i, m in enumerate(self.minutes_):
            rows = np.abs(minute - m) <= self.window
            if rows.sum() >= self.min_count:
                table[i] = np.sqrt(np.mean(np.square(X[rows, 1:]), axis=0))
        table = pd.DataFrame(table).ffill().bfill().to_numpy().copy()
        table[~np.isfinite(table) | (table == 0)] = 1.0
        self.table_ = table
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        idx = np.clip(X[:, 0].astype(int) - self.minutes_[0], 0, len(self.minutes_) - 1)
        X[:, 1:] = X[:, 1:] / self.table_[idx]
        return X


# ---------------------------------------------------------------- 교차검증
def cross_validate_lane(df: pd.DataFrame, lane: str, seeds=(0, 1, 2), n_splits=5,
                        modes=(("raw", False), ("time-norm", True)), exclude=(), only=None):
    """only: ["Lasso [time-norm]"] 처럼 특정 후보만 평가할 때 사용."""
    d = df[df.lane == lane].reset_index(drop=True)
    feats = LANE_FEATURES[lane]
    X, groups = d[feats].to_numpy(), d["match_id"].to_numpy()
    results, oof_store = [], {}
    for seed in seeds:
        gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        models = {f"{name} [{tag}]": m for tag, flag in modes
                  for name, m in make_models(seed, flag, exclude).items()}
        if only:
            models = {k: v for k, v in models.items() if k in only}
        oof = {name: np.zeros(len(d)) for name in models}
        y_true = np.zeros(len(d))
        for tr, te in gkf.split(X, groups=groups):
            ts = TargetScaler().fit(d.iloc[tr])
            y_tr, y_te = ts.transform(d.iloc[tr]), ts.transform(d.iloc[te])
            y_true[te] = y_te
            for name, model in models.items():
                m = clone(model).fit(X[tr], y_tr)
                oof[name][te] = m.predict(X[te])
        for name, pred in oof.items():
            mse = mean_squared_error(y_true, pred)
            row = {"lane": lane, "seed": seed, "model": name, "MSE": mse, "RMSE": np.sqrt(mse),
                   "MAE": mean_absolute_error(y_true, pred), "R2": r2_score(y_true, pred)}
            for lo, hi, tag in [(3, 10, "RMSE_3-10"), (11, 20, "RMSE_11-20"), (21, 999, "RMSE_21+")]:
                mask = (d.minute >= lo) & (d.minute <= hi)
                row[tag] = np.sqrt(mean_squared_error(y_true[mask], pred[mask]))
            m15 = (d.minute == 15).to_numpy()
            row["AUC_win@15"] = roc_auc_score(d.blue_win[m15], pred[m15])
            results.append(row)
        oof_store[seed] = (y_true, oof)
    return pd.DataFrame(results), d, oof_store


def summarize(cv: pd.DataFrame) -> pd.DataFrame:
    cols = ["MSE", "RMSE", "MAE", "R2", "RMSE_3-10", "RMSE_11-20", "RMSE_21+", "AUC_win@15"]
    agg = cv.groupby(["lane", "model"])[cols].mean()
    agg["RMSE_std"] = cv.groupby(["lane", "model"])["RMSE"].std()
    return agg.reset_index()


def fit_final(df: pd.DataFrame, lane: str, model_name: str = "Ridge", time_norm: bool = True):
    d = df[df.lane == lane].reset_index(drop=True)
    ts = TargetScaler().fit(d)
    y = ts.transform(d)
    model = clone(make_models(0, time_norm)[model_name]).fit(d[LANE_FEATURES[lane]].to_numpy(), y)
    return model, ts, d, y
