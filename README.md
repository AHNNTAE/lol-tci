# 라인별 TCI(Timeline Contribution Index) 모델

리그 오브 레전드 매치/타임라인 데이터로 TOP · JUNGLE · MIDDLE · BOTTOM 4개 라인의
분 단위 경기력 지표 `TCI_LANE(x, t; θ)`를 지도학습으로 만드는 코드입니다.
보고서는 [`REPORT.md`](REPORT.md)에 있습니다.

## 구조

```text
lol_tci_lane/
├── extract_features.py        # 대용량 원본 → 분 단위 피처 파일 (병렬·이어하기)
├── run_all.py                 # 전체 파이프라인 (데이터 → 타겟 평가 → 모델 비교 → 그림)
├── extract_players.py         # 경기별 소환사(puuid)·라인·팀·승패 추출
├── analyze_players.py         # 챌린저 소환사별 TCI 평가 (보고서 4.2절)
├── tci_lane/
│   ├── data_process.py        # data_process_sample.py 변형: 분 단위 누적 격차 피처 추출
│   ├── evaluate_target.py     # evaluate_y.py 확장: 타겟 후보의 승패 예측력 비교
│   └── train.py               # 라인별 피처, 타겟 정규화, TimeNormalizer, 모델 9종 × 피처 2종 교차검증
├── results/                   # 30경기 예비 실험 결과
├── results_full/              # 38,075경기 전체 결과 (보고서 기준, metrics.json, figures/, models/)
├── results_players/           # 소환사별 TCI 요약(익명)과 그림
├── REPORT.md                  # 양식에 맞춘 보고서
└── match_data/, data/         # (git 제외) match_*.json, timeline_*.json
```

## 실행

### 소규모 (수십~수백 경기)

```bash
pip install -r requirements.txt
python run_all.py --data data --out results     # data/ 안의 match_*.json, timeline_*.json
```

### 전체 데이터 (수천 개 파일, 수십 GB)

원본을 한 번에 읽지 않도록 두 단계로 나눕니다.

```bash
# 1) 파일 쌍(match_N, timeline_N)을 하나씩 읽어 분 단위 피처만 압축 저장 (중간에 끊겨도 다시 실행하면 이어서 처리)
python extract_features.py --data match_data --workers 4
#    → results/minute_features_full.csv.gz, results/end_of_game_full.csv.gz

# 2) 모델 비교(무작위 2,000경기) → 최종 모델을 전체 경기로 교차검증·학습
python run_all.py --features results/minute_features_full.csv.gz --end results/end_of_game_full.csv.gz --out results_full
# (모델 비교를 건너뛰고 최종 단계만 다시 돌릴 때: --reuse-compare)

# 3) 소환사별 TCI 평가 (puuid_challengers.txt 필요)
python extract_players.py --data match_data --workers 2
python analyze_players.py
```

`--workers`는 동시에 처리할 파일 수입니다. 타임라인 파일 하나를 읽는 데 프로세스당 1GB 안팎의 메모리를
쓰므로, RAM이 8GB이면 2, 16GB 이상이면 4를 권장합니다. 대규모 모드에서는 SVR(RBF)을 비교에서 제외합니다
(표본 수의 제곱에 비례해 느려짐).

## 저장된 모델 사용

```python
import joblib, pandas as pd
b = joblib.load("results_full/models/TCI_TOP.joblib")      # {"model", "target_scaler", "features"}
rows = pd.read_csv("results/minute_features_full.csv.gz").query("lane == 'TOP'")
tci = b["model"].predict(rows[b["features"]].to_numpy())  # 블루 기준, >0 이면 블루 TOP 우위
```

TCI 값의 단위는 "그 시점 해당 라인의 전형적인(RMS) 골드·경험치 격차"입니다. 1.0이면 그 분에
보통 벌어지는 격차만큼 앞서고 있다는 뜻입니다. 레드 측 선수의 TCI는 부호를 바꿔서 읽습니다.
