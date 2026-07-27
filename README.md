# Wind-Bidding HistGBR-Only Experimental Pipeline

This repository contains a HistGradientBoostingRegressor-only variant of the full experimental pipeline. All five point-forecast targets use the same model family. Smoke tests, diagnostics, failed branches, result-verification programs, and table/figure producers are excluded.


## Process order

```text
01 Fixed raw inputs
   ↓
02 Input preprocessing
   ↓
03 Point forecasting
   ↓
04 Forecast-error construction
   ↓
05 Gaussian/Laplace error-increment fitting
   ↓
06 S=5000 scenario generation
   ↓
07 M0/M1/M2 full-year optimization
```

The output of each stage is the direct input of the next stage.

## Code by stage

### 01. Fixed raw inputs

`01_입력데이터/`

External paper inputs are placed here. The raw data are distributed separately because the RT energy file alone exceeds GitHub's 100 MB per-file limit.

### 02. Input preprocessing

`02_입력데이터가공/코드/`

Execution order:

1. `01_풍력발전_가공데이터생성.py`
2. `02_하루전가격_가공데이터생성.py`
3. `03_실시간가격_가공데이터생성.py`
4. `04_DA_REG가격_가공데이터생성.py`
5. `05_RT_REG가격_가공데이터생성.py`

`90_입력데이터가공_공통유틸.py` is the shared feature/time-split implementation. Outputs are written to `02_입력데이터가공/가공데이터/`.

### 03. Point forecasting

`03_점예측/코드/`

For each target, the model is trained first and the paper-format predictions are exported second:

| Target | Train | Export | Selected downstream model |
|---|---|---|---|
| Wind | `08` | `11` | HistGradientBoostingRegressor |
| DA energy | `09` | `12` | HistGradientBoostingRegressor |
| RT energy | `10` | `13` | HistGradientBoostingRegressor |
| DA REG | `14` | `16` | HistGradientBoostingRegressor |
| RT REG | `15` | `17` | HistGradientBoostingRegressor |

Only HistGradientBoostingRegressor is fitted for all five targets. `90_점예측_공통유틸.py` contains the common features, time split, model parameters, and output rules. Outputs are written to `03_점예측/점예측결과/`.

### 04. Forecast errors

`04_오차데이터/코드/01_점예측기반_오차데이터생성.py`

Combines the selected predictions with realized values and creates the five error time series under `04_오차데이터/오차데이터결과/`.

### 05. Error-increment distributions

`05_오차증분_분포적합/코드/01_오차증분기반_분포적합생성.py`

Fits Gaussian and Laplace error-increment models for all five series. Outputs are written to `05_오차증분_분포적합/오차증분_분포적합결과/`.

### 06. Scenario generation

`06_시나리오생성/코드/01_점예측오차증분기반_시나리오생성.py`

Generates Gaussian and Laplace scenarios independently using:

- 5,000 scenarios;
- random seed 777;
- `variable_pc12_morton` matching;
- 2021 evaluation dates shared by all five point-forecast series.

The two families are run separately so each starts from seed 777. Outputs are written below `06_시나리오생성/reproduced/`.

### 07. Full-year optimization

`07_리스크관리_입찰곡선최적화/코드/`

Execution dependency:

```text
01_입력정산_공통유틸.py
→ 02_입찰곡선_기초연산.py
→ 03_M0_M1_M2_최적화.py
→ 04_연간최적화_실행.py
→ 05_최종6개실험_실행.py
```

The last file runs Laplace then Gaussian, with M0, M1, and M2 for each family. CPLEX uses an MIP relative-gap target of 0.001 where a mixed-integer solve is involved. Final outputs are written to `07_리스크관리_입찰곡선최적화/reproduced/`.

## One-command execution

Create the tested environment:

```powershell
conda env create -f environment.yml
conda activate wind-s5000-opt
```

Run the complete pipeline:

```powershell
python run_pipeline.py
```

A partial range can be executed without rerunning completed upstream stages:

```powershell
python run_pipeline.py --start-stage forecast --end-stage scenarios
python run_pipeline.py --start-stage optimization
```

The available stage names are `preprocess`, `forecast`, `errors`, `fit`, `scenarios`, and `optimization`.

## Environment

The calculation environment is Python 3.11.14 with NumPy 2.4.3, pandas 3.0.2, SciPy 1.17.1, scikit-learn 1.8.0, PyArrow 23.0.1, tqdm 4.67.3, and IBM ILOG CPLEX 22.1.2.0. No GPU acceleration is used. A valid CPLEX installation and license are required.

## Input data

Input data are not bundled with this code-only repository. The required filenames and placement rules are listed in `01_입력데이터/README.md`.

## Scope

The repository stops at the six final optimization outputs. Paper table and figure scripts are intentionally not included.
