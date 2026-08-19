# RL 시장 트레이더 1~7단계 개선 보고서

작성일: 2026-08-19
대상: KRX 5종목 (`005930`, `000660`, `034220`, `066570`, `009150`)

## 1. 결론 요약

데이터 오염, 종목별 날짜 불일치, 비현실적인 주문 단위, feature 사용 여부를 확인할
수 없는 문제를 수정했다. 이후 전체 11-feature와 신호 기반 축소 8-feature를 동일한
3-fold 조건으로 재학습했다.

축소형의 3-fold 평균 수익률은 `-0.0475%`로 전체형 `-0.1793%`보다 손실이 작았다.
그러나 축소형의 Hold 비율은 `99.927%`이고 validation을 통과한 폴드는 `0/3`이다.
상승장 두 폴드에서도 수익률이 `0%`였으므로, 이는 예측력 향상이 아니라 현금 보유로
손실을 회피한 결과다. 두 모델 모두 승격하지 않는다.

## 2. 단계별 작업

### 1단계 — 원천 데이터 복구

- 4종목에서 중복된 분봉 9,144행을 발견했다.
- 같은 시각의 중복 스냅샷은 `TradingValue`, `Volume`이 큰 행을 우선해 한 행만 남겼다.
- 잘못된 월 파티션에 들어간 행을 실제 timestamp 월로 재배치했다.
- 수정 전 24개 월 파일을 `data/quality_backups/20260818-152529`에 백업했다.
- 수정 후 재감사 결과 중복 timestamp는 0건이다.
- 로더와 저장기 모두 중복 및 잘못된 월 파티션을 다시 차단하도록 변경했다.

과거 중복은 거래량을 두 번 더하고 누적 거래대금 차분을 0으로 만들어 `vwap_dev`를
약 1.0까지 왜곡했다. 따라서 오염된 데이터로 학습한 이전 결과는 폐기 대상이다.

### 2단계 — 3-fold 날짜 구조 수정

기존 코드는 종목별 거래일의 합집합으로 fold 경계를 만들었다. 그 결과 같은 fold의
test가 종목마다 10~20일로 달랐다. 현재는 5종목에 모두 존재하는 공통 거래일만 사용한다.

- 종목별 feature 거래일: 223~269일
- 5종목 공통 거래일: 196일
- 공통 기간: 2025-07-29 ~ 2026-08-18
- 각 test 구간: 정확히 20거래일
- test 장세: Fold 1 상승, Fold 2 상승, Fold 3 하락

Fold 1 validation은 과거 데이터상 상승장만 포함한다. 미래 하락장을 끌어와 섞으면
누수가 되므로 경계를 인위적으로 바꾸지 않았다. 정식 승격 조건은 여전히 장세 2개
이상이며, 이번 screening에서만 실행을 위해 최소 조건을 1로 낮췄다.

### 3단계 — 거래 환경 현실화

- 초기자금: 1만원 → 1억원
- 소수점 주식 매수 → 정수 주식 매수
- 액션: `Hold`, `Add`, `Reduce 1 Unit`, `Clear`의 4개
- `Adv20`과 주문금액의 비율을 liquidity score로 변환
- 기본 슬리피지 0 → 0.02%, 유동성이 낮으면 비용 증가
- 학습과 serving이 동일한 유동성 계산식을 사용
- 포지션 lot별 주식 수와 원가를 추적

이 변경으로 기존 3-action artifact는 호환되지 않는다. action label 검증이 기존 모델의
로딩을 명시적으로 거부하므로 새 환경에서는 반드시 재학습해야 한다.

### 4단계 — RL 전 신호 검증

각 feature와 동일 거래일의 1·3·12봉 선행수익률 사이 Spearman IC를 계산했다. 미래
수익률은 거래일 경계를 넘지 않게 만들고, validation 3폴드 × 5종목의 방향 안정성을
측정했다.

주요 결과:

| Feature | Horizon | Validation 평균 IC | 방향 일관성 | Test 평균 IC | 해석 |
|---|---:|---:|---:|---:|---|
| `log_ret_12` | 12 | -0.0835 | 100.0% | -0.0562 | 가장 안정적인 평균회귀 |
| `vwap_dev` | 12 | -0.0612 | 80.0% | -0.0657 | VWAP 이격 평균회귀 |
| `log_ret_1` | 1 | -0.0377 | 93.3% | -0.0468 | 단기 평균회귀 |
| `vol_regime_30m` | 12 | -0.0433 | 86.7% | -0.0006 | test에서 소멸 |
| `macd_hist_30m` | 12 | 0.0289 | 73.3% | 0.0129 | 약하고 불안정 |

### 5단계 — 신호 기반 feature 축소

전체 11개와 아래 8개 축소형을 비교하도록 구성했다.

유지:

- `log_ret_1`, `log_ret_12`
- `realized_vol_12`, `vwap_dev`
- `trend_strength_30m`, `vol_regime_30m`
- `gap_open`, `relative_volume_tod`

제외:

- `relative_volume`: 시간대 보정형과 중복되고 정책 민감도가 낮음
- `macd_hist_30m`: trend와 중복되며 test 안정성이 낮음
- `log_ret_3`: `log_ret_1`과 중복되고 별도 기여가 약함

### 6단계 — 동일 조건 3-fold 재학습 비교

두 실험 모두 CPU, fold별 seed 42~44, 요청 50,000 step, validation-best checkpoint,
동일 거래비용과 동일 test 구간을 사용했다. 이는 구조 비교용 screening이며 최종
300,000-step 모델이 아니다.

#### 모델 결과

| 모델 | Fold 1 상승 | Fold 2 상승 | Fold 3 하락 | 3-fold 평균 | 평균 MDD | Hold | 통과 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 전체 11-feature | -0.1924% | -0.3454% | 0.0000% | -0.1793% | -0.2392% | 99.666% | 0/3 |
| 축소 8-feature | 0.0000% | 0.0000% | -0.1426% | -0.0475% | -0.0576% | 99.927% | 0/3 |

#### Baseline 결과

| Baseline | Fold 1 상승 | Fold 2 상승 | Fold 3 하락 | 3-fold 평균 |
|---|---:|---:|---:|---:|
| Buy & Hold | 18.85% | 96.03% | -27.12% | 29.25% |
| Random | -23.71% | -19.31% | -24.15% | -22.39% |
| MA crossover | -5.84% | 8.74% | -25.92% | -7.67% |

전체형은 평균 거래회전율 0.849, 종목당 fold 평균 4.27회 거래했다. 축소형은 각각
0.182와 0.93회로 더 적게 거래했다. 축소형의 낮은 손실은 신호 향상보다 거래 중단의
영향이 크다.

## 3. 현재 문제의 정확한 정의

1. feature에 약한 평균회귀 IC는 있지만 실제 거래비용을 안정적으로 넘을 정도로 강하지 않다.
2. 절대 포트폴리오 수익률을 최적화하면 불확실한 상황에서 현금 100%가 합리적인 해가 된다.
3. validation의 Hold 제한은 모델을 `qualified=false`로 표시하지만, screening artifact 자체는
   남기므로 결과를 볼 때 승격 여부를 별도로 확인해야 한다.
4. 공통 거래일은 196일뿐이고 첫 validation은 상승장 편향이다.
5. feature 제거만으로 cash-policy collapse는 해결되지 않았다.

## 4. 의사결정

- 전체 11-feature: 승격 거부
- 축소 8-feature: 승격 거부
- 300k 정식 재학습: 지금은 보류
- 다음 기본 feature 후보: 축소형을 그대로 채택하지 않고 연구 후보로만 유지
- 추가 기술지표: 현재는 추가하지 않음

다음 실험은 과거 하락·횡보 분봉을 먼저 확보한 뒤 수행한다. 이후 평균회귀 신호를
직접 검증하는 비용 차감 threshold baseline을 만들고, 그 baseline이 양수일 때만 PPO
300k/multi-seed 실험으로 넘어간다. 이렇게 해야 PPO가 단순 현금 정책을 학습한 결과와
실제 alpha를 구분할 수 있다.

## 5. 산출물

- 데이터 감사: `runs/data_quality/minute-duplicate-20260818-152529.json`
- 신호 리포트: `runs/signal_reports/signal-20260818-153000.json`
- 전체형: `runs/walk_forward/stage6-full-50k-20260819/summary.json`
- 축소형: `runs/walk_forward/stage6-reduced8-50k-20260819/summary.json`
- 비교 JSON: `runs/comparisons/stage6-feature-comparison-20260819.json`
- 중복 복구 도구: `scripts/audit_repair_minute_data.py`
- 비교 집계 도구: `experiments/compare_walk_forward_runs.py`

## 6. 검증

- 데이터·수집·환경·학습 핵심 테스트: 121개 통과
- serving observation/predictor 테스트: 17개 통과
- 4-action 지표 회귀 테스트: 27개 통과
- 분봉 재감사: 중복 0건

FastAPI가 학습 Conda 환경에 설치되어 있지 않아 FastAPI 통합 테스트는 그 환경에서
실행하지 못했다. 관측 생성 및 predictor artifact 계약 테스트는 통과했다.

## 7. 추가 전체 재점검 결과

### 7.1 현재 가장 큰 문제: 폴드별 데이터 기간과 품질이 서로 다름

현재 5개 종목이 모두 존재하는 공통 거래일은 196일이다. 각 폴드의 테스트 구간은
20개 공통 거래일로 구성되어 있지만, 실제 달력상 길이와 일봉 기준 거래일 수가 크게
다르다.

| 폴드 | 테스트 기간 | 달력 일수 | 일봉 거래일 | 공통 거래일 | 일봉 대비 coverage |
|---|---:|---:|---:|---:|---:|
| Fold 1 | 2026-03-13 ~ 2026-04-14 | 33일 | 23일 | 20일 | 87% |
| Fold 2 | 2026-04-15 ~ 2026-06-17 | 64일 | 42일 | 20일 | 48% |
| Fold 3 | 2026-06-22 ~ 2026-08-18 | 58일 | 40일 | 20일 | 50% |

따라서 현재의 “20 거래일 폴드”는 서로 같은 길이와 품질의 검증 구간이 아니다.
특히 Fold 2의 buy-and-hold 약 +96%는 64일의 달력 기간에 걸쳐 발생한 수치이므로,
다른 폴드와 단순 비교하면 왜곡될 수 있다.

종목별 원본 데이터의 무효·누락일도 확인됐다. 005930은 14일, 000660은 22일,
034220은 27일, 066570은 28일, 009150은 37일이며 최근 구간의 공통 coverage가
특히 낮다. 005930·000660·034220 세 종목만 사용하면 공통 feature 거래일은
225일로, 5개 종목의 196일보다 늘어난다.

다음 조치가 필요하다.

1. KRX 공식 거래일 달력을 기준으로 expected bar 수를 계산한다.
2. 종목·날짜별 coverage 95% 이상, 최대 연속 누락일, 실제 달력 기간을 검사한다.
3. 기준 미달 날짜는 재수집하고, 해결되지 않으면 해당 종목 또는 폴드에서 제외한다.
4. 단기적으로는 coverage가 좋은 3개 종목으로 실험하고, 데이터 복구 후 5개로 확장한다.
5. 상승장 편향을 줄이기 위해 2022~2024년 5분봉 또는 15분봉을 추가 확보해
   하락장·횡보장 검증 구간을 넓힌다.

### 7.2 학습·백테스트·실전 추론의 동작 불일치

현재 동일한 모델이라도 학습, 백테스트, serving에서 관측값과 가능한 행동이 완전히
같다고 보장되지 않는다.

- 학습 환경은 실제 `cost_basis`를 portfolio state에 전달하지만 serving 관측 생성은
  이를 받지 않아 초기 자본과 unit 비율로 추정한다. 정수 주식 수와 실제 체결가를 쓰는
  현재 구조에서는 서로 다른 값이 된다.
- validation은 환경의 정확한 action mask를 사용하지만 일반 backtest 경로는 보유 unit
  중심의 fallback mask를 사용할 수 있어 현금·유동성 제약이 다르게 적용될 수 있다.
- 학습과 백테스트는 축소된 8개 feature를 지원하지만 serving predictor는 전체 feature
  목록과 artifact가 정확히 같아야 한다고 검사해 축소 feature 모델을 배포할 수 없다.
- 매수 가능 수량이 0주인 경우에도 unit 목표가 증가하고 0주 lot이 추가될 가능성이 있다.
- 행동 체계가 4-action으로 변경됐지만 artifact format은 계속 v4라 계약 변경이 명확하지 않다.

Stage 6 결과가 대부분 현금 보유였기 때문에 이번 결과에 미친 영향은 제한적일 수 있으나,
실제 배포 전에는 반드시 수정해야 한다. 강제로 Add → Reduce → Clear를 수행하는 parity
테스트를 추가해 observation, cost basis, action mask, 체결 결과가 세 경로에서 완전히
동일한지 검증해야 한다. artifact 계약은 v5로 올리는 편이 안전하다.

### 7.3 qualification에 실패한 체크포인트도 최종 모델이 될 수 있음

현재 validation score가 기존 best보다 높으면 `qualified=false`여도 best parameter와
artifact 후보가 될 수 있다. `maximum_hold_action_rate` 같은 안전 기준을 위반한 모델이
최종 배포 모델이 될 위험이 있다.

모델 승격은 다음처럼 fail-closed로 바꿔야 한다.

1. 모든 필수 qualification을 통과한 모델만 best 후보로 인정한다.
2. 통과 모델이 하나도 없으면 학습 실패로 처리한다.
3. 실패 모델은 `rejected/diagnostics`에만 보관하고 serving registry가 읽지 못하게 한다.
4. cash baseline, 비용 차감 순수익, 상승장 capture, 최대 hold 비율을 명시적 승격 조건으로 둔다.

### 7.4 현금 보유 문제는 reward 계수만으로 해결되지 않음

현재 benchmark-relative reward는 같은 시장 경로에서 행동과 무관한 benchmark return을
빼는 구조다. 결과적으로 기본 수익률의 크기를 조정하고 상수를 더하는 효과가 커서,
정책이 현금만 보유하는 문제를 직접 해결하지 못한다.

거래 횟수를 강제로 늘리거나 reward로 억지 매수를 유도하면 수수료 손실만 증가할 수 있다.
롱온리 절대수익 목표라면 하락장에서 현금 보유는 올바른 행동이다. 상승장에서도 계속
현금이라면 진입 신호가 비용을 이기지 못하거나 학습 신호가 부족한 것이 핵심 문제다.

하락장에서도 수익을 내는 것이 목표라면 별도의 inverse ETF 또는 short action 설계가
필요하다. 그렇지 않다면 평가 목표를 상승장 capture와 하락장 방어로 나눠야 한다.
또한 학습 episode 종료 시 청산 비용을 reward에 반영할지, 계속 보유 상태로 볼지를
백테스트와 동일하게 통일해야 한다.

### 7.5 검증 프로토콜 개선

현재 테스트 기간은 signal 분석과 A/B 비교에 반복 사용되어 더 이상 완전히 독립적인
최종 holdout으로 보기 어렵다. 또한 fold마다 seed 하나만 배정되어 시장 구간 차이와
seed 변동성을 분리할 수 없다.

권장 구조는 다음과 같다.

- 기존 3개 폴드는 연구·개발용 validation fold로 취급한다.
- 각 폴드에서 최소 3개 seed를 실행한다.
- 내부 validation과 최종 holdout을 분리하는 nested walk-forward를 사용한다.
- 최종 미래 holdout 또는 paper trading 구간은 모든 설정을 확정한 뒤 한 번만 평가한다.
- 50k step은 빠른 선별용으로 사용하고, 통과한 설정만 100k와 300k로 확장한다.
- cash baseline, net alpha, bull capture, bear protection, exposure, turnover, MDD를 함께 기록한다.

### 7.6 feature와 신호의 다음 발전 방향

RSI, Bollinger Band, MACD처럼 기존 가격·추세 feature와 중복성이 큰 지표를 바로 늘리는
것은 우선순위가 낮다. 먼저 현재 유효 후보인 `log_ret_12`, `vwap_dev`, `log_ret_1`로
비용 인식형 mean-reversion 기준선을 만들어야 한다.

기준선은 train 구간에서만 임계값을 정하고, 예상 edge가 왕복 거래비용보다 클 때만
진입하며 변동성에 따라 포지션 크기를 조절한다. 이 단순 전략이 validation에서 비용
차감 후 수익을 내지 못하면 PPO가 학습할 안정적인 신호도 부족할 가능성이 높다.

signal report에는 다음 항목을 추가한다.

- 일별 IC와 block/bootstrap 신뢰구간
- 종목별·regime별 방향 안정성
- turnover 및 거래비용 차감 결과
- 겹치지 않는 예측 horizon 기준 검정
- train에서 정한 방향과 임계값을 validation에 그대로 적용한 결과

그다음 추가할 가치가 높은 feature는 다음과 같다.

- 개별 종목 수익률 - KOSPI/KOSDAQ 수익률
- 개별 종목 수익률 - 업종 ETF 수익률
- 시장 변동성 및 breadth
- rolling beta와 residual return
- `vwap_dev`, `log_ret_12`의 rolling z-score 또는 percentile
- 예상 왕복 비용, 주문 크기/ADV
- 결측·장중 gap 표시 feature

현재 shared policy에는 시장 전체 문맥과 종목 ID가 없으므로 개별 종목 움직임과 시장
공통 움직임을 구분하기 어렵다. 우선 market-relative feature를 추가하고, 데이터가 충분히
늘어난 후에만 선택적으로 symbol embedding을 검토한다.

### 7.7 행동 공간과 모델 구조

현재 Add/Reduce/Clear 방식은 같은 목표 비중까지 여러 행동을 반복해야 하며 이전 행동
경로에 의존한다. 기준선과 데이터 검증이 통과한 뒤에는 `Discrete(6)`으로 0~5 unit의
목표 포지션을 직접 지정하는 target-position action을 비교할 가치가 있다.

대안으로 supervised entry/edge estimator가 진입 가능성을 계산하고 RL이 포지션 크기와
청산을 담당하는 hybrid 구조, 또는 PPO에 미래 수익률 예측 auxiliary head를 추가하는
방식이 있다. 다만 Transformer나 RNN 확대는 데이터 coverage와 비용 차감 신호가 먼저
확인된 후 진행해야 한다.

## 8. 권장 실행 순서

### Phase 7A — 정확성 확보

1. 공식 거래일 기준 calendar/coverage 검사와 누락 데이터 복구
2. 학습·백테스트·serving의 observation, cost basis, action mask parity 수정
3. artifact v5 및 rejected model registry 도입

### Phase 7B — 신호와 검증 개선

4. 비용 인식형 mean-reversion 기준선 구현
5. 일별·regime별·종목별 신호 통계와 비용 검정 강화
6. 독립된 미래 holdout 또는 paper trading 구간 고정

### Phase 7C — 모델 확장

7. coverage가 좋은 3개 종목에서 3 folds × 3 seeds × 100k steps 실행
8. 통과할 때만 전체 5개 종목과 300k steps로 확장
9. target-position action 또는 supervised+RL hybrid A/B 비교

## 9. 다음 단계 통과 기준

- 데이터 coverage 95% 이상이며 승인되지 않은 장기 gap이 없을 것
- 학습·백테스트·serving의 관측값과 action mask가 정확히 일치할 것
- 모든 validation fold가 qualification을 통과할 것
- 거래비용 차감 후 validation 수익이 cash baseline보다 높을 것
- 상승장에서 의미 있는 capture를 보이고 하락장에서는 방어할 것
- seed가 달라도 결과 방향이 안정적일 것
- 설정 확정 전에는 잠근 최종 holdout을 열지 않을 것

현재 최우선 작업은 PPO step을 늘리는 것이 아니라 **calendar/coverage 교정 → parity 오류
수정 → 비용 인식형 단순 기준선 검증**이다. 이 세 단계 없이 학습량만 늘리면 현금 보유
정책을 더 강하게 학습할 가능성이 높다.

## 10. Stage 7 실제 적용 결과

이 절은 7장에서 제안한 개선안 가운데 이번 작업에서 실제 코드에 반영하고 검증한 내용을
정리한다. 사용자 요청에 따라 Stage 7 PPO 재학습은 실행하지 않았다. 따라서 아래의 새
수익률 결과는 RL 재학습 결과가 아니라 비용 인식형 규칙 기준선 결과이며, 기존 Stage 6
RL 결과와 구분해야 한다.

### 10.1 데이터 품질과 coverage

기존 판정은 모든 거래일이 09:00에 정확히 시작하고 분봉이 거의 완전해야 한다고 가정했다.
이 때문에 합법적인 10:00 지연 개장일과 2~3분 수준의 작은 무거래·누락도 하루 전체가
버려졌고, 5종목 공통 학습 기간이 과도하게 짧아졌다.

적용한 변경:

- 거래일 품질을 coverage, 최대 연속 누락, 중복, 판정 사유로 구조화했다.
- coverage 95% 이상이고 최대 연속 누락이 3분 이하인 날은 보존한다.
- 누락 bar를 임의로 채우지는 않으며, 큰 장중 gap은 계속 제외한다.
- 정확히 10:00부터 연속적으로 열린 지연 개장 세션을 보존한다.
- 수집기 감사와 feature 생성기가 같은 품질 판정을 사용한다.
- 원천 parquet의 SHA-256 manifest를 저장하여 raw 데이터가 바뀌면 feature cache를
  자동 무효화한다.

실데이터 재생성 결과:

| 항목 | 기존 | 변경 후 |
|---|---:|---:|
| 5종목 공통 feature 거래일 | 196일 | 251일 |
| 증가분 | - | +55일, 약 +28% |
| 공통 유효 raw 거래일 | - | 271일 |
| 보존된 10:00 지연 세션 | 0일 | 2일 |
| 계속 제외한 공통 대형 결손일 | - | 9일 |

한계도 남아 있다. 아직 공식 KRX 거래일·특수개장 달력을 연결하지 않았으므로, 현재는
정확히 10:00에 시작해 연속된 세션을 지연 개장으로 간주한다. PR 이후에는 공식 달력과
대조하는 보강이 필요하다.

### 10.2 학습·백테스트·serving parity 수정

감사에서 발견한 실행 경로 불일치를 다음처럼 수정했다.

- 백테스트가 learned policy에 환경의 정확한 `action_masks()`를 전달한다.
- rule baseline이 금지 행동을 요청하면 `Hold`로 정규화한다.
- 정수 주식 계산 결과가 0주인 `Add`는 unit과 cost basis를 바꾸지 않는다.
- 매수 가능 여부는 목표 비율이 아닌 실제 정수 주문금액으로 판정한다.
- `Reduce`와 `Clear`의 유동성 비용은 실제 매도 주문 크기와 ADV로 계산한다.
- serving 요청에 실제 `cost_basis`를 전달하며, v5 보유 포지션에는 이를 필수로 한다.
- artifact에 저장된 feature subset과 순서 그대로 serving observation을 만든다.
- 강제 `Add → Add → Reduce → Clear` replay에서 observation, mask, cost basis를 비교하는
  회귀 테스트를 추가했다.

종료 시점 가상청산도 수정했다. 이전 구현은 청산 비용은 실행가로 계산하면서 최종 자산은
종가로 평가해 종가와 경매 실행가가 다를 때 손익 일부가 누락될 수 있었다. 새
`estimate_terminal_settlement_adjustment()`는 `보유주식×Close - 보유주식×ExecPrice +
매도 friction`을 계산하며, 마지막 평가금액에서 이를 빼 실제 가상청산 현금과 맞춘다.

### 10.3 validation fail-closed와 artifact v5

기존에는 validation qualification을 통과하지 못한 checkpoint도 score만 높으면 best로
저장될 수 있었다. 이를 다음 정책으로 변경했다.

- 모든 qualification을 통과한 checkpoint만 배포 후보 best로 선택한다.
- 통과 모델이 없으면 가장 좋은 후보는 진단용으로만 남기고 배포 후보로 승격하지 않는다.
- artifact 계약을 v5로 올리고 `approved`, `rejected`, `research` 상태를 기록한다.
- 저장 경로도 상태별로 분리한다.
- serving은 v5 artifact가 `approved`, validation qualified, `trained_split=train`일 때만
  로드한다.
- 모델 파일 로드 후 observation/action space가 metadata와 맞는지 다시 검사한다.

즉, 성능 기준을 통과하지 못한 모델이 단순히 파일이 존재한다는 이유로 실전 추론에
사용되는 경로를 차단했다.

### 10.4 walk-forward와 seed 설계

종목별 데이터를 공통 날짜 행으로 강제 재색인하지 않고, 공통 reference calendar로 fold
경계만 고정하도록 변경했다. 각 fold·구간·종목별 coverage와 최대 연속 누락을 기록하고
최소 coverage보다 낮으면 fail-closed한다.

또한 기존의 `fold 1=seed 42`, `fold 2=seed 43`, `fold 3=seed 44` 구조는 시장 구간 효과와
seed 효과가 섞였다. 새 실행기는 각 fold에서 동일한 독립 seed 목록을 반복하도록 하여
`3 folds × 3 seeds` 구조를 지원한다. 기존 test 구간은 이미 반복 분석했기 때문에 결과에
`research_reused_not_pristine` 표시를 남긴다.

### 10.5 비용 인식형 mean-reversion 기준선

`log_ret_12`, `vwap_dev` 계열 신호가 거래비용을 실제로 이기는지 PPO보다 먼저 확인하는
기준선을 추가했다.

- 종목별 scale과 진입 threshold는 train 구간에서만 계산한다.
- validation에서 entry quantile 4개와 보유기간 3개만 선택한다.
- test에는 선택된 설정을 고정한다.
- 가격, 날짜, `Adv20`을 이용한 양방향 거래비용을 차감한다.
- 같은 날 겹치는 horizon 거래를 막는다.
- 최소 표본과 거래일 block-bootstrap 신뢰구간 gate를 사용한다.

5종목·3폴드 실제 결과:

| Fold | Validation gate | Test 순수익률 |
|---|---|---:|
| 1 | 실패 | +2.735% |
| 2 | 실패 | -5.186% |
| 3 | 통과 | -5.324% |
| 평균 | 1/3 통과 | -2.591% |

validation에서 선택된 신호가 test에서 반대로 무너졌고, 최악 test는 -5.324%였다. 따라서
현재 mean-reversion feature는 비용 차감 후 안정적인 OOS edge라고 볼 수 없다. 이 결과는
PPO를 100k/300k step으로 확대하지 말아야 할 사전 gate에 해당한다.

### 10.6 signal 진단 확장

`experiments/signal_report.py`에 일별 IC, 거래일 bootstrap 95% 신뢰구간, 겹치지 않는
horizon, train에서 동결한 tail threshold, gross/cost/net 조건부 수익을 추가했다.
기존 12-bar tail을 현재 비용 모델로 단순 점검하면 평균 왕복비용은 약 0.37~0.38%였다.
전체 표본 하위 5% 신호의 gross edge는 이 비용을 이기지 못했다. `vwap_dev` 최극단
0.5%만 평균 net이 약 +0.184%였지만 5종목 합계 312개 표본이고 승률도 47.8%여서,
독립 validation 없이 학습 feature로 승격하기에는 부족하다.

시장 상대 feature는 이번 PR에 억지로 추가하지 않았다. 현재 로컬 데이터에는 5개 개별주만
있고 KOSPI/KOSDAQ·업종 ETF 분봉이 없다. 향후 `069500`, `229200` 및 업종 ETF를 수집하고
serving 입력에도 같은 시장 문맥을 공급한 뒤 residual return, rolling beta, 시장 변동성,
breadth를 schema vNext로 추가해야 한다.

## 11. 검증 현황

수행한 검증:

- 핵심 환경·회계·serving 회귀 테스트 100개 통과
- 강제 포지션 lifecycle 및 v5 요청 parity 테스트 2개 통과
- 비용 인식형 기준선 테스트 5개 통과
- 결손일·중복 데이터 품질 테스트 7개 통과
- 변경 모듈 compile/AST 검사와 `git diff --check` 통과
- 5종목 feature cache와 source manifest 실제 재생성 확인

Windows sandbox의 pytest 임시 폴더 ACL과 일부 환경의 FastAPI 의존성 분리 때문에 전체
테스트 묶음은 아직 한 명령으로 완주하지 못했다. 통과 수를 중복 제거한 전체 테스트 수로
오해해서는 안 된다. PR에서는 CI가 전체 suite를 다시 실행해야 한다.

## 12. 현재 결론과 다음 실행 조건

이번 변경으로 짧아진 데이터 구간, 실행 경로 불일치, qualification 실패 모델의 배포 위험,
거래비용을 무시한 신호 판단은 상당 부분 바로잡았다. 반면 모델 수익성 문제는 해결됐다고
볼 수 없다. 새 기준선조차 3폴드 평균 -2.591%이고 validation gate가 1/3만 통과했다.

사용자 요청에 따라 새 PPO 학습은 실행하지 않았다. 따라서 현재 승인된 신규 모델은 없으며,
기존 Stage 6 결과도 배포 기준을 통과하지 못한 연구 결과로 유지한다.

다음 학습은 아래 조건을 먼저 만족할 때만 진행하는 것이 합리적이다.

1. 공식 KRX calendar로 지연 개장과 결손일 판정을 확정한다.
2. CI 전체 suite를 통과한다.
3. 시장·업종 문맥 데이터를 수집하고 serving 계약까지 함께 갱신한다.
4. 비용 인식형 단순 기준선이 모든 validation fold에서 cash보다 높아야 한다.
5. 그다음에만 50k screening을 3 folds × 3 seeds로 실행한다.
6. seed 방향성과 모든 qualification이 통과할 때만 100k/300k로 확대한다.
7. 설정을 확정한 뒤 아직 열어보지 않은 미래 paper holdout을 한 번 평가한다.

관련 산출물:

- 비용 기준선: `runs/cost_aware_baseline/baseline-20260819-initial.json`
- feature 감사: `docs/feature-audit-2026-08-18.md`
- walk-forward 실행기: `experiments/walk_forward_train.py`
- signal 진단: `experiments/signal_report.py`
- 비용 기준선 실행기: `experiments/cost_aware_baseline.py`
