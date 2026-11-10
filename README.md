# 거래비용·위험 제약 아래의 PPO 트레이딩 연구

국내 주식 5종목에서 목표 투자 비중을 선택하는 강화학습 연구 프로젝트입니다.
기본 PPO·고정 위험 페널티·PPO-Lagrangian을 동일한 조건에서 비교합니다.

## 구현

- 목표 비중 6개 행동: 0%, 20%, 40%, 60%, 80%, 100%
- 과거 12개 봉 관측, 훈련 구간 전용 정규화, 결손 구간의 관측 초기화
- 관측 가격 기반 주문 수량, 정수 주식 수·현금 제약, 비용·미체결·종료 정산
- 최소 비중 조정 폭과 결정 주기 비교, 경과 봉 수를 반영한 GAE
- 고정 위험 페널티와 라그랑주 승수 업데이트, 합성 시장·정책·가치함수 진단

구체적인 규칙은 [연구 실행 계약](docs/research-contract.md), 단독 clone 실행 방법은
[재현 안내](docs/reproduce.md)를 참고하세요.

## 설치와 테스트

검증 환경은 Linux, Python 3.11, PyTorch CPU입니다.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-research.lock.txt
python -m pytest -q env/tests agent/tests serving/tests
```

Git 이력·KIS 키·원본 시세가 없는 별도 소스 폴더에서 Python 테스트 622개를 통과했습니다.
의존성 설치에는 네트워크가 필요합니다. 다른 플랫폼이나 GPU 수치 동일성은 검증하지 않았습니다.

## 비교 산출물

3조건 × 3시드 × 3기간의 27회 학습을 평가했습니다.
[기간·조건별 지표](docs/research-assets/frozen_test_summary.csv)와
[시드별 대응 차이](docs/research-assets/paired_seed_differences.csv)를 제공합니다.

![고정된 역사적 평가 비교](docs/research-assets/frozen_comparison.png)

검증 위험 예산을 모든 종목·시드·기간에서 만족한 조건은 없습니다.
과거 데이터 재사용과 결손일 제외 때문에 일반화 주장은 제한적입니다.
3개 시드는 학습 초기화 변동을 보여주며 독립적인 시장 표본이 아닙니다.
[스트레스 평가](docs/research-assets/stress_comparison.png)는 탐색적 분석이며,
거래 중단 여부의 공식 판정은 완료되지 않았습니다.
[출처 및 해석 범위](docs/research-assets/provenance.json)를 함께 확인하세요.

## 데이터와 모델

원본 시세는 별도 [rl_trader_data](https://github.com/An-Choi/rl_trader_data)에 있습니다.
KIS 키는 추가 시세 수집에 필요합니다. 키·토큰·data/·runs/·모델은 Git에서 제외합니다.
전체 모델과 실험 로그는 로컬 전달본에 보관하며 이 저장소의 clone만으로 제공되지 않습니다.
2026-11-10~2026-12-07 미래 구간은 사전에 예약했으며 수집 후 평가해야 합니다.

## Java 연동

[rl-trading-core](https://github.com/An-Choi/rl-trading-core)의 researchDryRun CLI가
Python 연구 서버와 주문 계획·현금·주식 수·원가를 독립적으로 대조합니다.
실제 주문을 전송하지 않으며 기존 운영 serving 규약과 연구 규약은 별도입니다.

초기 설명은 [기존 구현 기록](docs/legacy-project-overview.md)에 보존했습니다.
현재 연구 규칙은 위 실행 계약을 기준으로 합니다.
