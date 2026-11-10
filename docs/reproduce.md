# 저장소 단독 실행 안내

명령은 real_market_rl_trader 루트에서 실행합니다. 연구 실행 계약의
`./market.sh python ...`은 로컬 작업 공간용이며 단독 clone에서는 가상환경을
활성화한 뒤 `python ...`으로 실행합니다. `./market.sh ablation ...`은
`python experiments/run_ablation.py ...`에 해당합니다.

## 설치

```bash
git clone https://github.com/An-Choi/real_market_rl_trader.git
cd real_market_rl_trader
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-research.lock.txt
python -m pytest -q env/tests agent/tests serving/tests
```

## 데이터와 새 실험

기존 data/가 없을 때만 다음 clone을 실행합니다.

```bash
git clone https://github.com/An-Choi/rl_trader_data.git data
python scripts/research_data_audit.py --help
python scripts/prepare_extended_study.py --output runs/new-study/plan.json
python experiments/run_ablation.py --help
```

학습 명령과 환경 규칙은 [research-contract.md](research-contract.md)에 있습니다.
기존 모델 재생에는 데이터 fingerprint 일치가 필요합니다. 데이터 저장소의 변경 이력과
로컬 정리 기록에 따라 파일 구성이 다를 수 있습니다. 자료가 부족하면 데이터 기준을
낮추지 말고 누락 구간을 확인하세요. 새 실험에는 새 출력 경로를 사용합니다.

비교 그래프·집계표는 docs/research-assets/에 있습니다. 전체 모델·trace는 Git에 포함되지 않습니다.
`scripts/verify_completion.py`는 원래 작업 공간의 저장된 증거를 검사하는 도구입니다.
fresh clone 검증에는 위 pytest 명령을 사용하세요.

## 연구 모델 재생

모델·정규화·manifest와 일치하는 데이터가 준비된 경우:

```bash
python serving/src/research_app.py --bundle runs/STUDY/fold_03/I/seed_42 --symbol 005930
```

Java 저장소에서 다음을 실행합니다. 출력은 새 경로여야 합니다.

```bash
./gradlew researchDryRun --args='--url http://127.0.0.1:8766 --output /tmp/NEW-replay'
```

서버는 loopback에서만 동작하며 실제 주문 API를 사용하지 않습니다.

## Git 파일 구분

코드·테스트·설정·문서와 소규모 공유용 그래프/집계표만 커밋합니다.
.env, 토큰, 개인키, data/, runs/, artifacts/, 가상환경, 로그, 로컬 압축본은 제외합니다.
이미 추적 중인 데이터는 .gitignore를 추가해도 추적이 해제되지 않습니다.
