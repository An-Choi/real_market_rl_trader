# serving — predict 서버 + shadow mode

## 서버 기동

```bash
# historical (기본): 백테스트 데이터로만
./.venv/bin/python serving/src/main.py --config serving/configs/serving.yaml

# live: 로컬 백필 + 당일 KIS 조회 병합 — .env에 KIS_APP_KEY/SECRET 필요
# serving.yaml에서 provider: live로 변경 후 동일 명령
```

## Shadow mode (Task 4 검증 — 주문 없음)

전제: 정상 09:00 개장 거래일, `provider: live` + 전용 `audit_log_dir`.

### Preflight (정식 판정 전 필수)

- `git status --porcelain` 출력이 **비어 있어야** 한다 — `current_git_sha()`는 untracked 파일도 dirty로 판정하므로, 로컬 전용 파일(CLAUDE.md 등)은 `.git/info/exclude`에 등록해 둔다.
- 서버·runner·익일 diff 모두 **같은 clean commit**에서 실행 (diff가 HEAD == manifest SHA를 강제한다).

```bash
# 1) 장 시작 전: live 서버 기동
# 2) runner (장중 자동 종료)
./.venv/bin/python serving/src/shadow_runner.py --symbol 005930
# 3) 익일: backfill 후 diff (exit 0 = 정식 통과)
./.venv/bin/python scripts/backfill.py --symbols 005930
./.venv/bin/python serving/src/shadow_diff.py --date YYYY-MM-DD \
  --audit-dir serving/logs --manifest-dir serving/logs/shadow \
  --config serving/configs/serving.yaml
```

launchd 예시(선택): 일일 수집 잡과 같은 패턴으로 09:00 기동·수동 확인.

diff exit code: 0 통과 / 1 값 불일치 / 2 coverage 실패 / 3 artifact·SHA 불일치 / 4 입력 문제
