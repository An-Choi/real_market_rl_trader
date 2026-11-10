# Git 커밋·푸시 준비

2026-11-10 로컬 작업 공간 기준으로 코드·문서·테스트를 staging했습니다.
커밋·원격 푸시는 아직 수행하지 않았습니다. 각 저장소의 변경 사항은 별도로 커밋합니다.

## Python 및 Java 저장소

각 저장소에서 staged diff를 확인한 뒤 작업 브랜치를 만들고 커밋합니다.
아래 브랜치가 이미 있으면 다른 이름을 사용하세요.

```bash
git diff --cached --stat
git diff --cached --check
git switch -c research/final-project
```

Python 저장소:

```bash
git commit -m "Add reproducible PPO trading research and diagnostics"
git push -u origin research/final-project
```

Java 저장소:

```bash
git commit -m "Add target-exposure research replay and ledger verification"
git push -u origin research/final-project
```

기존 SSH/HTTPS 인증 설정으로 푸시합니다. 토큰을 명령줄·URL·문서에 붙여 넣지 마세요.

## 데이터 저장소

이번 staging에는 .gitignore만 포함했습니다. 기존 추적 중인 원본 parquet 50개는
로컬 정리 결과이며 unstaged 상태로 보존했습니다. 원본 데이터 변경을 별도로 검토하기 전에는
`git add .`, `git add -A`, `git commit -a`를 사용하지 마세요.
`.gitignore`는 이미 추적 중인 원본 파일을 제외하지 않습니다.

```bash
git diff --cached --stat
git switch -c research/local-secret-ignore
git commit -m "Ignore local KIS credentials"
git push -u origin research/local-secret-ignore
```

## 포함 범위

- Python: 연구 환경·학습·평가·진단·테스트, 고정 의존성, README와 실행 안내, 작은 비교 그림/집계표
- Java: 연구용 재생 CLI, 회계 검증, 로컬 통합 테스트 지원과 안내
- 제외: 키·토큰, 원본 데이터 변경, 전체 실험 로그·모델, 압축 전달본, 가상환경과 로컬 도구

원래 작업 공간 최상위의 market.sh·FINAL_PROJECT.md·release 도구는 별도의 Git 저장소에
속하지 않습니다. GitHub 독자는 Python README와 reproduce.md의 단독 실행 명령을 사용합니다.
보관된 최종 압축본은 이전 시점의 검증된 전달본이며 이후 Git 공개용 문서 변경은 포함하지 않습니다.

## 검증 범위

Python 622개 테스트 및 Java 165개 테스트 통과 기록이 있습니다.
이번 Git 정리에서는 문서·ignore·의존성 목록·공유용 결과만 추가로 정리했습니다.
커밋 후보에서 현재 로컬 KIS 키·토큰 문자열과 5MiB 초과 파일을 검사했습니다.
Git 전체 이력에 대한 비밀정보 감사나 원격 푸시 성공을 의미하지 않습니다.
