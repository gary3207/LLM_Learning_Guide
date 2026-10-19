[← 목차](README.md)

# 가이드 추가·수정 방법

## 문서 작성 원칙

다른 독자가 코드만 보고도 순서대로 실습할 수 있도록 작성합니다.

1. **학습 목표와 선행 조건**
   - 무엇을 배우고 어느 장의 결과물이 필요한지 명시합니다.
   - 실행할 컴퓨터·터미널·현재 폴더·가상환경을 구분합니다.
2. **명령·전체 코드·해설**
   - 유지할 코드는 `source/`에 두고 상대 링크로 연결합니다.
   - 입력 → 변환 → 계산 → 출력 순서와 중요한 변수·텐서 크기를 설명합니다.
   - 생성·변경·덮어쓰는 파일을 명시합니다.
3. **실행 예시와 한계**
   - 확인한 출력은 참고 예시로 제시하고 환경·버전 차이를 설명합니다.
   - 예상 출력은 실제로 검증한 것처럼 표현하지 않습니다.
   - 작은 학습 데이터의 성공을 일반화나 서비스 성능으로 확대하지 않습니다.
4. **일반 독자 관점**
   - 작성자의 숙련도·진도·사정이나 대화 경위를 넣지 않습니다.
   - 내부 대화 식별자·접속정보·인증정보·업무 데이터를 포함하지 않습니다.
   - 설명은 “명령을 실행합니다”, “다음은 실행 예시입니다”, “결과를 확인합니다”처럼 작성합니다.

## 새 장 추가

- [장 작성 양식](templates/lesson.md)을 복사해 다음 번호의 Markdown을 루트에 만듭니다.
- 실습 코드와 필요한 검사는 `source/` 아래에 둡니다.
- `README.md` 목차와 이전/다음 링크를 갱신합니다.
- `07-resume.md`의 학습 점검·확장 범위와 `CHANGELOG.md`를 갱신합니다.
- 아직 구현하지 않은 주제는 확장 과제로 표시합니다.

### 새 파일이 Git에서 무시되지 않도록 허용하기

이 저장소는 가이드 파일만 추적하도록 `.gitignore`에 허용 목록을 사용합니다. 예를 들어 `08-example.md`와 `source/08-example/`를 추가한다면 `.gitignore`에 다음 두 줄도 추가합니다.

```gitignore
!/08-example.md
!/source/08-example/
```

Python 문법 검사에도 새 실습을 포함하려면 `source/check_guide.py`의 장별 디렉터리 목록에 `08-example`을 추가합니다. 새 장과 함께 `.gitignore` 및 검사 코드의 변경을 커밋합니다. 파일이 무시되는지 확인할 때는 `git check-ignore 08-example.md source/08-example/example.py`를 사용합니다. 허용된 경로라면 무시 규칙이 출력되지 않습니다.


## 게시 전 검사

**저장소 루트의 터미널**에서 실행합니다. 학습 VM에서 모델을 다시 학습하는 절차가 아닙니다.

```bash
python source/check_guide.py
python source/03-inference-and-data/verify_examples.py
```

첫 검사는 문서 링크·Python 문법·일부 민감 패턴을 확인합니다. 두 번째는 모델을 실행하지 않고 코드 전문 일치·데이터 분리·파일 보호를 검사합니다. 자동 검사만으로 모든 민감 내용이나 기술 오류를 찾을 수는 없으므로 본문도 검토합니다.

## GitHub에 반영

아래 파일명은 새 장을 추가하는 예시입니다. 실제 변경한 파일명으로 바꿉니다.

```bash
git status --short
git add .gitignore source/check_guide.py README.md CHANGELOG.md 07-resume.md 08-example.md source/08-example/
git diff --cached
git diff --cached --check
git commit -m "docs: add learning chapter"
git push origin main
```

| 명령 | 의미 |
|---|---|
| `git status` | 변경 상태 확인 |
| `git add` | 커밋할 파일 선택 |
| `git diff --cached` | 선택한 변경 검토 |
| `git commit` | 로컬 변경 이력 생성 |
| `git push` | 원격 저장소에 반영 |

오류 수정 시 수정 이유·영향·확인 방법을 설명합니다. 체크포인트·가상환경·비밀정보는 커밋하지 않습니다.
