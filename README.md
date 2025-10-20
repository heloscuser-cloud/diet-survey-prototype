# 🍽️ 식습관 문진 · 프로토타입 (실서비스 근접)

- **OTP 로그인(프로토타입)**: 휴대폰 번호 + 6자리 코드(서버 로그에 출력) → 운영 시 SMS로 교체
- **설문 → 점수 → PDF 리포트 → 포털에서 다운로드**
- **Docker & Render 배포 파일 포함**

## 0) 설치

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt

# 환경변수 파일
copy .env.example .env   # macOS/Linux는 cp
```

`.env`에서 `APP_SECRET` 값을 바꿔주세요.

## 1) 실행

```bash
uvicorn app.main:app --reload --port 8000
```

- `http://localhost:8000` 접속 → 로그인 → 설문 → 포털에서 리포트 다운로드
- OTP 코드는 **터미널 로그**에 출력됩니다. (운영에서는 SMS API 사용)

## 2) 한글 폰트

`app/fonts/` 에 `NotoSansKR-Regular.otf`, `NotoSansKR-Bold.otf` 를 넣으면 PDF에서 한글이 정상 표시됩니다.

## 3) Docker로 배포(Render 예시)

1. GitHub에 푸시
2. https://render.com → New → Blueprint → `render.yaml` 선택
3. 배포 후 제공되는 URL로 접속

환경변수:
- `APP_SECRET`: 자동 생성(필요 시 변경)

## 4) 운영 전환 체크리스트

- OTP → **SMS 제공사 연동** (issue_otp 함수에서 교체)
- SQLite → **PostgreSQL** (DATABASE_URL 변경, SQLModel 마이그레이션)
- 로컬 파일 → **S3/오브젝트 스토리지** (다운로드는 서버에서 권한 체크 후 스트리밍)
- HTTPS, 쿠키 Secure/HttpOnly/SameSite 설정 강화
- 동의문/개인정보 처리방침/로그 마스킹
