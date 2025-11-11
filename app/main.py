from fastapi import (
    FastAPI, Request, Form, Depends, Response,
    Cookie, HTTPException, UploadFile, File,
    APIRouter, BackgroundTasks
)
from fastapi.staticfiles import StaticFiles
import re
from fastapi.templating import Jinja2Templates
from fastapi.routing import APIRoute
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse, JSONResponse
from sqlmodel import SQLModel, Field, Session, create_engine, select, Relationship
from pydantic import BaseModel
from typing import Optional, List, Any, Dict
from fastapi import Query
from datetime import datetime, timedelta, date, timezone
from starlette.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from random import randint
from sqlalchemy import func, Column, LargeBinary, Integer, text
from sqlalchemy import text as sa_text
from sqlalchemy.dialects.postgresql import JSONB
from itsdangerous import (
    TimestampSigner, BadSignature, SignatureExpired,
    URLSafeSerializer, URLSafeTimedSerializer
)
import os, smtplib, ssl, socket, traceback, time, sys, base64, hashlib
from Crypto.Cipher import AES
from email.message import EmailMessage
import zipfile
import csv
import itsdangerous
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from io import StringIO, BytesIO
from openpyxl import Workbook
import secrets
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import logging, pathlib
from app.vendors.datahub_client import DatahubClient, DatahubError, encrypt_field, _crypto_selftest, pick_latest_general


# ★ 로그 설정(강제적용)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True,
)

APP_ENV = (os.getenv("APP_ENV", "dev") or "").lower()
if APP_ENV == "prod":
    # uvicorn access 로그(요청 라인) 소음 감소
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    # 템플릿/SQL 등 과다 로거도 억제하고 싶으면 여기서 조절
    # logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

# ★ 현재 실행 파일 경로/버전 태그 찍기
logging.info("[BOOT] main.py loaded from %s", __file__)
logging.info("[BOOT] CWD=%s PYTHONPATH=%s", os.getcwd(), sys.path)
logging.info("[BOOT] BUILD_TS=%s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


# --- DataHub Client ---
print("[BOOT] creating DATAHUB client...")
DATAHUB = DatahubClient()
print("[BOOT] DATAHUB ready")


APP_SECRET = os.environ.get("APP_SECRET", "dev-secret")
ADMIN_USER = os.environ.get("ADMIN_USER")
ADMIN_PASS = os.environ.get("ADMIN_PASS")


NHIS_MAX_LIGHT_FETCH = int(os.getenv("NHIS_MAX_LIGHT_FETCH", "1"))     # light는 기본 1회만
NHIS_FETCH_INTERVAL  = float(os.getenv("NHIS_FETCH_INTERVAL", "2.0"))  # 폴링 간격(초)
NHIS_POLL_MAX_SEC    = int(os.getenv("NHIS_POLL_MAX_SEC", "120"))      # 최대 대기(초)

KST = ZoneInfo("Asia/Seoul")

def to_kst(dt: datetime) -> datetime:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(KST)

KST = timezone(timedelta(hours=9))
def now_kst():
    return datetime.now(tz=KST)

def kst_date_range_to_utc_datetimes(d_from: date | None, d_to: date | None):
    """
    KST 날짜 구간 [d_from, d_to] (둘 다 포함)을 UTC naive datetime 구간
    [start_utc, end_utc) 로 변환한다.
    DB의 naive UTC(datetime.utcnow())와 비교하기 위해 tzinfo 제거.
    """
    start_kst = datetime(d_from.year, d_from.month, d_from.day, 0, 0, 0, tzinfo=KST) if d_from else None
    end_kst   = datetime(d_to.year,   d_to.month,   d_to.day,   23,59,59,999999, tzinfo=KST) if d_to else None
    if d_to:
        # end exclusive: 다음날 00:00 KST
        end_kst = datetime(d_to.year, d_to.month, d_to.day, 0,0,0, tzinfo=KST) + timedelta(days=1)

    start_utc = start_kst.astimezone(timezone.utc).replace(tzinfo=None) if start_kst else None
    end_utc   = end_kst.astimezone(timezone.utc).replace(tzinfo=None)   if end_kst   else None
    return start_utc, end_utc

def ensure_not_completed(survey_completed: str | None = Cookie(default=None)):
    if survey_completed == "1":
        # 이미 완료된 세션은 설문으로 접근 시 포털로 보냄
        raise HTTPException(status_code=307, detail="completed")

# 질문 로드 (앱 기동시 1회)
QUESTIONS_PATH = Path("app/data/survey_questions.json")
with QUESTIONS_PATH.open("r", encoding="utf-8") as f:
    ALL_QUESTIONS = json.load(f)

# 페이지 그룹: (start_id, end_id)
SURVEY_STEPS = [(1, 8), (9, 16), (17, 23)]


def get_questions_for_step(step: int):
    start_id, end_id = SURVEY_STEPS[step-1]
    return [q for q in ALL_QUESTIONS if start_id <= q["id"] <= end_id]


# ---- Basic app setup ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DB_PATH = os.path.join(ROOT_DIR, "app", "data", "app.db")
os.makedirs(os.path.join(ROOT_DIR, "app", "data"), exist_ok=True)

DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DB_PATH}")
engine = create_engine(DATABASE_URL, echo=False)


app = FastAPI(title="Diet Survey Prototype")

app.mount("/static", StaticFiles(directory=os.path.join(ROOT_DIR, "app", "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(ROOT_DIR, "app", "templates"))

APP_SECRET = os.environ.get("APP_SECRET", "dev-secret")
signer = TimestampSigner(APP_SECRET)

# 임시 라우트 확인 필요
@app.on_event("startup")
def _on_startup():
    global DATAHUB
    logging.info("[BOOT] startup hook fired")
    try:
        if DATAHUB is None:
            logging.info("[BOOT] startup: creating DATAHUB client again...")
            DATAHUB = DatahubClient()
        # selftest 추가 호출(개발 모드에서만)
        app_env   = (os.getenv("APP_ENV", "dev") or "").strip().lower()
        st_flag   = (os.getenv("DATAHUB_SELFTEST", "1") or "").strip()
        logging.info("[BOOT] startup env: APP_ENV=%s SELFTEST=%s", app_env, st_flag)
        if app_env != "prod" and st_flag == "1":
            _crypto_selftest()
    except Exception as e:
        logging.exception("[BOOT] startup error: %r", e)


@app.exception_handler(HTTPException)
async def completed_redirect_handler(request: Request, exc: HTTPException):
    # 설문 뒤로가기 차단 307 → 메인
    if exc.status_code == 307 and exc.detail == "completed":
        return RedirectResponse(url="/", status_code=307)
       # 관리자 호스트 강제 307 → Location 헤더 그대로 사용
    if exc.status_code == 307 and exc.detail == "admin-host-redirect":
        return RedirectResponse(url=exc.headers.get("Location") or "/", status_code=307)

    # ★ 관리자 보호: 401이면 로그인 화면으로
    if exc.status_code == 401 and (request.url.path or "").startswith("/admin"):
        return RedirectResponse(url="/admin/login", status_code=303)
    return await http_exception_handler(request, exc)

@app.middleware("http")
async def no_store_for_survey(request: Request, call_next):
    p = request.url.path
    if p == "/healthz":
        return await call_next(request)  # 최소 비용 통과
    response = await call_next(request)
    if p.startswith("/survey"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response

# ---- Font registration (optional for Korean) ----
FONT_DIR = os.path.join(ROOT_DIR, "app", "fonts")
if os.path.isdir(FONT_DIR):
    try:
        pdfmetrics.registerFont(TTFont("NotoSansKR", os.path.join(FONT_DIR, "NotoSansKR-Regular.otf")))
        pdfmetrics.registerFont(TTFont("NotoSansKR-Bold", os.path.join(FONT_DIR, "NotoSansKR-Bold.otf")))
        pdfmetrics.registerFontFamily("NotoSansKR", normal="NotoSansKR", bold="NotoSansKR-Bold")
        DEFAULT_FONT = "NotoSansKR"
        DEFAULT_FONT_BOLD = "NotoSansKR-Bold"
    except Exception as e:
        print("폰트 등록 실패:", e)
        DEFAULT_FONT = "Helvetica"
        DEFAULT_FONT_BOLD = "Helvetica-Bold"
else:
    DEFAULT_FONT = "Helvetica"
    DEFAULT_FONT_BOLD = "Helvetica-Bold"

AUTH_COOKIE_NAME = "auth"
AUTH_MAX_AGE = 3600 * 1  # 1 hours

def sign_user(user_id: int) -> str:
    return signer.sign(f"user:{user_id}").decode("utf-8")

def verify_user(token: str) -> int:
    try:
        raw = signer.unsign(token, max_age=AUTH_MAX_AGE).decode("utf-8")
        if not raw.startswith("user:"):
            return -1
        return int(raw.split(":")[1])
    except Exception:
        return -1

# --- 민감값 마스킹 헬퍼 (재사용) --- #
def _mask_phone(s: str) -> str:
    if not s: return ""
    d = re.sub(r"[^0-9]", "", s)
    if len(d) < 7: return "***"
    return d[:3] + "-" + "*"*4 + "-" + d[-4:]

def _mask_birth(s: str) -> str:
    # YYYYMMDD → YYYY-**-**
    if not s: return ""
    d = re.sub(r"[^0-9]", "", s)
    if len(d) >= 8:
        return f"{d[:4]}-**-**"
    return "***"



# ---- Models ----
class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    phone_hash: str
    name_enc: Optional[str] = None
    birth_year: Optional[int] = None
    gender: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    birth_date: Optional[date] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None

class SurveyResponse(SQLModel, table=True):
    # ▶ 테이블명 고정 + 중복 정의 방지 보호막
    __tablename__ = "surveyresponse"
    __table_args__ = {"extend_existing": True}

    # ▶ PK 반드시 필요
    id: Optional[int] = Field(default=None, primary_key=True)

    # ▶ 기존 필드들 (당신 코드 기준으로 이름 유지)
    respondent_id: Optional[int] = None
    answers_json: Optional[str] = None
    score: Optional[int] = None
    submitted_at: Optional[datetime] = None

    # ▶ NHIS 컬럼: JSONB로 정확히 선언 (dict를 그대로 넣어도 저장됨)
    nhis_json: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSONB))
    nhis_raw:  Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSONB))

class Respondent(SQLModel, table=True):
    __tablename__ = "respondent"
    # (선택 안전장치) 이미 같은 테이블이 메타데이터에 있을 경우 재정의 허용
    __table_args__ = {"extend_existing": True}

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(index=True)
    campaign_id: str = Field(default="default")
    status: str = Field(default="draft")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # 인적정보 스냅샷
    applicant_name: str | None = None
    birth_date: date | None = None
    gender: str | None = None
    height_cm: float | None = None
    weight_kg: float | None = None

    # 신청번호(고정 순번) — DB 시퀀스에서 자동 발급
    serial_no: int | None = Field(
        default=None,
        sa_column=Column(
            Integer,
            server_default=text("nextval('respondent_serial_no_seq')"),
            unique=True,
            index=True,
        ),
    )


class ReportFile(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    survey_response_id: int = Field(index=True)
    filename: str
    content: bytes = Field(sa_column=Column(LargeBinary))
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
  
class Otp(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    phone: str = Field(index=True)
    code: str
    expires_at: datetime
    consumed: bool = Field(default=False)

def init_db():
    SQLModel.metadata.create_all(engine)

@app.on_event("startup")
def on_startup():
    init_db()


# --- NHIS 저장 헬퍼 (rtoken 없이 rid가 확실할 때) ---
def _save_nhis_to_db_with_id(session, respondent_id: int, picked: dict, raw: dict):
    try:
        from sqlalchemy import text as sa_text
        stmt = sa_text("""
            UPDATE surveyresponse
               SET nhis_json = :js,
                   nhis_raw  = :raw
             WHERE respondent_id = :rid
        """).bindparams(
            rid=respondent_id,
            js=json.dumps(picked or {}, ensure_ascii=False),
            raw=json.dumps(raw or {}, ensure_ascii=False),
        )
        session.exec(stmt)
        session.commit()
        logging.info("[NHIS][DB] saved-by-id rid=%s (json=%s, raw=%s)",
                     respondent_id, bool(picked), bool(raw))
    except Exception as e:
        logging.error("[NHIS][DB][ERR][by-id] %r", e)


# --- NHIS 저장 헬퍼 (rtoken으로 rid 복구 시도) ---
def _save_nhis_to_db(session, request, picked: dict, raw: dict):
    try:
        rid = None
        try:
            tok = (request.query_params.get("rtoken") or request.cookies.get("rtoken") or "")
            rid = verify_token(tok) if tok else -1
            if rid and rid < 0:
                rid = None
        except Exception:
            rid = None

        if not rid:
            logging.info("[NHIS][DB] skip save: no respondent_id")
            return

        _save_nhis_to_db_with_id(session, rid, picked, raw)
    except Exception as e:
        logging.error("[NHIS][DB][ERR] %r", e)



#---- 공단검진 결과 데이터 가공/저장 헬퍼 (legacy)----#

def pick_latest_general_checkup(nhis_data: dict) -> dict | None:
    """
    데이터허브 응답(data.INCOMELIST[])에서 최근 10년 내 가장 최근 1건을 반환.
    일반검진만 포함(암검진 제외) 규칙이 명시돼 있으면 필드로 구분, 없으면 전체에서 최신.
    """
    items = (nhis_data or {}).get("INCOMELIST") or []
    if not items:
        return None

    def parse_date(y, md):
        # GUNYEAR: "2022", GUNDATE: "11/02" 형태 가정
        try:
            y = int(str(y).strip()[:4])
            m, d = md.split("/")
            return datetime(y, int(m), int(d))
        except Exception:
            return None

    ten_years_ago = datetime.now() - timedelta(days=365*10)
    candidates = []
    for it in items:
        dt = parse_date(it.get("GUNYEAR"), it.get("GUNDATE",""))
        if dt and dt >= ten_years_ago:
            # 일반/암 구분이 따로 온다면 여기서 필터(it.get("TYPE") == "GENERAL" 같은 식)
            candidates.append((dt, it))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]




# ---- Simple helpers ----
def hash_phone(phone: str) -> str:
    # naive demo hash (do NOT use in prod). Replace with SHA-256 + salt.
    return "ph_" + str(abs(hash("SALT::" + phone)))

def get_session():
    with Session(engine) as session:
        yield session

#---- 이메일 접수 알림 헬퍼 ----
def mask_second_char(name: str | None) -> str:
    """신청자 이름의 두 번째 글자를 *로 가림 (한글 포함, 1글자면 그대로)"""
    if not name:
        return ""
    s = list(name)
    if len(s) >= 2:
        s[1] = "*"
    return "".join(s)

def send_submission_email(serial_no: int, applicant_name: str, created_at_kst_str: str):
    host = os.getenv("SMTP_HOST")
    port_env = os.getenv("SMTP_PORT", "").strip()
    user = (os.getenv("SMTP_USER") or "").strip()
    password = (os.getenv("SMTP_PASS") or "").strip()
    mail_from = (os.getenv("SMTP_FROM") or "").strip()
    mail_to = (os.getenv("SMTP_TO") or "").strip()
    timeout = int(os.getenv("SMTP_TIMEOUT", "25"))  # 넉넉히 25초

    # 환경 체크
    if not (host and user and password and mail_from and mail_to):
        print("[EMAIL] SMTP env not configured, skip.")
        return

    # 네이버는 보통 '전체 이메일 주소'로 로그인해야 안정적
    login_user = user if "@" in user else f"{user}@naver.com"

    # 메일 만들기
    msg = EmailMessage()
    msg["Subject"] = f"[GaonnSurvey] 새 문진 접수 #{serial_no}"
    msg["From"] = mail_from
    msg["To"] = mail_to
    body = (
        f"새 문진이 접수되었습니다.\n"
        f"- 일련번호: {serial_no}\n"
        f"- 신청자: {applicant_name or '(미입력)'}\n"
        f"- 접수시각(KST): {created_at_kst_str}\n"
        f"\n관리자 페이지에서 확인하세요."
    )
    msg.set_content(body)

    ctx = ssl.create_default_context()

    def try_587():
        print("[EMAIL] connecting 587 STARTTLS...")
        with smtplib.SMTP(host, 587, timeout=timeout) as s:
            s.set_debuglevel(1)  # SMTP 대화 로그 출력
            s.ehlo()
            s.starttls(context=ctx)
            s.ehlo()
            s.login(login_user, password)
            s.send_message(msg)
        print("[EMAIL] sent OK via 587")

    def try_465():
        print("[EMAIL] connecting 465 SSL...")
        with smtplib.SMTP_SSL(host, 465, timeout=timeout, context=ctx) as s:
            s.set_debuglevel(1)
            s.login(login_user, password)
            s.send_message(msg)
        print("[EMAIL] sent OK via 465")

    # IPv6 경로가 느린 환경에서 타임아웃을 줄이기 위해(선택) IPv4 우선 DNS 확인 로그
    try:
        ipv4s = [ai[4][0] for ai in socket.getaddrinfo(host, None, socket.AF_INET)]
        print(f"[EMAIL] DNS A records (IPv4): {ipv4s}")
    except Exception as _e:
        print("[EMAIL] DNS A lookup failed (non-fatal):", repr(_e))

    # 시도: 587 → 실패 시 465 폴백
    try:
        try_587()
        return
    except Exception as e1:
        print("[EMAIL] 587 failed:", repr(e1))
        traceback.print_exc()

    try:
        try_465()
        return
    except Exception as e2:
        print("[EMAIL] 465 failed:", repr(e2))
        traceback.print_exc()

    print("[EMAIL] send failed: both 587 and 465 attempts failed")


# ---- OTP helpers ----
def issue_otp(session: Session, phone: str) -> str:
    code = f"{randint(0, 999999):06d}"
    otp = Otp(phone=phone, code=code, expires_at=datetime.utcnow() + timedelta(minutes=5))
    session.add(otp)
    session.commit()
    print(f"[OTP] {phone} -> code: {code} (5분 유효)")  # prototype: replace with SMS provider
    return code

def verify_otp(session: Session, phone: str, code: str) -> bool:
    stmt = select(Otp).where(Otp.phone == phone, Otp.code == code, Otp.consumed == False)
    otp = session.exec(stmt).first()
    if not otp:
        return False
    if otp.expires_at < datetime.utcnow():
        return False
    otp.consumed = True
    session.add(otp)
    session.commit()
    return True

# ---- Routes ----
def _host(request: Request) -> str:
    return (request.headers.get("host") or "").split(":")[0].lower()

ADMIN_HOST = "admin.gaonnsurvey.store"

@app.get("/info", response_class=HTMLResponse)
def info_form(request: Request, auth: str | None = Cookie(default=None, alias=AUTH_COOKIE_NAME)):
    user_id = verify_user(auth) if auth else -1
    if user_id < 0:
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse("info.html", {"request": request})

# -----------------------------------------------
# NHIS 건강검진 조회 페이지 (info 전 단계)
# -----------------------------------------------

@app.get("/nhis")
def nhis_page(request: Request):
    auth_base = os.getenv("DATAHUB_API_BASE", "https://datahub-dev.scraping.co.kr").rstrip("/")
    return templates.TemplateResponse(
        "nhis_fetch.html",
        {"request": request, "next_url": "/survey", "datahub_auth_base": auth_base}
    )

    
@app.get("/healthz")
def healthz():
    return PlainTextResponse("ok", status_code=200)

@app.post("/info")
async def info_submit(
    request: Request,
    name: str = Form(...),
    birth_date: str = Form(...),   # "YYMMDD"
    gender: str = Form(...),       # "남" | "여"
    height_cm: str = Form(None),
    weight_kg: str = Form(None),
    auth: str | None = Cookie(default=None, alias=AUTH_COOKIE_NAME),
    session: Session = Depends(get_session),
):
    user_id = verify_user(auth) if auth else -1
    if user_id < 0:
        return RedirectResponse(url="/login", status_code=302)

    # 간단 검증
    try:
        bd = datetime.strptime(birth_date, "%Y-%m-%d").date()
    except:
        return templates.TemplateResponse("error.html", {"request": request, "message": "생년월일 형식이 올바르지 않습니다(YYMMDD)."}, status_code=400)
    if gender not in ("남", "여"):
        return templates.TemplateResponse("error.html", {"request": request, "message": "성별은 남/여 중 선택해주세요."}, status_code=400)

    # 숫자 파싱(선택)
    def to_float(s):
        try:
            return float(s) if s not in (None, "") else None
        except:
            return None
    h_cm = to_float(height_cm)
    w_kg = to_float(weight_kg)

    # 다음 설문 세션 Respondent에 스냅샷으로 옮길 수 있도록 User에도 저장(옵션)
    user = session.get(User, user_id)
    if user:
        user.name_enc = name.strip()
        user.gender = gender.strip()
        user.birth_year = bd.year                       # 호환용 유지
        user.birth_date = bd                            # 실제 생년월일 저장
        user.height_cm = h_cm
        user.weight_kg = w_kg
        session.add(user)
        session.commit()   

    # 설문 시작
    return RedirectResponse(url="/survey", status_code=303)

# --- Session (admin 인증 단일 쿠키) ---
SESSION_MAX_AGE = 30 * 60  # 30분

app.add_middleware(
    SessionMiddleware,
    secret_key=APP_SECRET,   # (이미 위쪽에 APP_SECRET가 있음)
    max_age=SESSION_MAX_AGE, # 초 단위
    same_site="none",        # 서브도메인/리다이렉트 고려
    https_only=True          # Secure
)

@app.middleware("http")
async def rolling_session_middleware(request: Request, call_next):
    # 요청 처리
    response = await call_next(request)

    # admin 세션이면 만료 임박 시 갱신(쿠키 재발급)
    if request.session.get("admin"):
        now = int(datetime.now(timezone.utc).timestamp())
        issued_at = int(request.session.get("_iat", 0))
        # 남은 시간 < 5분이면 갱신
        if now - issued_at > (SESSION_MAX_AGE - 300):
            request.session["_iat"] = now  # 세션 값 변경 -> Set-Cookie 재발급

    return response

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    if _host(request) == ADMIN_HOST:
        # 관리자 서브도메인으로 들어오면 관리자 로그인으로 보냄
        return RedirectResponse(url="/admin/login", status_code=302)
    # 기존 사용자용 홈 유지
    return templates.TemplateResponse("index.html", {"request": request})

#사용자 로그인 화면 렌더
@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


#로그인 코드 검증, 진행
@app.post("/login/verify")
def login_verify_phone(
    request: Request,
    phone: str = Form(...),
    session: Session = Depends(get_session),
):
    # 1) 폰번호 정규화
    phone_digits = "".join(c for c in (phone or "") if c.isdigit())
    if len(phone_digits) < 10 or len(phone_digits) > 11:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "번호 형식이 올바르지 않습니다."},
            status_code=400,
        )

    # 2) user_admin에서 코드(=phone) 존재/활성 확인
    row = session.exec(
        sa_text("""
            SELECT id, name, phone, is_active
            FROM user_admin
            WHERE phone = :p AND is_active = TRUE
            LIMIT 1
        """).bindparams(p=phone_digits)
    ).first()
    if not row:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "등록되지 않은 코드입니다."},
            status_code=401,
        )

    # 3) User 조회/생성 (기존 방식 유지: phone_hash 기반)
    ph = hash_phone(phone_digits)
    user = session.exec(select(User).where(User.phone_hash == ph)).first()
    if not user:
        user = User(phone_hash=ph)
        session.add(user)
        session.commit()
        session.refresh(user)

    # 4) AUTH 쿠키 발급 (기존 방식 그대로)
    SECURE_COOKIE = bool(int(os.getenv("SECURE_COOKIE", "1")))
    resp = RedirectResponse(url="/nhis", status_code=303)
    resp.set_cookie(
        AUTH_COOKIE_NAME,
        sign_user(user.id),
        httponly=True,
        secure=SECURE_COOKIE,
        samesite="lax",
        max_age=AUTH_MAX_AGE,
    )
    # 혹시 남아있을 수도 있는 완료 쿠키 제거(새 설문 방해 방지)
    resp.delete_cookie("survey_completed")

    # 5) Respondent 생성 + rtoken 쿠키(설문 접근/후속 저장에 필요)
    try:
        rid = session.exec(
            sa_text("INSERT INTO respondent (status, created_at) VALUES ('started', now()) RETURNING id")
        ).first()[0]

        # 프로젝트에 이미 있는 signer를 재사용해 rtoken 생성 (verify_token과 호환)
        try:
            tok = signer.sign(f"rid:{rid}").decode("utf-8")
        except Exception:
            tok = str(rid)  # 임시(가능하면 signer 사용 권장)

        request.session["rtoken"] = tok
        resp.set_cookie("rtoken", tok, max_age=60*60*2, httponly=True, samesite="Lax", secure=SECURE_COOKIE)
    except Exception as e:
        # rtoken 발급 실패해도 로그인은 진행되지만, /survey 접근 가드에서 막힐 수 있음
        # 문제 시 로그만 남기고 그대로 진행
        logging.debug("[LOGIN][RTOKEN][WARN] %r", e)

    # 6) 감사용 세션 마크(선택)
    request.session["admin_phone"] = phone_digits

    return resp


@app.get("/logout")
def logout():
    resp = RedirectResponse(url="/", status_code=302)
    resp.delete_cookie(AUTH_COOKIE_NAME)
    return resp


def verify_token(token: str) -> int:
    try:
        raw = signer.unsign(token, max_age=3600*3)
        return int(raw.decode("utf-8"))
    except (BadSignature, SignatureExpired):
        return -1


@app.get("/report/ready", response_class=HTMLResponse)
def report_ready(request: Request, rtoken: str):
    return templates.TemplateResponse("report_ready.html", {"request": request, "rtoken": rtoken})

@app.get("/portal", response_class=HTMLResponse)
def portal(request: Request, auth: str | None = Cookie(default=None, alias=AUTH_COOKIE_NAME), session: Session = Depends(get_session)):
    user_id = verify_user(auth) if auth else -1
    if user_id < 0:
        return RedirectResponse(url="/login", status_code=302)

    resps = session.exec(select(Respondent).where(Respondent.user_id == user_id)).all()
    reports = []
    for r in resps:
        srs = session.exec(select(SurveyResponse).where(SurveyResponse.respondent_id == r.id)).all()
        for sr in srs:
            reports.append({
                "id": sr.id,
                "submitted_at": to_kst(sr.submitted_at).strftime("%Y-%m-%d %H:%M"),
                "score": sr.score
            })
    reports.sort(key=lambda x: x["id"], reverse=True)
    return templates.TemplateResponse("portal.html", {
        "request": request,
    })

def _normalize_date_str(s: str) -> str | None:
    """
    'YYYY-M-D' 같이 들어와도 'YYYY-MM-DD'로 0패딩해서 돌려줍니다.
    날짜 형식이 아니면 None.
    """
    s = (s or "").strip()
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if not m:
        return None
    y, mm, dd = m.groups()
    try:
        return f"{int(y):04d}-{int(mm):02d}-{int(dd):02d}"
    except ValueError:
        return None



#---- 관리자 로그인 ----#

def admin_required(request: Request):
    # 세션에 admin 플래그가 있으면 통과
    try:
        if request.session.get("admin"):
            return
    except Exception:
        pass
    raise HTTPException(status_code=401)

@app.get("/admin/login", response_class=HTMLResponse)
def admin_login_form(request: Request):
    # 에러 메시지 표시용 기본값 포함
    return templates.TemplateResponse("admin/login.html", {"request": request, "error": None})


# 관리자 전용 라우터
admin_router = APIRouter(
    prefix="/admin",
    dependencies=[Depends(admin_required)]
)

@app.post("/admin/login")
async def admin_login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):

    # 1) ENV에서 계정 목록 읽기 (콤마 지원)
    #    예: ADMIN_USER="admin1,admin2"  ADMIN_PASS="pass1,pass2"
    env_users = [u.strip() for u in (os.getenv("ADMIN_USER") or "").replace("\n", "").split(",") if u.strip()]
    env_pwds  = [p.strip() for p in (os.getenv("ADMIN_PASS") or "").replace("\n", "").split(",") if p.strip()]

    # 2) 방어: 개수 불일치 시 뒤쪽 잘라내기
    n = min(len(env_users), len(env_pwds))
    env_users = env_users[:n]
    env_pwds  = env_pwds[:n]

    # 3) 1:1 인덱스 매칭으로 검증
    valid = any((username == env_users[i] and password == env_pwds[i]) for i in range(n))

    if not valid:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "message": "인증 실패"},
            status_code=401,
        )

    # 4) 세션 발급 (기존 키 그대로)
    request.session.clear()
    request.session["admin"] = True
    request.session["_iat"] = int(datetime.now(timezone.utc).timestamp())

    return RedirectResponse(url="/admin/responses", status_code=303)


@app.get("/admin/logout")
def admin_logout(request: Request):
    request.session.clear()
    resp = RedirectResponse(url="/admin/login", status_code=303)
    # 세션 쿠키 이름은 기본 "session"
    resp.delete_cookie("session", path="/")
    return resp



# 목록
@admin_router.get("/responses", response_class=HTMLResponse)
def admin_responses(
    request: Request,
    response: Response,
    page: int = 1,
    page_size: str = "50",  # "50"(기본) / "100" / "all"
    q: Optional[str] = None,
    status: Optional[str] = None,  # submitted/accepted/report_uploaded or ""
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = None,
    session: Session = Depends(get_session),
):
    # 안전 파싱
    try:
        page = max(1, int(page))
    except Exception:
        page = 1

    # --- page_size 해석 ---
    page_size_norm = (page_size or "50").lower()
    if page_size_norm == "100":
        PAGE_SIZE = 100
    elif page_size_norm == "all":
        PAGE_SIZE = None  # 전체 보기
    else:
        PAGE_SIZE = 50

    # --- 기본 쿼리 ---
    stmt = (
        select(SurveyResponse, Respondent, User, ReportFile)
        .join(Respondent, Respondent.id == SurveyResponse.respondent_id)
        .join(User, User.id == Respondent.user_id)
        .join(ReportFile, ReportFile.survey_response_id == SurveyResponse.id, isouter=True)
    )

    # --- 검색어 필터 (생년월일 yyyy-mm-dd 지원) ---
    if q:
        like = f"%{q}%"
        q_birth = _normalize_date_str(q)

        if q_birth:
            # 정확한 일자 매칭(=) + 기존 like들
            stmt = stmt.where(
                (func.to_char(User.birth_date, "YYYY-MM-DD") == q_birth)
                | (func.to_char(Respondent.birth_date, "YYYY-MM-DD") == q_birth)
                | (User.name_enc.ilike(like))
                | (User.phone_hash.ilike(like))
                | (SurveyResponse.answers_json.ilike(like))
                | (ReportFile.filename.ilike(like))
                | (func.to_char(Respondent.created_at, "YYYY-MM-DD").ilike(like))
            )
        else:
            # 일반 텍스트 검색: 생년월일도 부분검색 허용
            stmt = stmt.where(
                (User.name_enc.ilike(like))
                | (User.phone_hash.ilike(like))
                | (SurveyResponse.answers_json.ilike(like))
                | (ReportFile.filename.ilike(like))
                | (func.to_char(Respondent.created_at, "YYYY-MM-DD").ilike(like))
                | (func.to_char(User.birth_date, "YYYY-MM-DD").ilike(like))
                | (func.to_char(Respondent.birth_date, "YYYY-MM-DD").ilike(like))
            )


    # --- 날짜 필터 ---
    def parse_date(s: Optional[str]):
        try:
            return datetime.strptime(s, "%Y-%m-%d").date()
        except Exception:
            return None

    d_from = parse_date(from_)
    d_to = parse_date(to)
    start_utc, end_utc = kst_date_range_to_utc_datetimes(d_from, d_to)
    if start_utc:
        stmt = stmt.where(Respondent.created_at >= start_utc)
    if end_utc:
        stmt = stmt.where(Respondent.created_at < end_utc)

    # --- 상태 필터 ---
    if status in ("submitted", "accepted", "report_uploaded"):
        stmt = stmt.where(Respondent.status == status)

    # --- 정렬: 최신 제출일 → 응답 ID 내림차순 ---
    stmt = stmt.order_by(
        SurveyResponse.submitted_at.desc(),
        SurveyResponse.id.desc()
    )

    # --- 전체 개수 ---
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = session.exec(count_stmt).one()

    # --- 페이징 ---
    if PAGE_SIZE is None:
        # 전체보기: offset/limit 제거
        rows = session.exec(stmt).all()
        total_pages = 1
        page = 1
    else:
        rows = session.exec(
            stmt.offset((page - 1) * PAGE_SIZE).limit(PAGE_SIZE)
        ).all()
        total_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE

    # --- 출력 변환 ---
    to_kst_str = lambda dt: to_kst(dt).strftime("%Y-%m-%d %H:%M")


    # --- 렌더 ---
    return templates.TemplateResponse(
        "admin/responses.html",
        {
            "request": request,
            "rows": rows,
            "page": page,
            "total": total,
            "total_pages": total_pages,
            "from": from_,
            "to": to,
            "status": status or "",
            "q": q,
            "page_size": page_size_norm,
            "to_kst_str": to_kst_str,
        },
    )


# 접수완료 처리 (POST + Form)
@admin_router.post("/responses/accept")
async def admin_bulk_accept(
    request: Request,
    response: Response, 
    ids: str = Form(...),  # "1,2,3"
    session: Session = Depends(get_session),
):

    # ... 나머지 처리 및 RedirectResponse 반환
    id_list = [int(x) for x in ids.split(",") if x.strip().isdigit()]
    if not id_list:
        return RedirectResponse(url="/admin/responses", status_code=303)
    srs = session.exec(select(SurveyResponse).where(SurveyResponse.id.in_(id_list))).all()
    for sr in srs:
        resp = session.get(Respondent, sr.respondent_id)
        if resp:
            resp.status = "accepted"
            session.add(resp)
    session.commit()
    return RedirectResponse(url="/admin/responses", status_code=303)


# 리포트 업로드 (POST + multipart/form-data)
@admin_router.post("/response/{rid}/report")
async def admin_upload_report(
    rid: int,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    sr = session.get(SurveyResponse, rid)
    if not sr:
        raise HTTPException(status_code=404, detail="not found")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF만 업로드 가능합니다.")
    content = await file.read()

    # 기존 파일 있으면 교체
    old = session.exec(select(ReportFile).where(ReportFile.survey_response_id == rid)).first()
    if old:
        session.delete(old); session.commit()

    rf = ReportFile(survey_response_id=rid, filename=file.filename, content=content)
    session.add(rf)

    # 상태 업데이트
    resp = session.get(Respondent, sr.respondent_id)
    if resp:
        resp.status = "report_uploaded"
        session.add(resp)

    session.commit()
    return RedirectResponse(url="/admin/responses", status_code=303)


# 리포트 삭제 (POST)
@admin_router.post("/response/{rid}/report/delete")
def admin_delete_report(
    rid: int,
    session: Session = Depends(get_session),
):
    sr = session.get(SurveyResponse, rid)
    if not sr:
        raise HTTPException(status_code=404, detail="not found")
    rf = session.exec(select(ReportFile).where(ReportFile.survey_response_id == rid)).first()
    if rf:
        session.delete(rf)
    # 상태는 업로드 이전 단계로(접수완료 유지)
    resp = session.get(Respondent, sr.respondent_id)
    if resp and resp.status == "report_uploaded":
        resp.status = "accepted"
        session.add(resp)
    session.commit()
    return RedirectResponse(url="/admin/responses", status_code=303)


# CSV (GET)
@admin_router.get("/responses.csv")
def admin_responses_csv(
    q: Optional[str] = None,
    min_score: Optional[str] = None,
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = Query(None, alias="to"),
    status: Optional[str] = None,
    session: Session = Depends(get_session),
):
    def parse_date(s: Optional[str]):
        try: return datetime.strptime(s, "%Y-%m-%d").date()
        except: return None
    d_from, d_to = parse_date(from_), parse_date(to)

    def to_kst_str(dt):
        return to_kst(dt).strftime("%Y-%m-%d %H:%M") if dt else ""

    stmt = (
        select(SurveyResponse, Respondent, User, ReportFile)
        .join(Respondent, Respondent.id == SurveyResponse.respondent_id)
        .join(User, User.id == Respondent.user_id)
        .join(ReportFile, ReportFile.survey_response_id == SurveyResponse.id, isouter=True)
    )

    if q:
        like = f"%{q}%"
        q_birth = _normalize_date_str(q)

        if q_birth:
            stmt = stmt.where(
                (func.to_char(User.birth_date, "YYYY-MM-DD") == q_birth)
                | (func.to_char(Respondent.birth_date, "YYYY-MM-DD") == q_birth)
                | (User.name_enc.ilike(like))
                | (User.phone_hash.ilike(like))
                | (SurveyResponse.answers_json.ilike(like))
                | (ReportFile.filename.ilike(like))
                | (func.to_char(Respondent.created_at, "YYYY-MM-DD").ilike(like))
            )
        else:
            stmt = stmt.where(
                (User.name_enc.ilike(like))
                | (User.phone_hash.ilike(like))
                | (SurveyResponse.answers_json.ilike(like))
                | (ReportFile.filename.ilike(like))
                | (func.to_char(Respondent.created_at, "YYYY-MM-DD").ilike(like))
                | (func.to_char(User.birth_date, "YYYY-MM-DD").ilike(like))
                | (func.to_char(Respondent.birth_date, "YYYY-MM-DD").ilike(like))
            )


    try:
        min_score_i = int(min_score) if (min_score not in (None, "")) else None
    except ValueError:
        min_score_i = None
    if min_score_i is not None:
        stmt = stmt.where(SurveyResponse.score >= min_score_i)

    # 날짜 필터(KST→UTC)
    start_utc, end_utc = kst_date_range_to_utc_datetimes(d_from, d_to)
    if start_utc:
        stmt = stmt.where(Respondent.created_at >= start_utc)
    if end_utc:
        stmt = stmt.where(Respondent.created_at < end_utc)

    if status in ("submitted", "accepted", "report_uploaded"):
        stmt = stmt.where(Respondent.status == status)

    def generate():
        yield "\ufeff"
        s = StringIO(); w = csv.writer(s)
        w.writerow(["신청번호","신청자","PDF 파일명","업로드 날짜","진행 상태값","신청일","제출일"])
        yield s.getvalue(); s.seek(0); s.truncate(0)

        result = session.exec(stmt.order_by(SurveyResponse.submitted_at.desc())).all()
        for idx, (sr, resp, user, rf) in enumerate(result, start=1):
            applicant = f"{resp.applicant_name or (user.name_enc or '')} ({(resp.birth_date or '')}, {resp.gender or (user.gender or '')})"
            status_h = "신청완료" if resp.status=="submitted" else "접수완료" if resp.status=="accepted" else "리포트 업로드 완료" if resp.status=="report_uploaded" else (resp.status or "")
            row = [
                resp.serial_no or "",
                applicant,
                (rf.filename if rf else ""),
                (to_kst_str(rf.uploaded_at) if rf else ""),
                status_h,
                to_kst_str(resp.created_at),
                to_kst_str(sr.submitted_at),
            ]
            w.writerow(row)
            yield s.getvalue(); s.seek(0); s.truncate(0)

    return StreamingResponse(
        generate(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="responses.csv"'}
    )


# 개별 응답 2행 CSV (GET)
@admin_router.get("/response/{rid}.csv")
def admin_response_two_rows_csv(
    rid: int,
    session: Session = Depends(get_session),
):
    sr = session.get(SurveyResponse, rid)
    if not sr:
        raise HTTPException(status_code=404, detail="not found")

    try:
        payload = json.loads(sr.answers_json) if sr.answers_json else {}
    except Exception:
        payload = {}

    answers = payload.get("answers_indices") or []

    # 질문 타이틀 추출(질문 id 오름차순, 키 다양성 대응)
    def q_title(q: dict):
        return q.get("title") or q.get("text") or q.get("label") or q.get("question") or q.get("prompt") or q.get("name") or f"Q{q.get('id','')}"
    questions = [q_title(q) for q in sorted(ALL_QUESTIONS, key=lambda x: x["id"])]

    def fmt(val):
        if isinstance(val, list):
            return ",".join(str(v) for v in val)
        return "" if val is None else str(val)

    answer_numbers = [fmt(v) for v in answers]

    sio = StringIO()
    w = csv.writer(sio)
    w.writerow(questions)
    w.writerow(answer_numbers)
    csv_bytes = sio.getvalue().encode("utf-8-sig")

    return Response(
        content=csv_bytes,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename=\"response_{rid}.csv\"'},
    )

app.include_router(admin_router)

#---- 문진 가져오기 ----#

@app.get("/survey")

def survey_root(auth: str | None = Cookie(default=None, alias=AUTH_COOKIE_NAME),
                session: Session = Depends(get_session),):
    user_id = verify_user(auth) if auth else -1
    if user_id < 0:
        return RedirectResponse(url="/login", status_code=302)

    user = session.get(User, user_id)

    # 임시로그. 문진 정상 동작시 삭제 (251024)
    print("SURVEY GUARD",
      "name=", bool(user and user.name_enc),
      "gender=", bool(user and user.gender),
      "birth_date=", bool(user and getattr(user, "birth_date", None)),
      "birth_year=", bool(user and user.birth_year))

    # 필수 인적사항: 이름/성별 + (birth_date 또는 birth_year)
    has_birth = bool(getattr(user, "birth_date", None) or getattr(user, "birth_year", None))
    if not user or not user.name_enc or not user.gender or not has_birth:
        return RedirectResponse(url="/info", status_code=303)
    resp = Respondent(user_id=user.id, campaign_id="demo", status="draft")
    session.add(resp)
    session.commit()
    session.refresh(resp)
    
    # User 정보 스냅샷을 Respondent에 저장(관리자 테이블 출력용)
    # 실제 생년월일 우선 스냅샷
    bd = None
    try:
        bd = user.birth_date
    except Exception:
        pass
    if not bd and user.birth_year:
        bd = date(user.birth_year, 1, 1)
    resp.applicant_name = user.name_enc
    resp.birth_date = bd
    resp.gender = user.gender
    # 있으면 스냅샷
    try:
        if getattr(user, "height_cm", None) is not None:
            resp.height_cm = float(user.height_cm)
        if getattr(user, "weight_kg", None) is not None:
            resp.weight_kg = float(user.weight_kg)
    except Exception:
        pass
    session.add(resp)
    session.commit()

    rtoken = signer.sign(str(resp.id)).decode("utf-8")
    # 새 설문 시작: 완료 쿠키 제거
    redirect = RedirectResponse(url=f"/survey/step/1?rtoken={rtoken}", status_code=303)
    redirect.delete_cookie("survey_completed")
    return redirect


@app.get("/survey/step/{step}", response_class=HTMLResponse)
def survey_step_get(request: Request, step: int, rtoken: str, acc: str | None = None,  _guard: None = Depends(ensure_not_completed)):
    if step < 1 or step > 3:
        return RedirectResponse(url="/survey/step/1", status_code=303)
    respondent_id = verify_token(rtoken)
    if respondent_id < 0:
        return RedirectResponse(url="/login", status_code=302)

    questions = get_questions_for_step(step)
    return templates.TemplateResponse("survey_page.html", {
        "request": request,
        "step": step,
        "questions": questions,
        "acc": acc or "{}",
        "rtoken": rtoken,
        "is_last": step == 3,
        "is_first": step == 1
    })


@app.post("/survey/step/{step}")
async def survey_step_post(request: Request, step: int,
                           acc: str = Form("{}"),
                           rtoken: str = Form(...)):
    respondent_id = verify_token(rtoken)
    if respondent_id < 0:
        return RedirectResponse(url="/login", status_code=302)

    form = await request.form()

    try:
        acc_data = json.loads(acc) if acc else {}
    except Exception:
        acc_data = {}

    for q in get_questions_for_step(step):
        key = f"q{q['id']}"
        if q["type"] == "checkbox":
            vals = form.getlist(key)
            if vals:
                acc_data[key] = vals
        else:
            val = form.get(key)
            if val:
                acc_data[key] = val

    acc_q = json.dumps(acc_data, ensure_ascii=False)
    if step < 3:
        return RedirectResponse(
            url=f"/survey/step/{step+1}?acc={acc_q}&rtoken={rtoken}",
            status_code=303
        )
    else:
        return RedirectResponse(
            url=f"/survey/finish?acc={acc_q}&rtoken={rtoken}",   # ← 제출 엔드포인트 변경
            status_code=303
        )


@app.get("/survey/finish")
def survey_finish(
    request: Request,
    acc: str,
    rtoken: str,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
):
    """문진 제출 처리 및 NHIS 검진 데이터 저장"""
    respondent_id = verify_token(rtoken)
    if respondent_id < 0:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "message": "세션이 만료되었습니다. 다시 시작해주세요."},
            status_code=401,
        )

    # === NHIS 최종 수집: 세션의 '작은 값'(picked_tmp) + nhis_audit 원문(raw_from_audit) ===
    try:
        import json
        from sqlalchemy import text as sa_text

        picked_tmp = (request.session or {}).get("nhis_latest") or {}

        # 1순위: 이번 세션의 callbackId로 감사로그에서 가장 최근 원문
        cbid = (request.session or {}).get("nhis_callback_id")
        raw_from_audit = None
        if cbid:
            row = session.exec(sa_text("""
                SELECT response_json
                  FROM nhis_audit
                 WHERE callback_id = :cbid
                 ORDER BY id DESC
                 LIMIT 1
            """).bindparams(cbid=cbid)).first()
            if row:
                raw_from_audit = row[0]

        # 2순위: respondent_id로 최근 감사로그
        if raw_from_audit is None:
            row2 = session.exec(sa_text("""
                SELECT response_json
                  FROM nhis_audit
                 WHERE respondent_id = :rid
                 ORDER BY id DESC
                 LIMIT 1
            """).bindparams(rid=respondent_id)).first()
            if row2:
                raw_from_audit = row2[0]

        # 표준값이 비어있고 원문이 있으면, 원문으로부터 '최근 1건' 표준화 생성
        if (not picked_tmp) and raw_from_audit:
            try:
                picked_tmp = pick_latest_general(raw_from_audit, mode="latest")
            except Exception:
                picked_tmp = {}

        # 이후 로직에서 사용할 이름으로 통일
        nhis_latest = picked_tmp or {}
        nhis_raw    = raw_from_audit or {}

    except Exception as e:
        logging.error("[NHIS][FINISH][ERR-prep] %r", e)
        nhis_latest, nhis_raw = {}, {}

    # ────────────────────────────────────────────────────────────────
    # 응답 폼(acc) 파싱
    try:
        acc_obj = json.loads(acc) if acc else {}
    except Exception:
        acc_obj = {}

    # 정규화된 답안 인덱스 구성
    answers_indices = []
    for q in sorted(ALL_QUESTIONS, key=lambda x: x["id"]):
        key = f"q{q['id']}"
        if q.get("type") == "checkbox":
            raw_list = acc_obj.get(key, [])
            if isinstance(raw_list, str):
                raw_list = [s.strip() for s in raw_list.split(",") if s.strip()]
            idx_list = []
            for v in raw_list:
                try:
                    i = int(v)
                    if i >= 1:
                        idx_list.append(i)
                except:
                    pass
            answers_indices.append(idx_list)
        else:
            v = acc_obj.get(key)
            try:
                i = int(v) if v not in (None, "") else None
                answers_indices.append(i if (i is None or i >= 1) else None)
            except:
                answers_indices.append(None)

    normalized_payload = {"answers_indices": answers_indices}
    if nhis_latest:
        normalized_payload["nhis"] = nhis_latest

    # 응답자/사용자 조회 (인적사항)
    resp = session.get(Respondent, respondent_id)
    user = session.get(User, resp.user_id) if resp and resp.user_id else None

    def calc_age(bd, ref_date):
        if not bd:
            return ""
        return ref_date.year - bd.year - ((ref_date.month, ref_date.day) < (bd.month, bd.day))

    today = now_kst().date()
    name = (resp.applicant_name if resp and resp.applicant_name else (user.name_enc if user and user.name_enc else "")) or ""
    bd = resp.birth_date if (resp and resp.birth_date) else (getattr(user, "birth_date", None) if user else None)
    age = calc_age(bd, today) if bd else ""
    gender = (resp.gender if resp and resp.gender else (user.gender if user and user.gender else "")) or ""
    height = (getattr(resp, "height_cm", None) if resp else None) or (getattr(user, "height_cm", None) if user else None)
    weight = (getattr(resp, "weight_kg", None) if resp else None) or (getattr(user, "weight_kg", None) if user else None)

    # SurveyResponse 생성 (이번 제출 레코드에 NHIS를 '직접' 저장)
    sr = SurveyResponse(
        respondent_id=respondent_id,
        answers_json=json.dumps(normalized_payload, ensure_ascii=False),
        score=None,
        submitted_at=now_kst(),
        nhis_json=nhis_latest,   # 표준화된 작은 dict
        nhis_raw=nhis_raw,       # 원문 전체
    )
    session.add(sr)
    if resp:
        resp.status = "submitted"
        session.add(resp)
    session.commit()
    session.refresh(sr)

    # 일련번호 채번
    if resp and resp.serial_no is None:
        next_val = session.exec(sa_text("SELECT nextval('respondent_serial_no_seq')")).one()[0]
        resp.serial_no = next_val
        session.add(resp)
        session.commit()
        session.refresh(resp)

    # 건강검진 데이터 임시로그
    try:
        print("[NHIS][SAVE] latest_keys=", list((nhis_latest or {}).keys()) if isinstance(nhis_latest, dict) else type(nhis_latest))
    except Exception as e:
        print("[NHIS][SAVE][WARN1]", repr(e))
    ey = (nhis_latest.get("EXAMYEAR") or nhis_latest.get("GUNYEAR") or nhis_latest.get("exam_year"))
    print("[NHIS][SAVE] exam_year=", ey, "| has_latest:", bool(nhis_latest))


    # ── 알림메일 비동기 발송 ───────────────────────────────
    try:
        applicant_name = (
            (resp.applicant_name or (user.name_enc if user else ""))
            if resp else ""
        )
        created_at_kst_str = (
            to_kst(resp.created_at).strftime("%Y-%m-%d %H:%M")
            if resp and resp.created_at else now_kst().strftime("%Y-%m-%d %H:%M")
        )
        serial_no_val = resp.serial_no or 0
        background_tasks.add_task(
            send_submission_email,
            serial_no_val,
            applicant_name,
            created_at_kst_str,
        )
    except Exception as e:
        print("[EMAIL][ERR]", repr(e))

    # 세션 정리 (작은 dict만 보관했었다면 이제 비워도 OK)
    request.session.pop("nhis_latest", None)
    request.session.pop("nhis_raw", None)

    response = RedirectResponse(url="/portal", status_code=302)
    response.set_cookie(
        "survey_completed", "1",
        max_age=60*60*24*7,
        httponly=True,
        samesite="Lax",
        secure=bool(int(os.getenv("SECURE_COOKIE","1"))),
    )
    return response




@app.post("/admin/responses/export.xlsx")
async def admin_export_xlsx(
    request: Request,
    response: Response,
    ids: str = Form(...),
    session: Session = Depends(get_session),
    _auth: None = Depends(admin_required),
):
    # ---------------------------
    # 0) 유틸
    # ---------------------------
    def get_nhis_dict(v):
        """jsonb(dict) 또는 JSON 문자열 모두 수용"""
        if not v:
            return {}
        if isinstance(v, dict):
            return v
        try:
            return json.loads(v)
        except Exception:
            return {}

    def nhis_extract_all(nhis_json: dict | None, nhis_raw: dict | None) -> dict:
        """
        - 최우선: nhis_json(표준화된 최근 1건)에서 바로 꺼낸다.
        - 보조: nhis_raw.data.INCOMELIST가 있으면, 최신년도 1건으로 보강.
        - 결과: 엑셀 병합용 dict 리턴 (검진년도만 사용, 기관은 공란)
        - 반환 키(영문): exam_year,height,weight,bmi,bp,vision,hearing,hemoglobin,fbs,tc,hdl,ldl,tg,
                        gfr,creatinine,ast,alt,ggt,urine_protein,chest,judgment
        """
        nj = nhis_json or {}

        def _year_of(d: dict) -> str:
            if not isinstance(d, dict):
                return ""
            for k in ("EXAMYEAR", "GUNYEAR", "YEAR", "YY"):
                v = d.get(k)
                if isinstance(v, int):
                    return str(v)
                if isinstance(v, str) and v.isdigit():
                    return v
            for k in ("EXAMDATE", "EXAM_DATE", "검진일자", "exam_date", "GUNDATE"):
                v = d.get(k)
                if isinstance(v, str) and len(v) >= 4 and v[:4].isdigit():
                    return v[:4]
            return ""

        # 원본에서 최신 1건 추출
        raw_item = None
        if nhis_raw and isinstance(nhis_raw, dict):
            data = nhis_raw.get("data") or {}
            income = data.get("INCOMELIST") or []
            if isinstance(income, list) and income:
                def _yr(row: dict) -> int:
                    y = _year_of(row)
                    return int(y) if (y and y.isdigit()) else -1
                try:
                    income_sorted = sorted(income, key=_yr, reverse=True)
                except Exception:
                    income_sorted = list(income)
                raw_item = income_sorted[0] if income_sorted else None

        # 값 선택: 표준값 우선 → 없으면 원본 보강
        def pick(*keys: str) -> str:
            for k in keys:
                v = nj.get(k)
                if v not in (None, "", []):
                    return str(v)
            if raw_item:
                for k in keys:
                    v = raw_item.get(k)
                    if v not in (None, "", []):
                        return str(v)
            return ""

        exam_year = _year_of(nj) or _year_of(raw_item or {})

        out = {
            "exam_year":      exam_year,            # 검진년도(연도만)
            "height":         pick("HEIGHT"),
            "weight":         pick("WEIGHT"),
            "bmi":            pick("BODYMASS", "BMI"),
            "bp":             pick("BLOODPRESS"),
            "vision":         pick("SIGHT"),
            "hearing":        pick("HEARING"),
            "hemoglobin":     pick("HEMOGLOBIN"),
            "fbs":            pick("BLOODSUGAR"),   # 공복혈당
            "tc":             pick("TOTCHOLESTEROL"),
            "hdl":            pick("HDLCHOLESTEROL", "HDL_CHOLESTEROL"),
            "ldl":            pick("LDLCHOLESTEROL", "LDL_CHOLESTEROL"),
            "tg":             pick("TRIGLYCERIDE"),
            "gfr":            pick("GFR"),
            "creatinine":     pick("SERUMCREATININE"),
            "ast":            pick("SGOT"),
            "alt":            pick("SGPT"),
            "ggt":            pick("YGPT", "GAMMAGTP"),
            "urine_protein":  pick("YODANBAK"),
            "chest":          pick("CHESTTROUBLE"),
            "judgment":       pick("JUDGMENT"),
        }
        return out

    def fmt(v):
        if isinstance(v, list):
            return ",".join(str(x) for x in v)
        return "" if v is None else str(v)

    def calc_age(bd, ref_date):
        if not bd:
            return ""
        return ref_date.year - bd.year - ((ref_date.month, ref_date.day) < (bd.month, bd.day))

    # ---------------------------
    # 1) ids 파싱
    # ---------------------------
    print("export.xlsx ids raw:", repr(ids))
    id_list = [int(x) for x in (ids or "").split(",") if x.strip().isdigit()]
    if not id_list:
        return RedirectResponse(url="/admin/responses", status_code=303)

    # ---------------------------
    # 2) 질문 준비
    # ---------------------------
    def q_title(q: dict):
        return (q.get("title") or q.get("text") or q.get("label")
                or q.get("question") or q.get("prompt") or q.get("name")
                or f"Q{q.get('id','')}")
    questions_sorted = sorted(ALL_QUESTIONS, key=lambda x: x["id"])
    questions = [q_title(q) for q in questions_sorted]

    def extract_answers(payload: dict, questions_sorted: list[dict]) -> list:
        def to_num(v):
            if v is None or v == "": return ""
            if isinstance(v, list):
                return [int(x) for x in v if str(x).isdigit()]
            if isinstance(v, str) and "," in v:
                return [int(x) for x in v.split(",") if x.strip().isdigit()]
            return int(v) if str(v).isdigit() else ""

        ai = payload.get("answers_indices")
        if isinstance(ai, list) and len(ai) == len(questions_sorted):
            candidates = [payload]
            for k in ("answers", "acc", "data"):
                if isinstance(payload.get(k), dict):
                    candidates.append(payload[k])
            def fallback(qid):
                for cand in candidates:
                    v = cand.get(f"q{qid}")
                    num = to_num(v)
                    if num != "": return num
                return ""
            out = []
            for i, q in enumerate(questions_sorted):
                v = ai[i]
                empty = (v is None) or (v == "") or (isinstance(v, list) and len(v) == 0)
                out.append(fallback(q["id"]) if empty else v)
            return out

        candidates = [payload]
        for k in ("answers", "acc", "data"):
            if isinstance(payload.get(k), dict):
                candidates.append(payload[k])
        for cand in candidates:
            out, ok = [], 0
            for q in questions_sorted:
                num = to_num(cand.get(f"q{q['id']}"))
                out.append(num)
                if num != "": ok += 1
            if ok > 0:
                return out
        return ["" for _ in questions_sorted]

    # ---------------------------
    # 3) 워크북/시트 + 헤더
    # ---------------------------
    wb = Workbook()
    ws = wb.active
    ws.title = "문진결과"

    today = now_kst().date()

    fixed_headers = ["no.", "신청번호", "이름", "생년월일", "나이(만)", "성별"]
    nhis_headers  = [
        "검진년도","신장(NHIS)","체중(NHIS)","BMI",
        "혈압","시력","청력","혈색소","공복혈당",
        "총콜레스테롤","HDL","LDL","중성지방",
        "GFR","크레아티닌","AST","ALT","GGT",
        "요단백","흉부소견","종합판정",
    ]
    ws.append(fixed_headers + nhis_headers + questions)

    # ---------------------------
    # 4) 데이터 행
    # ---------------------------
    for idx, rid in enumerate(id_list, start=1):
        sr = session.get(SurveyResponse, rid)
        if not sr:
            continue

        resp = session.get(Respondent, sr.respondent_id) if sr.respondent_id else None
        user = session.get(User, resp.user_id) if resp and resp.user_id else None

        # 인적사항
        name = (resp.applicant_name if resp and resp.applicant_name else (user.name_enc if user and user.name_enc else "")) or ""
        bd = resp.birth_date if (resp and resp.birth_date) else (getattr(user, "birth_date", None) if user else None)
        age = calc_age(bd, today) if bd else ""
        gender = (resp.gender if resp and resp.gender else (user.gender if user and user.gender else "")) or ""
        height = (getattr(resp, "height_cm", None) if resp else None) or (getattr(user, "height_cm", None) if user else None)
        weight = (getattr(resp, "weight_kg", None) if resp else None) or (getattr(user, "weight_kg", None) if user else None)
        serial_no = resp.serial_no if (resp and resp.serial_no is not None) else ""

        # 답 추출
        try:
            payload = json.loads(sr.answers_json) if sr.answers_json else {}
        except Exception as e:
            print("export.xlsx: bad answers_json for rid", rid, "err:", repr(e))
            payload = {}
        answers = extract_answers(payload, questions_sorted)

        # NHIS 표준+백업 추출 (표준: nhis_json, 원본: nhis_raw)
        nhis_std = nhis_extract_all(
            get_nhis_dict(sr.nhis_json),
            get_nhis_dict(sr.nhis_raw),
        )

        row = [
            idx,
            serial_no,
            name,
            (bd.isoformat() if bd else ""),
            age,
            gender,

            # NHIS 열들 (검진년도만, 기관 없음)
            nhis_std.get("exam_year", ""),
            nhis_std.get("height", ""),
            nhis_std.get("weight", ""),
            nhis_std.get("bmi", ""),
            nhis_std.get("bp", ""),
            nhis_std.get("vision", ""),
            nhis_std.get("hearing", ""),
            nhis_std.get("hemoglobin", ""),
            nhis_std.get("fbs", ""),
            nhis_std.get("tc", ""),
            nhis_std.get("hdl", ""),
            nhis_std.get("ldl", ""),
            nhis_std.get("tg", ""),
            nhis_std.get("gfr", ""),
            nhis_std.get("creatinine", ""),
            nhis_std.get("ast", ""),
            nhis_std.get("alt", ""),
            nhis_std.get("ggt", ""),
            nhis_std.get("urine_protein", ""),
            nhis_std.get("chest", ""),
            nhis_std.get("judgment", ""),
        ] + [fmt(v) for v in answers]

        ws.append(row)

    # ---------------------------
    # 5) 바이너리 응답
    # ---------------------------
    mem = BytesIO()
    wb.save(mem)
    mem.seek(0)
    return StreamingResponse(
        mem,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="responses.xlsx"'}
    )



@app.get("/nhis/progress")
def nhis_progress(request: Request, mode: str = "", callbackId: str = "", next: str = "/info"):
    # mode = 'callback' | 'direct'
    return templates.TemplateResponse(
        "nhis_progress.html",
        {
            "request": request,
            "mode": mode,
            "callbackId": callbackId,
            "next_url": next or "/info",
        },
    )



from fastapi import Body, Request, HTTPException

# ===========================================
# DataHub 간편인증 Step1: 시작
# ===========================================

@app.post("/api/dh/simple/start")
async def dh_simple_start(
    request: Request,
    session: Session = Depends(get_session),   # ★ 추가: 감사로그에 씁니다
):
    payload = await request.json()

    loginOption  = str(payload.get("loginOption", "")).strip()
    telecom      = str(payload.get("telecom", "")).strip()
    userName     = str(payload.get("userName", "")).strip()
    hpNumber     = str(payload.get("hpNumber", "")).strip()
    juminOrBirth = re.sub(r"[^0-9]", "", str(payload.get("juminOrBirth") or payload.get("birth") or ""))

    # 8자리 YYYYMMDD로 강제
    if len(juminOrBirth) >= 8:
        juminOrBirth = juminOrBirth[-8:]

    # ✅ LOGINOPTION 허용값: 0~7
    allowed = {"0","1","2","3","4","5","6","7"}

    #필수 입력값 점검
    missing = []
    if not loginOption or loginOption not in allowed:  missing.append("loginOption(0~7)")
    if not userName:                                   missing.append("userName")
    if not hpNumber:                                   missing.append("hpNumber")
    if not juminOrBirth:                               missing.append("birth(YYYYMMDD)")
    elif not re.fullmatch(r"\d{8}", juminOrBirth):     missing.append("birth(YYYYMMDD 8자리)") 
    if loginOption == "3" and not telecom:
        missing.append("telecom(PASS: 1~6, SKT|KT|LGU+ 등)")

    if missing:
        logging.warning("[DH-START][VALIDATION] missing=%s", missing)
        return JSONResponse({"result":"FAIL","message":"필수 입력 누락","missing":missing}, status_code=400)

    # 새 트랜잭션 시작: 낡은 콜백/상태 제거  ← ★ 이 줄부터 추가
    for k in ("nhis_callback_id", "nhis_callback_type", "dh_callback"):
        request.session.pop(k, None)
    
    # hpNumber: 숫자만, 하이픈 없음
    hpNumber = re.sub(r'[^0-9]', '', hpNumber or '')

    # 콜백형 강제 규격 (LOGINOPTION 0~7 지원)
    dh_body = {
        "LOGINOPTION": loginOption,
        "HPNUMBER":    hpNumber,
        "USERNAME":    userName,
        "JUMIN":       juminOrBirth,
    }
    if loginOption == "3" and telecom:
        dh_body["TELECOMGUBUN"] = telecom  # 1~6
    
    # (선택) 민감값 마스킹 로그
    _safe = {**dh_body, "HPNUMBER": _mask_phone(dh_body.get("HPNUMBER","")), "JUMIN": _mask_birth(dh_body.get("JUMIN",""))}
    logging.debug("[DH-START][BODY]%s", _safe)
    
    #성별 세션 보관
    gender = str(payload.get("gender","")).strip() 
    request.session["nhis_gender"] = gender if gender in ("남","여") else ""
    
    #인적정보 세션 보관
    request.session["nhis_start_payload"] = dh_body

    # 1) 시작 호출
    rsp = DATAHUB.simple_auth_start(
        login_option=dh_body["LOGINOPTION"],      # "0"~"7"
        user_name=dh_body["USERNAME"],
        hp_number=dh_body["HPNUMBER"],
        jumin_or_birth=dh_body["JUMIN"],
        telecom_gubun=dh_body.get("TELECOMGUBUN"),
    )


    try:
        stmt = sa_text("""
            INSERT INTO nhis_audit (respondent_id, callback_id, request_json, response_json)
            VALUES (:rid, :cbid, :req, :res)
        """).bindparams(
            rid=None,  # 알 수 없으면 None
            cbid=(rsp.get("data") or {}).get("callbackId") or "",
            req=json.dumps({"step":"start","body":dh_body}, ensure_ascii=False),
            res=json.dumps(rsp or {}, ensure_ascii=False),
        )
        session.exec(stmt)
        session.commit()
    except Exception as e:
        print("[NHIS][AUDIT][ERR][start]", repr(e))


    err  = str(rsp.get("errCode","")).strip()
    data = rsp.get("data") or {}
    cbid = (data.get("callbackId") or "").strip()

    if err == "0001" and cbid:
        request.session["nhis_callback_id"]   = cbid
        request.session["nhis_callback_type"] = "SIMPLE"
        return JSONResponse({"errCode":"0001","message":"NEED_CALLBACK","data":{"callbackId": cbid}}, status_code=200)

    # ★ 여기서부터는 전부 실패로 처리 (0000이라도 콜백 없으면 실패)
    msg = (rsp.get("errMsg") or "간편인증 시작 실패").strip()
    return JSONResponse({"errCode": err or "9999", "message": msg, "data": data}, status_code=200)


# ===========================================
# DataHub 간편인증 Step2: 완료(captcha)
# ===========================================


@app.post("/api/dh/simple/complete")
async def dh_simple_complete(
    request: Request,
    session: Session = Depends(get_session),
):
    """
    콜백형 완료:
      1) /scrap/captcha (Step2) 1회 호출 (callbackResponse* 키 포함, 예외/타임아웃 안전 처리)
      2) Step2 응답에 INCOMELIST가 있으면 즉시 채택하여 종료
      3) 없으면 같은 callbackId로 /scrap/common/...Simple 폴링 재조회 (light만)
      4) 최대 120초 폴링, 미완료면 202
    """

    payload = await request.json()

    # 0) 세션(or 요청)에서 콜백 값 복구
    cbid = (request.session or {}).get("nhis_callback_id") or str(payload.get("callbackId") or "")
    cbtp = (request.session or {}).get("nhis_callback_type") or str(payload.get("callbackType") or "SIMPLE")

    # callbackid/type 확인 디버그 로그
    logging.debug(
        "[DH-COMPLETE][DEBUG] callbackId sources => session:%s | payload:%s | final:%s",
        (request.session or {}).get("nhis_callback_id"),
        payload.get("callbackId"),
        cbid,
    )
    logging.debug(
        "[DH-COMPLETE][DEBUG] callbackType sources => session:%s | payload:%s | final:%s",
        (request.session or {}).get("nhis_callback_type"),
        payload.get("callbackType"),
        cbtp,
    )

    # 해당 시점 rtoken 유무 확인 디버그 로그
    try:
        rtok = (request.query_params.get("rtoken") or request.cookies.get("rtoken") or "")
        rid_dbg = verify_token(rtok) if rtok else None
        logging.debug("[RTOKEN][DBG][complete] raw=%s | rid=%s", ("yes" if rtok else "no"), (rid_dbg if rid_dbg else "None"))
    except Exception as e:
        logging.debug("[RTOKEN][DBG][complete][ERR] %r", e)


    # 0-1) 최소 검증 (DataHub 호출 낭비 방지)
    if not cbid or not cbtp:
        logging.warning("[DH-COMPLETE][VALIDATION] missing=%s", [k for k, v in {"callbackId": cbid, "callbackType": cbtp}.items() if not v])
        return JSONResponse({"result": "FAIL", "message": "필수 입력 누락", "missing": ["callbackId", "callbackType"]}, status_code=400)

    # 1) Step2: /scrap/captcha (키는 모두 포함; 값은 비어도 OK) — 예외/타임아웃 안전 처리
    try:
        step2_res = DATAHUB.simple_auth_complete(
            callback_id=cbid,
            callback_type=cbtp,
            callbackResponse=str(payload.get("callbackResponse") or ""),
            callbackResponse1=str(payload.get("callbackResponse1") or ""),
            callbackResponse2=str(payload.get("callbackResponse2") or ""),
            retry=str(payload.get("retry") or ""),
        )
    except Exception as e:
        # 네트워크/타임아웃 등 오류가 나더라도 폴링으로 진행
        print("[DH-COMPLETE][captcha][ERR]", repr(e))
        step2_res = {"errCode": "TIMEOUT", "result": "FAIL", "data": {}}

    # 1-1) 감사로그: Step2 요청/응답 저장
    try:
        resp_id = None
        try:
            tok = (request.query_params.get("rtoken") or request.cookies.get("rtoken") or "")
            rid = verify_token(tok) if tok else -1
            if rid > 0:
                resp_id = rid
        except Exception:
            pass

        stmt = sa_text("""
            INSERT INTO nhis_audit (respondent_id, callback_id, request_json, response_json)
            VALUES (:rid, :cbid, :req, :res)
        """).bindparams(
            rid=resp_id,
            cbid=cbid,
            req=json.dumps({"step": "captcha", "body": {
                "callbackId": cbid, "callbackType": cbtp,
                "callbackResponse": "", "callbackResponse1": "", "callbackResponse2": "", "retry": ""
            }}, ensure_ascii=False),
            res=json.dumps(step2_res or {}, ensure_ascii=False),
        )
        session.exec(stmt)
        session.commit()
    except Exception as e:
        print("[NHIS][AUDIT][ERR][captcha]", repr(e))

    # 1-2) Step2 응답 자체에 INCOMELIST가 있으면 즉시 채택
    try:
        step2_data = (step2_res or {}).get("data") or {}
        income2 = step2_data.get("INCOMELIST") or []
        if isinstance(income2, list) and len(income2) > 0:
            want_all = (request.query_params.get("all") or "").lower() in ("1", "true", "yes")
            picked = pick_latest_general(step2_res, mode=("all" if want_all else "latest"))
            request.session["nhis_latest"] = picked if isinstance(picked, dict) else {}

            # ★ NHIS결과 DB 저장 (엑셀 병합용)
            try:
                picked_one = pick_latest_general(step2_res, mode="latest")
                _save_nhis_to_db(session, request, picked_one, step2_res)
                # 저장이 스킵될 수도 있으니 세션에도 임시 보관
                request.session["nhis_latest"] = picked_one or {}
                #request.session["nhis_raw"]    = step2_res or {}  #쿠키 터짐 저장하지 않음
            except Exception as e:
                print("[NHIS][DB][WARN][captcha-save]", repr(e))


            # --- 성공 직전 User 인적정보 업데이트(이름/성별/생년월일) ---
            try:
                from datetime import date
                auth_cookie = request.cookies.get(AUTH_COOKIE_NAME)
                user_id = verify_user(auth_cookie) if auth_cookie else -1
                if user_id and user_id > 0:
                    user = session.get(User, user_id)
                    if user:
                        sp = (request.session or {}).get("nhis_start_payload") or {}
                        nm = str(sp.get("USERNAME") or "").strip()
                        bd8 = str(sp.get("JUMIN") or "").strip()
                        gd = (request.session or {}).get("nhis_gender") or ""
                        # 생년월일 파싱(YYYYMMDD)
                        bd_date = None
                        if len(bd8) == 8 and bd8.isdigit():
                            bd_date = date(int(bd8[0:4]), int(bd8[4:6]), int(bd8[6:8]))
                        # 저장(있을 때만 덮어씀)
                        if nm: user.name_enc = nm
                        if gd in ("남","여"): user.gender = gd
                        if bd_date: user.birth_date = bd_date; user.birth_year = bd_date.year
                        session.add(user); session.commit()
            except Exception as _e:
                logging.debug("[NHIS][USER-SNAPSHOT][WARN] %r", _e)

            return JSONResponse({"ok": True, "errCode": "0000", "message": "OK", "data": picked}, status_code=200)
  
    except Exception as e:
        print("[DH-COMPLETE][WARN][captcha-pick]", repr(e))

    # 2) 결과 재조회 폴링 (light 1회 확인 후 → full만)
    max_wait_sec = NHIS_POLL_MAX_SEC
    deadline = time.time() + max_wait_sec
    attempt = 0

    # 시작 단계 값 복구
    SP = (request.session or {}).get("nhis_start_payload") or {}
    loginOption  = str(SP.get("LOGINOPTION", "")).strip()
    userName     = str(SP.get("USERNAME", "")).strip()
    hpNumber     = str(SP.get("HPNUMBER", "")).strip()
    juminVal     = str(SP.get("JUMIN", "")).strip() or str(SP.get("JUMINNUM", "")).strip()
    telecomGubun = str(SP.get("TELECOMGUBUN", "")).strip() if loginOption == "3" else None

    want_all = ((request.query_params.get("all") or "").lower() in ("1", "true", "yes"))

    did_full = False
    while time.time() < deadline:
        attempt += 1
        try:
            if attempt <= NHIS_MAX_LIGHT_FETCH:
                # ➊ light 1회만
                fetch_body = {"CALLBACKID": cbid, "CALLBACKTYPE": cbtp}
                rsp2 = DATAHUB.medical_checkup_simple(fetch_body)
                kind = "light"
            else:
                # ➋ 이후는 계속 full
                rsp2 = DATAHUB.medical_checkup_simple_with_identity(
                    callback_id=cbid,
                    callback_type=cbtp,
                    login_option=loginOption,
                    user_name=userName,
                    hp_number=hpNumber,
                    jumin_or_birth=juminVal,
                    telecom_gubun=telecomGubun
                )
                kind = "full"
                did_full = True
        except Exception as e:
            logging.warning("[DH-COMPLETE][FETCH][ERR] %r", e)
            time.sleep(NHIS_FETCH_INTERVAL)
            continue

        # 감사로그(요약)
        try:
            stmt = sa_text("""
                INSERT INTO nhis_audit (respondent_id, callback_id, request_json, response_json)
                VALUES (:rid, :cbid, :req, :res)
            """).bindparams(
                rid=None, cbid=cbid,
                req=json.dumps({"step": "fetch", "kind": kind}, ensure_ascii=False),
                res=json.dumps(rsp2 or {}, ensure_ascii=False),
            )
            session.exec(stmt); session.commit()
        except Exception as e:
            logging.warning("[NHIS][AUDIT][ERR][fetch-log] %r", e)

        err2   = str((rsp2 or {}).get("errCode") or "")
        data2  = (rsp2 or {}).get("data") or {}
        income = data2.get("INCOMELIST") or []

        # 내부 에러 힌트만 DEBUG로
        inner_ecode  = data2.get("ECODE")
        inner_errmsg = data2.get("ERRMSG")
        if inner_ecode and inner_ecode != "0000":
            logging.debug("[DH-COMPLETE][FETCH][INNER] ecode=%s msg=%s", inner_ecode, inner_errmsg)

        logging.info("[DH-COMPLETE][FETCH] attempt=%s kind=%s err=%s income_len=%s",
                     attempt, kind, err2, (len(income) if isinstance(income, list) else "NA"))

        if err2 == "0000" and isinstance(income, list) and len(income) > 0:
            picked = pick_latest_general(rsp2, mode=("all" if want_all else "latest"))
            request.session["nhis_latest"] = picked if isinstance(picked, dict) else {}
            # DB 저장 (엑셀 병합용)
            try:
                picked_one = pick_latest_general(rsp2, mode="latest")
                _save_nhis_to_db(session, request, picked_one, rsp2)
                request.session["nhis_latest"] = picked_one or {}
            except Exception as e:
                logging.warning("[NHIS][DB][WARN][fetch-save] %r", e)
                
            # --- 성공 직전 User 인적정보 업데이트(이름/성별/생년월일) ---
            try:
                from datetime import date
                auth_cookie = request.cookies.get(AUTH_COOKIE_NAME)
                user_id = verify_user(auth_cookie) if auth_cookie else -1
                if user_id and user_id > 0:
                    user = session.get(User, user_id)
                    if user:
                        sp = (request.session or {}).get("nhis_start_payload") or {}
                        nm = str(sp.get("USERNAME") or "").strip()
                        bd8 = str(sp.get("JUMIN") or "").strip()
                        gd = (request.session or {}).get("nhis_gender") or ""
                        # 생년월일 파싱(YYYYMMDD)
                        bd_date = None
                        if len(bd8) == 8 and bd8.isdigit():
                            bd_date = date(int(bd8[0:4]), int(bd8[4:6]), int(bd8[6:8]))
                        # 저장(있을 때만 덮어씀)
                        if nm: user.name_enc = nm
                        if gd in ("남","여"): user.gender = gd
                        if bd_date: user.birth_date = bd_date; user.birth_year = bd_date.year
                        session.add(user); session.commit()
            except Exception as _e:
                logging.debug("[NHIS][USER-SNAPSHOT][WARN] %r", _e)
            
            return JSONResponse({"ok": True, "errCode": "0000", "message": "OK", "data": picked}, status_code=200)

        time.sleep(NHIS_FETCH_INTERVAL)


    return JSONResponse({"ok": False, "errCode": "2020", "message": "아직 인증이 완료되지 않았거나 데이터가 준비되지 않았습니다."}, status_code=202)




# ---- 유틸: 최신 1건 선택 (연/월/일 기준) ----
def pick_latest_one(data: dict) -> dict:
    """
    data.INCOMELIST[] 중 가장 최근(연/월/일) 1건만 골라 요약해 리턴.
    형식은 가이드의 필드명을 그대로 사용.
    """
    items = (data or {}).get("INCOMELIST") or []
    best = None
    best_key = None
    for it in items:
        year = (it.get("GUNYEAR") or "").strip()
        date = (it.get("GUNDATE") or "").strip()  # 'MM/DD' 형태 예시
        # 키를 YYYYMMDD 정수로 만들어 비교
        try:
            mm, dd = (date.split("/") + ["0", "0"])[:2]
            key = int(f"{int(year):04d}{int(mm):02d}{int(dd):02d}")
        except Exception:
            # 연도만 있는 항목은 월/일 0으로
            try:
                key = int(f"{int(year):04d}0000")
            except Exception:
                key = -1
        if best is None or key > best_key:
            best, best_key = it, key
    return best or {}



# ===========================================
# DataHub 인증서 방식(필요 시): 건강검진 결과 조회
# ===========================================
@app.post("/api/dh/nhis/result")
def dh_nhis_result(payload: dict = Body(...), request: Request = None):
    """
    요청 JSON:
    {
      "jumin": "주민번호13자리",
      "certName": "cn=...,ou=...,o=...,c=kr",
      "certPwd": "인증서비번",
      "derB64": "<DER base64>",
      "keyB64": "<KEY base64>"
    }
    """
    try:
        jumin   = (payload.get("jumin") or "").strip()
        cname   = (payload.get("certName") or "").strip()
        cpwd    = (payload.get("certPwd") or "").strip()
        der_b64 = (payload.get("derB64") or "").strip()
        key_b64 = (payload.get("keyB64") or "").strip()
        if not (jumin and cname and cpwd and der_b64 and key_b64):
            raise HTTPException(400, "jumin/certName/certPwd/derB64/keyB64는 모두 필수입니다.")
        rsp = DATAHUB.nhis_medical_checkup(jumin, cname, cpwd, der_b64, key_b64)
        try:
            picked = pick_latest_general(rsp)
            if request is not None:
                request.session["nhis_latest"] = picked
        except Exception:
            pass
        return rsp
    except DatahubError as e:
        raise HTTPException(502, f"DATAHUB error: {e}")
    except Exception as e:
        raise HTTPException(500, f"Internal error: {e}")


#임시 디버그 라우트, 로그. 운영 시 삭제
from fastapi.responses import JSONResponse

@app.get("/debug/datahub-selftest")
def debug_datahub_selftest():
    import os, base64, hashlib
    from app.vendors.datahub_client import encrypt_field, _get_key_iv, _get_text_encoding

    plain  = (os.getenv("DATAHUB_SELFTEST_PLAIN", "") or "").strip()
    expect = (os.getenv("DATAHUB_SELFTEST_EXPECT", "") or "").strip()
    if not plain or not expect:
        return JSONResponse({"error":"set DATAHUB_SELFTEST_PLAIN & EXPECT"}, status_code=400)

    # 현재 FORCE 설정/키/IV 요약
    kb  = int(os.getenv("DATAHUB_FORCE_KEY_BITS", "256") or "256")
    ivm = (os.getenv("DATAHUB_FORCE_IV_MODE", "ENV") or "ENV").upper()
    ksh = (os.getenv("DATAHUB_FORCE_KEY_SHAPE", "right") or "right").lower()
    enc = _get_text_encoding()

    key, iv = _get_key_iv()
    key_sha = hashlib.sha256(key).hexdigest()[:16]
    iv_hex  = iv.hex()[:16]

    got = encrypt_field(plain)
    return JSONResponse({
        "env": {
            "DATAHUB_TEXT_ENCODING": enc,
            "DATAHUB_FORCE_KEY_BITS": kb,
            "DATAHUB_FORCE_IV_MODE": ivm,
            "DATAHUB_FORCE_KEY_SHAPE": ksh
        },
        "kiv": {
            "key_bits": kb,
            "key_len": len(key),
            "key_sha256_head": key_sha,
            "iv_len": len(iv),
            "iv_head_hex": iv_hex
        },
        "plain": plain,
        "expect": expect,
        "got": got,
        "match": (got == expect)
    })



#임시 디버그 라우트, 로그. 운영 시 삭제
@app.get("/debug/whoami")
def debug_whoami():
    return JSONResponse({
        "main_file": __file__,
        "cwd": os.getcwd(),
        "env": {
            "APP_ENV": os.getenv("APP_ENV"),
            "DATAHUB_SELFTEST": os.getenv("DATAHUB_SELFTEST"),
            "DATAHUB_SELFTEST_PLAIN_set": bool(os.getenv("DATAHUB_SELFTEST_PLAIN")),
            "DATAHUB_SELFTEST_EXPECT_set": bool(os.getenv("DATAHUB_SELFTEST_EXPECT")),
        }
    })

#인코딩 테스트
@app.get("/debug/datahub-finder")
def debug_datahub_finder():
    """
    API 호출 없이, ENV에 있는 PlainData/EncData 쌍을 기준으로
    - 평문 인코딩(utf-8/cp949)
    - 키 비트(128/256)
    - IV 모드(ENV/ZERO)
    - EncKey/IV 해석(., urlsafe, pad, hex, raw 등 변형)
    - 키/IV 길이 보정 방식(left/right pad)
    - (보너스) 키 유도(SHA-256/MD5) 방식
    조합을 브루트포스로 시도해 'expect'와 일치하는 암호문을 찾는다.
    """
    import os, base64, hashlib
    from Crypto.Cipher import AES

    def pkcs7_pad(b: bytes, block=16) -> bytes:
        n = block - (len(b) % block)
        return b + bytes([n]) * n

    plain  = (os.getenv("DATAHUB_SELFTEST_PLAIN", "") or "").strip()
    expect = (os.getenv("DATAHUB_SELFTEST_EXPECT", "") or "").strip()
    enc_spec = (os.getenv("DATAHUB_ENC_SPEC", "") or "").strip().upper()
    enc_key_raw = (os.getenv("DATAHUB_ENC_KEY_B64", "") or "").strip()
    enc_iv_raw  = (os.getenv("DATAHUB_ENC_IV_B64", "") or "").strip()

    if not plain or not expect or not enc_key_raw:
        return JSONResponse({"error":"set DATAHUB_SELFTEST_PLAIN/EXPECT and ENC_KEY/IV"}, status_code=400)

    # 1) key/iv 해석 후보 생성
    def pad4(s: str) -> str:
        return s + ("=" * ((4 - len(s) % 4) % 4))

    def b64_try(s: str):
        cands = [
            s,
            s.replace("-", "+").replace("_", "/"),
            s.replace(".", ""),           # dot 제거
            s.replace(".", "+"),
            s.replace(".", "/"),
        ]
        seen = set()
        out = []
        for c in cands:
            c = pad4(c)
            if c in seen: continue
            seen.add(c)
            try:
                out.append(base64.b64decode(c))
            except Exception:
                pass
        return out

    def hex_try(s: str):
        try:
            return [bytes.fromhex(s)]
        except Exception:
            return []

    def raw_try(s: str):
        return [s.encode("utf-8")]

    key_bytes_cands = b64_try(enc_key_raw) + hex_try(enc_key_raw) + raw_try(enc_key_raw)
    iv_bytes_cands  = b64_try(enc_iv_raw)  + hex_try(enc_iv_raw)  + raw_try(enc_iv_raw)
    if not iv_bytes_cands:
        iv_bytes_cands = [b"\x00"*16]  # IV 미제공 대비

    # 2) 길이 보정/유도 함수들
    def shape_key(k: bytes, bits: int, mode: str) -> bytes:
        need = 32 if bits == 256 else 16
        if mode == "right":
            return (k[:need]).ljust(need, b"\x00")
        elif mode == "left":
            return (k[-need:]).rjust(need, b"\x00")
        elif mode == "sha256":
            d = hashlib.sha256(k).digest()
            return d[:need]
        elif mode == "md5":
            d = hashlib.md5(k).digest()
            # 128비트만 직접 충족, 256은 md5 두 번 접합
            return (d if need==16 else (d+hashlib.md5(d).digest()))[:need]
        else:
            return (k[:need]).ljust(need, b"\x00")

    def shape_iv(iv: bytes, mode: str) -> bytes:
        if mode == "ZERO":
            return b"\x00"*16
        v = iv[:16]
        if len(v) < 16: v = v.ljust(16, b"\x00")
        return v

    # 3) 시도할 조합들
    encodings   = ["utf-8", "cp949"]
    key_bits    = [256, 128]
    key_shapes  = ["right", "left", "sha256", "md5"]  # 키 길이/유도 방식
    iv_modes    = ["ENV", "ZERO"]
    iv_shapes   = ["keep"]  # 필요 시 확장
    tried = 0
    matches = []

    for enc in encodings:
        p = plain.encode(enc, errors="strict")
        for kb in key_bits:
            for ks in key_shapes:
                for key0 in key_bytes_cands:
                    key = shape_key(key0, kb, ks)
                    for ivm in iv_modes:
                        for ivs in iv_shapes:
                            for iv0 in iv_bytes_cands:
                                iv = shape_iv(iv0, ivm)
                                data = pkcs7_pad(p, 16)
                                try:
                                    ct = AES.new(key, AES.MODE_CBC, iv).encrypt(data)
                                    b64 = base64.b64encode(ct).decode("ascii")
                                except Exception:
                                    continue
                                tried += 1
                                if b64 == expect:
                                    matches.append({
                                        "encoding": enc,
                                        "key_bits": kb,
                                        "key_shape": ks,
                                        "iv_mode": ivm,
                                        "iv_note": ("ENV" if ivm=="ENV" else "ZERO"),
                                        "key_src_len": len(key0),
                                        "iv_src_len": len(iv0),
                                        "b64": b64
                                    })
                                    # 바로 리턴(첫 일치)
                                    return JSONResponse({
                                        "status":"MATCH",
                                        "tried": tried,
                                        "enc_spec": enc_spec,
                                        "match": matches[0]
                                    })

    return JSONResponse({"status":"NO_MATCH", "tried": tried, "enc_spec": enc_spec})


def ensure_nhis_audit_table():
    try:
        with Session(engine) as s:
            s.exec(text("""
            CREATE TABLE IF NOT EXISTS nhis_audit (
              id BIGSERIAL PRIMARY KEY,
              created_at timestamptz DEFAULT now(),
              stage text,             -- start / complete / fetch
              callback_id text,
              err_code text,
              user_mask text,
              rsp_json jsonb
            );
            """))
            s.commit()
    except Exception as e:
        print("[NHIS][AUDIT][WARN]", repr(e))

def audit_nhis(stage, err_code, callback_id, rsp_json=None, user_mask=None):
    try:
        with Session(engine) as s:
            s.exec(text("""
            INSERT INTO nhis_audit (stage, err_code, callback_id, user_mask, rsp_json)
            VALUES (:stage, :err, :cbid, :mask, :rsp)
            """), {"stage":stage, "err":err_code, "cbid":callback_id, "mask":user_mask or "", "rsp":json.dumps(rsp_json or {})})
            s.commit()
    except Exception as e:
        print("[NHIS][AUDIT][ERR]", repr(e))


@app.get("/_routes")
def _routes():
    return [{"path": r.path, "methods": list(r.methods)} for r in app.routes if isinstance(r, APIRoute)]
