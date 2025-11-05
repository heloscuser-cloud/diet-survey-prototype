from fastapi import (
    FastAPI, Request, Form, Depends, Response,
    Cookie, HTTPException, UploadFile, File,
    APIRouter, BackgroundTasks
)
from fastapi.staticfiles import StaticFiles
from app.vendors.datahub_client import encrypt_field
import re
from fastapi.templating import Jinja2Templates
from fastapi.routing import APIRoute
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse, JSONResponse
from sqlmodel import SQLModel, Field, Session, create_engine, select, Relationship
from pydantic import BaseModel
from typing import Optional, List, Any
from fastapi import Query
from datetime import datetime, timedelta, date, timezone
from starlette.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from random import randint
from sqlalchemy import func, Column, LargeBinary, Integer, text
from sqlalchemy import text as sa_text
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
from app.vendors.datahub_client import DatahubClient, DatahubError, pick_latest_general
import logging, pathlib
from app.vendors.datahub_client import DatahubClient, _crypto_selftest, encrypt_field

# ★ 로그 설정(강제적용)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True,
)

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

KST = ZoneInfo("Asia/Seoul")

def to_kst(dt: datetime) -> datetime:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(KST)

def now_kst() -> datetime:
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
   
    response = await call_next(request)
    #디버그용 경로허용
    if request.url.path.startswith("/debug/"):
        return await call_next(request)
    p = request.url.path
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

class SurveyResponse(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    respondent_id: int = Field(index=True)
    answers_json: str
    score: Optional[int] = None
    submitted_at: datetime = Field(default_factory=datetime.utcnow)

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


#---- 공단검진 결과 데이터 가공/저장 헬퍼 ----#

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
    return templates.TemplateResponse("nhis_fetch.html", {"request": request, "next_url": "/info", "datahub_auth_base": auth_base})

    
@app.get("/healthz")
def healthz():
    return PlainTextResponse("ok", status_code=200)

@app.post("/info")
async def info_submit(
    request: Request,
    name: str = Form(...),
    birth_date: str = Form(...),   # "YYYY-MM-DD"
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
        return templates.TemplateResponse("error.html", {"request": request, "message": "생년월일 형식이 올바르지 않습니다(YYYY-MM-DD)."}, status_code=400)
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

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login/send")
def login_send(request: Request, phone: str = Form(...), session: Session = Depends(get_session)):
    phone_digits = "".join([c for c in phone if c.isdigit()])
    if len(phone_digits) < 10 or len(phone_digits) > 11:
        return templates.TemplateResponse("error.html", {"request": request, "message": "휴대폰 번호를 올바르게 입력해주세요."}, status_code=400)
    issue_otp(session, phone_digits)
    tmp = signer.sign(f"otp:{phone_digits}").decode("utf-8")
    return RedirectResponse(url=f"/login/verify?t={tmp}", status_code=303)

def read_tmp_phone(t: str) -> str | None:
    try:
        raw = signer.unsign(t, max_age=300).decode("utf-8")
        if raw.startswith("otp:"):
            return raw.split(":")[1]
    except Exception:
        return None
    return None

@app.get("/login/verify", response_class=HTMLResponse)
def login_verify_page(request: Request, t: str):
    phone = read_tmp_phone(t)
    if not phone:
        return templates.TemplateResponse("error.html", {"request": request, "message": "인증이 만료되었습니다. 다시 시도해주세요."}, status_code=400)
    return templates.TemplateResponse("verify.html", {"request": request, "t": t})

@app.post("/login/verify")
def login_verify(request: Request, t: str = Form(...), code: str = Form(...), session: Session = Depends(get_session)):
    phone = read_tmp_phone(t)
    if not phone:
        return templates.TemplateResponse("error.html", {"request": request, "message": "인증이 만료되었습니다. 다시 시도해주세요."}, status_code=400)
    if not verify_otp(session, phone, code):
        return templates.TemplateResponse("error.html", {"request": request, "message": "인증 코드가 올바르지 않습니다."}, status_code=400)

    ph = hash_phone(phone)
    user = session.exec(select(User).where(User.phone_hash == ph)).first()
    if not user:
        user = User(phone_hash=ph)
        session.add(user)
        session.commit()
        session.refresh(user)
   
    resp = RedirectResponse(url="/nhis", status_code=303)  # 로그인 → NHIS 조회 → info
    SECURE_COOKIE = bool(int(os.getenv("SECURE_COOKIE", "1")))
    resp.set_cookie(
        AUTH_COOKIE_NAME,
        sign_user(user.id),
        httponly=True,
        secure=SECURE_COOKIE,
        samesite="lax",
        max_age=AUTH_MAX_AGE
    )
    # 혹시 남아있을 수도 있는 완료 쿠키 제거(새 설문 시작을 방해하지 않도록)
    resp.delete_cookie("survey_completed")
    return resp

@app.get("/logout")
def logout():
    resp = RedirectResponse(url="/", status_code=302)
    resp.delete_cookie(AUTH_COOKIE_NAME)
    return resp

#---- 기존 테스트용 문진 ----#

@app.get("/survey_legacy", response_class=HTMLResponse)
def survey(request: Request, session: Session = Depends(get_session), auth: str | None = Cookie(default=None, alias=AUTH_COOKIE_NAME)):
    user_id = verify_user(auth) if auth else -1
    if user_id < 0:
        return RedirectResponse(url="/login", status_code=302)

    # create respondent draft if not exists (simple one per visit)
    user = session.get(User, user_id)
    resp = Respondent(user_id=user.id, campaign_id="demo", status="draft")
    session.add(resp)
    session.commit()
    session.refresh(resp)

    # prefill (masked)
    prefill = {
        "name_masked": f"{(user.name_enc or '사용자')[0]}*",
        "birth_year": user.birth_year or "",
        "gender": user.gender or "",
        "phone_masked": "****-****"
    }
    token = signer.sign(str(resp.id)).decode("utf-8")
    return templates.TemplateResponse("survey.html", {"request": request, "prefill": prefill, "token": token})

def verify_token(token: str) -> int:
    try:
        raw = signer.unsign(token, max_age=3600*3)
        return int(raw.decode("utf-8"))
    except (BadSignature, SignatureExpired):
        return -1

# --- NHIS: 인증 완료 후 실제 데이터 1건 조회 + 세션 저장 헬퍼---
def _fetch_and_save_latest_nhis(request, start_payload, callback_id: Optional[str] = None):
    """
    start_payload 예:
    {
      "loginOption": "3",
      "telecom": "3",
      "userName": "홍길동",
      "hpNumber": "01012345678",
      "birth": "19900101"
    }
    """
    loginOption   = str(start_payload.get("loginOption","")).strip()
    telecom       = str(start_payload.get("telecom","")).strip()
    userName      = str(start_payload.get("userName","")).strip()
    hpNumber      = str(start_payload.get("hpNumber","")).strip()
    birth         = str(start_payload.get("birth","")).strip()
    telecom_gubun = telecom if loginOption == "3" and telecom else None

    # 실제 데이터 조회
    rsp2 = DATAHUB.medical_checkup_simple(
        login_option=loginOption,
        user_name=userName,
        hp_number=hpNumber,
        jumin_or_birth=birth,     # 라이브러리 시그니처가 jumin_or_birth면 birth를 그대로 넣음
        telecom_gubun=telecom_gubun,
        callback_id=callback_id,
    )

    # 가장 최근 일반검진 1건만 추출
    try:
        picked = pick_latest_general(rsp2.get("data") or rsp2)
    except Exception:
        picked = None

    request.session["nhis_latest"] = picked
    request.session["nhis_raw"] = rsp2  # 디버깅용

    return rsp2


class SurveyIn(BaseModel):
    meals_per_day: int
    late_snack_per_week: int
    veggies_servings_per_day: int
    sugary_drink_per_week: int
    alcohol_days_per_week: int

def score_survey(data: SurveyIn) -> int:
    score = 0
    score += max(0, min(20, data.veggies_servings_per_day * 4))
    score += max(0, 10 - min(10, data.late_snack_per_week))
    score += max(0, 10 - min(10, data.sugary_drink_per_week))
    score += max(0, 10 - min(7, data.alcohol_days_per_week))
    score += max(0, min(10, data.meals_per_day * 2))
    return int(score)

@app.post("/survey/submit_legacy")
async def survey_submit(request: Request,
                        token: str = Form(...),
                        meals_per_day: int = Form(...),
                        late_snack_per_week: int = Form(...),
                        veggies_servings_per_day: int = Form(...),
                        sugary_drink_per_week: int = Form(...),
                        alcohol_days_per_week: int = Form(...),
                        session: Session = Depends(get_session)):
    respondent_id = verify_token(token)
    if respondent_id < 0:
        return templates.TemplateResponse("error.html", {"request": request, "message": "세션이 만료되었어요. 다시 시작해주세요."}, status_code=401)

    resp_obj = SurveyIn(meals_per_day=meals_per_day,
                        late_snack_per_week=late_snack_per_week,
                        veggies_servings_per_day=veggies_servings_per_day,
                        sugary_drink_per_week=sugary_drink_per_week,
                        alcohol_days_per_week=alcohol_days_per_week)
    score = score_survey(resp_obj)

    import json as pyjson
    sr = SurveyResponse(respondent_id=respondent_id, answers_json=pyjson.dumps(resp_obj.dict(), ensure_ascii=False), score=score)
    session.add(sr)
    resp = session.get(Respondent, respondent_id)
    resp.status = "submitted"
    session.add(resp)
    nhis_latest = request.session.get("nhis_latest")
    nhis_raw    = request.session.get("nhis_raw")
    if nhis_latest is not None:
        try:
            sr.nhis_json = nhis_latest
        except Exception:
            pass
    if nhis_raw is not None:
        try:
            sr.nhis_raw = nhis_raw
        except Exception:
            pass
    session.commit()
    session.refresh(sr)

    rtoken = signer.sign(f"{sr.id}").decode("utf-8")
    return RedirectResponse(url=f"/report/ready?rtoken={rtoken}", status_code=303)

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


# 관리자 전용 라우터
admin_router = APIRouter(
    prefix="/admin",
    dependencies=[Depends(admin_required)]
)

@app.get("/admin/login", response_class=HTMLResponse)
def admin_login_form(request: Request):
    return templates.TemplateResponse("admin/login.html", {"request": request, "error": None})

@app.post("/admin/login")
def admin_login(request: Request, username: str = Form(...), password: str = Form(...)):
    ok = (
        ADMIN_USER and ADMIN_PASS and
        secrets.compare_digest(username, ADMIN_USER) and
        secrets.compare_digest(password, ADMIN_PASS)
    )
    if not ok:
        return templates.TemplateResponse(
            "admin/login.html",
            {"request": request, "error": "아이디 또는 비밀번호가 올바르지 않습니다."},
            status_code=401
        )

    # --- 세션 발급 ---
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
def survey_finish(request: Request,
                  acc: str,
                  rtoken: str,
                  background_tasks: BackgroundTasks,
                  session: Session = Depends(get_session)):
    respondent_id = verify_token(rtoken)
    if respondent_id < 0:
        return templates.TemplateResponse("error.html", {"request": request, "message": "세션이 만료되었습니다. 다시 시작해주세요."}, status_code=401)

    # --- [NEW] 폼(acc) → "1부터 시작하는 번호"로 정규화 ---
    # acc: {"q1":"2","q2":"1",...,"q23":["1","4"]} 형태(문자열/문자열리스트)라고 가정
    try:
        acc_obj = json.loads(acc) if acc else {}
    except Exception:
        acc_obj = {}

    # 질문 id 오름차순(1..23) 기준으로 번호만 추출
    answers_indices = []
    # ALL_QUESTIONS는 기동 시 로드된 전체 질문(JSON)
    # 각 질문 객체는 {"id": 1, "title": "...", "type": "radio"/"checkbox", ...} 형태라고 가정
    for q in sorted(ALL_QUESTIONS, key=lambda x: x["id"]):
        key = f"q{q['id']}"
        if q.get("type") == "checkbox":
            raw_list = acc_obj.get(key, [])
            if isinstance(raw_list, str):
                # 혹시 "1,3,4" 같이 들어오면 분할
                raw_list = [s.strip() for s in raw_list.split(",") if s.strip()]
            # 정수로 정규화(1부터). 비정상 값은 제외
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


    nhis_latest = request.session.get("nhis_latest")
    if nhis_latest:
        sr.nhis_json = nhis_latest

    # DB에는 깔끔하게 번호만 저장
    normalized_payload = {"answers_indices": answers_indices}
    resp = session.get(Respondent, respondent_id)

   
    nhis = request.session.get("nhis_latest")

    if nhis:
        normalized_payload["nhis"] = nhis 


    sr = SurveyResponse(
        respondent_id=respondent_id,
        answers_json=json.dumps(normalized_payload, ensure_ascii=False),
        score=None
    )
    session.add(sr)
    
    if resp:
        resp.status = "submitted"
        session.add(resp)
    session.commit()
    session.refresh(sr)
    
    if resp.serial_no is None:
        next_val = session.exec(sa_text("SELECT nextval('respondent_serial_no_seq')")).one()[0]
        resp.serial_no = next_val
        session.add(resp)
        session.commit()
        session.refresh(resp)

    # --- 알림 메일 비동기 발송 트리거 ---
    user = session.get(User, resp.user_id) if resp else None
    try:
        applicant_name = (resp.applicant_name or (user.name_enc if 'user' in locals() and user else "")) if resp else ""
        created_at_kst_str = to_kst(resp.created_at).strftime("%Y-%m-%d %H:%M") if resp and resp.created_at else now_kst().strftime("%Y-%m-%d %H:%M")
        serial_no_val = resp.serial_no or 0

        # 환경 구성 체크 로그
        print("[EMAIL] TRY send",
            "SMTP_HOST=", os.getenv("SMTP_HOST"),
            "SMTP_USER=", os.getenv("SMTP_USER"),
            "SMTP_FROM=", os.getenv("SMTP_FROM"),
            "SMTP_TO=", os.getenv("SMTP_TO"))

        # BackgroundTasks로 비동기 발송
        background_tasks.add_task(
            send_submission_email,
            serial_no_val,
            applicant_name,
            created_at_kst_str
        )
    except Exception as e:
        print("[EMAIL] enqueue failed:", repr(e))

    response = RedirectResponse(url="/portal", status_code=302)
    
    # 설문 완료 플래그(브라우저 뒤로가기로 입력 복귀 차단용)
    response.set_cookie("survey_completed", "1", max_age=60*60*24*7, httponly=True, samesite="Lax", secure=bool(int(os.getenv("SECURE_COOKIE","1"))))
    return response


@app.post("/admin/responses/export.xlsx")
async def admin_export_xlsx(
    request: Request,
    response: Response,
    ids: str = Form(...),
    session: Session = Depends(get_session),
    _auth: None = Depends(admin_required),
):

    # 디버그 로그
    print("export.xlsx ids raw:", repr(ids))

    # ids 파싱
    id_list = [int(x) for x in (ids or "").split(",") if x.strip().isdigit()]
    if not id_list:
        return RedirectResponse(url="/admin/responses", status_code=303)

    # 답 파서 (여러 저장포맷 대응)
    def extract_answers(payload: dict, questions_sorted: list[dict]) -> list:
        """
        answers_indices가 있으면 우선 사용하되,
        그 안에서 빈칸(None, "", [])은 q{id} 값으로 보정.
        없으면 q{id}로 전체 구성.
        """
        def to_num(v):
            if v is None or v == "": return ""
            if isinstance(v, list):
                return [int(x) for x in v if str(x).isdigit()]
            if isinstance(v, str) and "," in v:
                return [int(x) for x in v.split(",") if x.strip().isdigit()]
            return int(v) if str(v).isdigit() else ""

        ai = payload.get("answers_indices")
        if isinstance(ai, list) and len(ai) == len(questions_sorted):
            # 보정용 map (answers/acc/data 등 nested에서도 q{id} 찾아봄)
            candidates = [payload]
            for k in ("answers", "acc", "data"):
                if isinstance(payload.get(k), dict):
                    candidates.append(payload[k])
            def fallback(idx, qid):
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

        # answers_indices가 없으면 q{id}로 구성
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

    # 엑셀 워크북/시트
    wb = Workbook()
    ws = wb.active
    ws.title = "문진결과"

    # 질문 타이틀
    def q_title(q: dict):
        return (q.get("title") or q.get("text") or q.get("label")
                or q.get("question") or q.get("prompt") or q.get("name")
                or f"Q{q.get('id','')}")
    questions_sorted = sorted(ALL_QUESTIONS, key=lambda x: x["id"])
    questions = [q_title(q) for q in questions_sorted]

    # 헤더
    fixed_headers = ["no.", "신청번호", "이름", "생년월일", "나이(만)", "성별", "신장", "체중"]
    ws.append(fixed_headers + questions)

    # 유틸
    today = now_kst().date()
    def calc_age(bd, ref_date):
        if not bd: return ""
        return ref_date.year - bd.year - ((ref_date.month, ref_date.day) < (bd.month, bd.day))
    def fmt(v):
        if isinstance(v, list): return ",".join(str(x) for x in v)
        return "" if v is None else str(v)

    # 데이터 행
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

        row = [
            idx,
            serial_no,
            name,
            (bd.isoformat() if bd else ""),
            age,
            gender,
            ("" if height is None else height),
            ("" if weight is None else weight),
        ] + [fmt(v) for v in answers]
        ws.append(row)

    # 바이너리 응답
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
async def dh_simple_start(request: Request):
    payload = await request.json()
    loginOption  = str(payload.get("loginOption","")).strip()
    telecom      = str(payload.get("telecom","")).strip()
    userName     = str(payload.get("userName","")).strip()
    hpNumber     = str(payload.get("hpNumber","")).strip()
    juminOrBirth = str(payload.get("juminOrBirth","")).strip()

    telecom_gubun = telecom if loginOption == "3" and telecom else None

    # 시작 전용 호출
    rsp = DATAHUB.simple_auth_start(
        login_option=loginOption,
        user_name=userName,
        hp_number=hpNumber,
        jumin_or_birth=juminOrBirth,
        telecom_gubun=telecom_gubun,
    )

    # 안전 로그(마스킹)
    print("[DH-START][SAFE]", {
        "LOGINOPTION": loginOption, "USERNAME": userName[:1]+"*"*(max(0,len(userName)-1)),
        "HPNUMBER_LAST4": hpNumber[-4:], "TELECOMGUBUN": telecom_gubun or "", "JUMIN":"***MASKED***"
    })

    err  = str(rsp.get("errCode",""))
    data = rsp.get("data") or rsp.get("Data") or {}

    # 콜백형: callbackId 반환
    cbid = (data.get("callbackId") or rsp.get("callbackId"))
    if err in ("0001","1") and cbid:
        # 세션에도 저장(선택)
        request.session["dh_callback"] = {"id": cbid, "type": "ANY"}
        return JSONResponse({"errCode":"0001","message":"NEED_CALLBACK","data":{"callbackId": cbid}}, status_code=200)

    # 즉시형: 0000
    # 0000 즉시형: 이 응답 안에 결과가 동봉되는 케이스가 있으므로
    # 최신 1건을 골라 세션에 저장해둔다.
    if err in ("0000", "0"):
        try:
            # rsp 전체에서 최신 1건 추출 (네 프로젝트에서 쓰는 헬퍼명 유지)
            picked = pick_latest_general(rsp)
        except Exception:
            picked = None

        request.session["nhis_latest"] = picked
        request.session["nhis_raw"] = rsp

        # 프론트가 자동 완료 호출(direct) 하도록 플래그 내려줌
        return JSONResponse({
            "errCode": "0000",
            "message": "IMMEDIATE_OK",
            "data": {"immediate": True}
        }, status_code=200)

    # 실패
    return JSONResponse({"errCode": err or "9999", "message": rsp.get("message") or "FAIL"}, status_code=400)



# ===========================================
# DataHub 간편인증 Step1-2: 진행 중
# ===========================================
# 현재 post_captcha가 대체 중으로 사용하지 않음
# @app.get("/api/dh/simple/status")
# def dh_simple_status(callback_id: str):
#     rsp = DATAHUB.post_captcha(callbackId=callback_id, callbackType="ANY")
#     return rsp


# ===========================================
# DataHub 간편인증 Step2: 완료(captcha)
# ===========================================

@app.post("/api/dh/simple/complete")
async def dh_simple_complete(request: Request):
    payload = await request.json()

    # 1) 즉시형(direct) 처리
    if bool(payload.get("direct")):
        latest = (request.session or {}).get("nhis_latest")
        if latest:
            return JSONResponse({"errCode":"0000","message":"IMMEDIATE_OK","data": latest}, status_code=200)
        # 혹시 세션 저장이 누락됐을 경우를 대비한 안전장치(선택)
        # 필요 없다면 아래 2줄은 지워도 됨
        # fallback = (request.session or {}).get("nhis_raw")
        # if fallback: return JSONResponse({"errCode":"0000","message":"IMMEDIATE_OK","data": pick_latest_general(fallback)}, status_code=200)
        return JSONResponse({"errCode":"9002","message":"즉시형 결과가 세션에 없습니다. 다시 시도해 주세요."}, status_code=400)

    # 2) 콜백형 처리
    cbid = str(payload.get("callbackId","")).strip()
    if not cbid:
        # 세션 폴백(선택)
        sess_cb = (request.session or {}).get("dh_callback") or {}
        cbid = sess_cb.get("id","")
    if not cbid:
        return JSONResponse({"errCode":"9001","message":"callbackId가 없습니다."}, status_code=400)

    # (a) 캡차/콜백 상태 폴링
    max_wait_sec = 60
    deadline = time.time() + max_wait_sec
    while time.time() < deadline:
        try:
            cap = DATAHUB.simple_auth_complete(callback_id=cbid, callback_type="ANY")
        except Exception as e:
            print("[DH-COMPLETE][ERR][captcha]", repr(e))
            time.sleep(2)
            continue

        # (b) 콜백ID로 결과 재조회(새 인증 시작 X)
        try:
            res = DATAHUB.post_medical_glance_simple_with_callbackid(callbackId=cbid)
        except Exception as e:
            print("[DH-COMPLETE][ERR][glance]", repr(e))
            time.sleep(2)
            continue

        err = str(res.get("errCode",""))
        if err == "0000":
            # 최신 1건 추려서 세션 저장(엑셀 병합용)
            try:
                picked = pick_latest_general(res)
                request.session["nhis_latest"] = picked
            except Exception as e:
                print("[DH-COMPLETE][WARN][pick]", repr(e))
            return JSONResponse({"errCode":"0000","message":"OK","data":{}}, status_code=200)

        # 아직 원천기관 처리 중 → 잠깐 대기 후 재시도
        time.sleep(2)

    # 타임아웃 → 프론트엔 재시도 유도(=202)
    return JSONResponse({"errCode":"2020","message":"PENDING"}, status_code=202)



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

#임시 인코딩 테스트 용도 
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
