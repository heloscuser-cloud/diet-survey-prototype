from fastapi import FastAPI, Request, Form, Depends, Response, Cookie, HTTPException, UploadFile, File, APIRouter
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.routing import APIRoute
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import RedirectResponse, HTMLResponse, StreamingResponse
from fastapi.requests import Request
from sqlmodel import SQLModel, Field, Session, create_engine, select, Relationship
from pydantic import BaseModel
from typing import Optional, List
from fastapi import Query
from datetime import datetime, timedelta, date, timezone
from random import randint
from sqlalchemy import func, Column, LargeBinary, Integer, text
from sqlalchemy import text as sa_text
from itsdangerous import TimestampSigner, BadSignature, SignatureExpired, URLSafeSerializer
import smtplib
from email.message import EmailMessage
from fastapi import BackgroundTasks
import zipfile
import csv
import itsdangerous
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from io import StringIO, BytesIO
from openpyxl import Workbook
import csv
import secrets
import json
from pathlib import Path
from zoneinfo import ZoneInfo
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
AUTH_MAX_AGE = 3600 * 6  # 6 hours

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
    """신청 완료 알림 메일 발송 (네이버 SMTP/STARTTLS)"""
    host = os.getenv("SMTP_HOST", "smtp.naver.com")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASS")
    from_addr = os.getenv("SMTP_FROM", user or "")
    to_addr = os.getenv("SMTP_TO")

    if not (user and password and to_addr and from_addr):
        print("[EMAIL] SMTP env not configured, skip.")
        return

    masked = mask_second_char(applicant_name or "")
    subject = f"[서비스 접수 알림] 신청번호:{serial_no} / 신청자:{masked} / 신청일:{created_at_kst_str}"
    body = (
        "문진이 접수되었습니다.\n"
        f"신청번호:{serial_no}\n"
        f"신청자:{masked}\n"
        f"신청일:{created_at_kst_str}"
    )

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.set_content(body)

    try:
        with smtplib.SMTP(host, port, timeout=15) as s:
            s.ehlo()
            s.starttls()  # TLS
            s.ehlo()
            s.login(user, password)
            s.send_message(msg)
        print("[EMAIL] sent OK")
    except Exception as e:
        print("[EMAIL] send failed:", repr(e))


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
        session.add(user)
        session.commit()

    # 설문 시작
    return RedirectResponse(url="/survey", status_code=303)

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
   
    resp = RedirectResponse(url="/info", status_code=303)  # 로그인 → 인적정보 → 설문
    resp.set_cookie(AUTH_COOKIE_NAME, sign_user(user.id), httponly=True, secure=False, samesite="lax", max_age=AUTH_MAX_AGE)
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

# --- Admin host gate ---
ADMIN_HOST = "admin.gaonnsurvey.store"

def _norm_host(h: str) -> str:
    return (h or "").split(":")[0].strip().lower().rstrip(".")

@app.middleware("http")
async def require_admin_host(request: Request, call_next):
    p = request.url.path or ""
    h = _norm_host(request.headers.get("host"))
    if p.startswith("/admin") and h not in (ADMIN_HOST, "localhost", "127.0.0.1"):
        # 같은 경로로 관리자 호스트로 보냄
        target = f"https://{ADMIN_HOST}{p}"
        if request.url.query:
            target += f"?{request.url.query}"
        print(f"[ADMIN HOST] redirect {h} -> {ADMIN_HOST} path={p}")
        return RedirectResponse(target, status_code=307)
    return await call_next(request)




#---- 관리자 로그인 ----#

APP_SECRET = os.environ.get("APP_SECRET", "dev-secret")
ADMIN_USER = os.environ.get("ADMIN_USER")
ADMIN_PASS = os.environ.get("ADMIN_PASS")
_signer = itsdangerous.URLSafeSerializer(APP_SECRET, salt="admin-cookie")
COOKIE_NAME = "admin"
COOKIE_MAX_AGE = 30  # 로그인 세션 유지 시간 30분 설정
SECURE_COOKIE = os.environ.get("SECURE_COOKIE", "1") == "1"

def create_admin_cookie() -> str:
    return _signer.dumps({"role": "admin", "iat": int(datetime.utcnow().timestamp())})

def validate_admin_cookie(token: str) -> bool:
    try:
        return _signer.loads(token).get("role") == "admin"
    except itsdangerous.BadSignature:
        return False

def admin_required(request: Request):
    token = request.cookies.get(COOKIE_NAME)
    #진단 로그 임시
    print("[ADMIN AUTH]", request.method, request.url.path, "has_cookie=", bool(token))
    print("[ADMIN AUTH] cookie header =", request.headers.get("cookie"))
    print("[ADMIN AUTH]", request.method, request.url.path, "has_cookie=", bool(token))
    if not token or not validate_admin_cookie(token):
        raise HTTPException(status_code=401, detail="Unauthorized")

# 관리자 전용 라우터
admin_router = APIRouter(
    prefix="/admin",
    dependencies=[Depends(require_admin_host), Depends(admin_required)]
)

@app.get("/admin/login", response_class=HTMLResponse)
def admin_login_form(request: Request, _h: None = Depends(require_admin_host)):
    return templates.TemplateResponse("admin/login.html", {"request": request, "error": None})

@app.post("/admin/login")
def admin_login(request: Request, username: str = Form(...), password: str = Form(...), _h: None = Depends(require_admin_host)):
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
    resp = RedirectResponse(url="/admin/responses", status_code=303)
    resp.set_cookie(
        COOKIE_NAME,                # key
        create_admin_cookie(),      # value
        httponly=True,
        secure=True,                # HTTPS
        samesite="lax",             # 동일 사이트 탐색/POST에 항상 전송
        max_age=COOKIE_MAX_AGE,
        path="/",
        )
    print("[ADMIN LOGIN] set-cookie for host OK")
    return resp

@app.get("/admin/logout")
def admin_logout():
    resp = RedirectResponse(url="/admin/login", status_code=303)
    resp.delete_cookie(COOKIE_NAME, path="/")
    return resp



# 목록
@admin_router.get("/responses", response_class=HTMLResponse)
def admin_responses(
    request: Request,
    page: int = 1,
    q: Optional[str] = None,
    min_score: Optional[str] = None,   # ← int 대신 str로 받기
    status: Optional[str] = None,      # all | submitted | accepted | report_uploaded
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = Query(None, alias="to"),
    session: Session = Depends(get_session),
):
    PAGE_SIZE = 20
    page = max(1, page)

    # 안전 파싱
    try:
        min_score_i = int(min_score) if (min_score is not None and min_score != "") else None
    except ValueError:
        min_score_i = None

    stmt = (
        select(SurveyResponse, Respondent, User, ReportFile)
        .join(Respondent, Respondent.id == SurveyResponse.respondent_id)
        .join(User, User.id == Respondent.user_id)
        .join(ReportFile, ReportFile.survey_response_id == SurveyResponse.id, isouter=True)
    )
    if q:
        like = f"%{q}%"
        stmt = stmt.where(
            (User.name_enc.ilike(like)) |
            (User.phone_hash.ilike(like)) |
            (SurveyResponse.answers_json.ilike(like)) |
            (ReportFile.filename.ilike(like)) |
            (func.to_char(Respondent.created_at, 'YYYY-MM-DD').ilike(like))
        )
    if min_score_i is not None:
        stmt = stmt.where(SurveyResponse.score >= min_score_i)

    # KST 날짜 → UTC naive 범위 변환 함수 사용
    def parse_date(s: Optional[str]):
        try: return datetime.strptime(s, "%Y-%m-%d").date()
        except: return None
    d_from = parse_date(from_)
    d_to = parse_date(to)
    start_utc, end_utc = kst_date_range_to_utc_datetimes(d_from, d_to)
    if start_utc:
        stmt = stmt.where(Respondent.created_at >= start_utc)
    if end_utc:
        stmt = stmt.where(Respondent.created_at < end_utc)

    if status in ("submitted", "accepted", "report_uploaded"):
        stmt = stmt.where(Respondent.status == status)

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = session.exec(count_stmt).one()
    rows = session.exec(
        stmt.order_by(SurveyResponse.submitted_at.desc())
            .offset((page-1)*PAGE_SIZE)
            .limit(PAGE_SIZE)
    ).all()
    total_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE

    to_kst_str = lambda dt: to_kst(dt).strftime("%Y-%m-%d %H:%M")

    return templates.TemplateResponse("admin/responses.html", {
        "request": request, "rows": rows, "page": page,
        "total": total, "total_pages": total_pages,
        "q": q, "min_score": min_score, "from": from_, "to": to,
        "status": status,
        "to_kst_str": to_kst_str,
    })


# 접수완료 처리 (POST + Form)
@admin_router.post("/responses/accept")
def admin_bulk_accept(
    ids: str = Form(...),  # "1,2,3"
    session: Session = Depends(get_session),
):
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
        stmt = stmt.where(
            (User.name_enc.ilike(like)) |
            (User.phone_hash.ilike(like)) |
            (SurveyResponse.answers_json.ilike(like)) |
            (ReportFile.filename.ilike(like)) |
            (func.to_char(Respondent.created_at, 'YYYY-MM-DD').ilike(like))
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

    # DB에는 깔끔하게 번호만 저장
    normalized_payload = {"answers_indices": answers_indices}

    sr = SurveyResponse(
        respondent_id=respondent_id,
        answers_json=json.dumps(normalized_payload, ensure_ascii=False),
        score=None
    )
    session.add(sr)
    resp = session.get(Respondent, respondent_id)
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

    response = RedirectResponse(url="/portal", status_code=302)
    # 설문 완료 플래그(브라우저 뒤로가기로 입력 복귀 차단용)
    response.set_cookie("survey_completed", "1", max_age=60*60*24*7, httponly=True, samesite="Lax", secure=bool(int(os.getenv("SECURE_COOKIE","1"))))
    return response


@app.post("/admin/responses/export.xlsx")
def admin_export_xlsx(
    ids: str = Form(...),  # "1,2,3"
    session: Session = Depends(get_session),
    _h: None = Depends(require_admin_host),
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
        if isinstance(v, list): return ";".join(str(x) for x in v)
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


@app.get("/_routes")
def _routes():
    return [{"path": r.path, "methods": list(r.methods)} for r in app.routes if isinstance(r, APIRoute)]
