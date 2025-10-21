from fastapi import FastAPI, Request, Form, Depends, Response, Cookie, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import SQLModel, Field, Session, create_engine, select
from pydantic import BaseModel
from typing import Optional, List
from itsdangerous import TimestampSigner, BadSignature, SignatureExpired
from datetime import datetime, timedelta
from random import randint
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import os
import secrets
from datetime import datetime, timedelta
from typing import Optional

import itsdangerous
from fastapi import Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from sqlmodel import Session, select
from sqlalchemy import func

import json
from pathlib import Path
from fastapi import Request, Form
from fastapi.responses import RedirectResponse


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

class Respondent(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True)
    campaign_id: str = Field(default="default")
    status: str = Field(default="draft")
    created_at: datetime = Field(default_factory=datetime.utcnow)

class SurveyResponse(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    respondent_id: int = Field(index=True)
    answers_json: str
    score: Optional[int] = None
    submitted_at: datetime = Field(default_factory=datetime.utcnow)

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
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
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

    resp = RedirectResponse(url="/portal", status_code=303)
    resp.set_cookie(AUTH_COOKIE_NAME, sign_user(user.id), httponly=True, secure=False, samesite="lax", max_age=AUTH_MAX_AGE)
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

def render_pdf(path: str, title: str, score: int, tips: List[str]):
    c = canvas.Canvas(path, pagesize=A4)
    width, height = A4
    x_margin, y_margin = 20 * mm, 20 * mm

    c.setFont(DEFAULT_FONT_BOLD, 20)
    c.drawString(x_margin, height - y_margin, title)

    c.setFont(DEFAULT_FONT, 12)
    c.drawString(x_margin, height - y_margin - 20, f"생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    c.drawString(x_margin, height - y_margin - 40, f"총점: {score} / 60")

    c.line(x_margin, height - y_margin - 50, width - x_margin, height - y_margin - 50)

    c.setFont(DEFAULT_FONT_BOLD, 14)
    c.drawString(x_margin, height - y_margin - 80, "개선 팁")
    c.setFont(DEFAULT_FONT, 12)

    y = height - y_margin - 100
    for tip in tips:
        c.circle(x_margin + 3, y + 3, 2, fill=1)
        c.drawString(x_margin + 12, y, tip)
        y -= 18
        if y < 40 * mm:
            c.showPage()
            y = height - y_margin

    c.showPage()
    c.save()

@app.get("/report/pdf")
def report_pdf(rtoken: str, session: Session = Depends(get_session)):
    try:
        raw = signer.unsign(rtoken, max_age=3600*24)
    except (BadSignature, SignatureExpired):
        return Response("링크가 만료되었어요.", status_code=401)

    report_id = int(raw.decode("utf-8"))
    sr = session.get(SurveyResponse, report_id)
    if not sr:
        return Response("리포트를 찾을 수 없어요.", status_code=404)

    tips = []
    if sr.score < 30:
        tips.append("설탕이 많은 음료수를 물 또는 무가당 차로 대체해보세요.")
        tips.append("야식 빈도를 주 1회 이하로 줄여보세요.")
        tips.append("채소 섭취를 하루 3회 이상 목표로 해보세요.")
    else:
        tips.append("균형 잡힌 식단을 잘 유지하고 있어요. 👍")
        tips.append("채소 섭취를 꾸준히 이어가세요.")
        tips.append("가끔 간단한 간식은 과일로 대체해보세요.")

    reports_dir = os.path.join(ROOT_DIR, "app", "data")
    os.makedirs(reports_dir, exist_ok=True)
    pdf_path = os.path.join(reports_dir, f"report_{report_id}.pdf")

    render_pdf(pdf_path, "식습관 문진 결과 리포트", sr.score or 0, tips)

    return FileResponse(pdf_path, filename=f"report_{report_id}.pdf", media_type="application/pdf")

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
                "submitted_at": sr.submitted_at.strftime("%Y-%m-%d %H:%M"),
                "score": sr.score
            })
    reports.sort(key=lambda x: x["id"], reverse=True)
    return templates.TemplateResponse("portal.html", {"request": request, "reports": reports})

@app.get("/portal/report/{report_id}/download")
def portal_report_download(report_id: int, auth: str | None = Cookie(default=None, alias=AUTH_COOKIE_NAME), session: Session = Depends(get_session)):
    user_id = verify_user(auth) if auth else -1
    if user_id < 0:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")

    sr = session.get(SurveyResponse, report_id)
    if not sr:
        raise HTTPException(status_code=404, detail="리포트를 찾을 수 없습니다.")

    resp = session.get(Respondent, sr.respondent_id)
    if not resp or resp.user_id != user_id:
        raise HTTPException(status_code=403, detail="접근 권한이 없습니다.")

    reports_dir = os.path.join(ROOT_DIR, "app", "data")
    pdf_path = os.path.join(reports_dir, f"report_{report_id}.pdf")
    if not os.path.exists(pdf_path):
        tips = ["균형 잡힌 식단을 유지하세요.", "당류 음료를 줄여보세요.", "채소 섭취를 늘려보세요."]
        render_pdf(pdf_path, "식습관 문진 결과 리포트", sr.score or 0, tips)

    filename = f"report_{report_id}.pdf"
    return FileResponse(pdf_path, filename=filename, media_type="application/pdf")


# --- Admin host gate ---
ADMIN_ALLOWED_HOSTS = {
    "admin.gaonnsurvey.store",  # 프로덕션
    "localhost", "127.0.0.1"    # 로컬 테스트
}

def require_admin_host(request: Request):
    host = (request.headers.get("host") or "").split(":")[0].lower()
    if host not in ADMIN_ALLOWED_HOSTS:
        raise HTTPException(status_code=404, detail="Not found")



#---- 관리자 로그인 ----#

APP_SECRET = os.environ.get("APP_SECRET", "dev-secret")
ADMIN_USER = os.environ.get("ADMIN_USER")
ADMIN_PASS = os.environ.get("ADMIN_PASS")
_signer = itsdangerous.URLSafeSerializer(APP_SECRET, salt="admin-cookie")
COOKIE_NAME = "admin"
COOKIE_MAX_AGE = 7200  # 2시간
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
    if not token or not validate_admin_cookie(token):
        raise HTTPException(status_code=401, detail="Unauthorized")

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
    resp.set_cookie(COOKIE_NAME, create_admin_cookie(), httponly=True, secure=SECURE_COOKIE,
                    samesite="lax", max_age=COOKIE_MAX_AGE, path="/")
    return resp

@app.get("/admin/logout")
def admin_logout():
    resp = RedirectResponse(url="/admin/login", status_code=303)
    resp.delete_cookie(COOKIE_NAME, path="/")
    return resp

@app.get("/admin/responses", response_class=HTMLResponse)
def admin_responses(
    request: Request,
    page: int = 1,
    q: Optional[str] = None,
    min_score: Optional[int] = None,
    from_: Optional[str] = None,
    to: Optional[str] = None,
    session: Session = Depends(get_session),
    _auth: None = Depends(admin_required),
):
    PAGE_SIZE = 20
    page = max(1, page)

    stmt = (
        select(SurveyResponse, Respondent, User)
        .join(Respondent, Respondent.id == SurveyResponse.respondent_id)
        .join(User, User.id == Respondent.user_id)
    )
    if q:
        like = f"%{q}%"
        stmt = stmt.where(
            (User.name_enc.ilike(like)) |
            (User.phone_hash.ilike(like)) |
            (SurveyResponse.answers_json.ilike(like))
        )
    if min_score:
        stmt = stmt.where(SurveyResponse.score >= min_score)

    def parse_date(s: Optional[str]):
        try: return datetime.strptime(s, "%Y-%m-%d").date()
        except: return None

    d_from = parse_date(from_)
    d_to = parse_date(to)
    if d_from:
        stmt = stmt.where(SurveyResponse.submitted_at >= datetime(d_from.year, d_from.month, d_from.day))
    if d_to:
        stmt = stmt.where(SurveyResponse.submitted_at < datetime(d_to.year, d_to.month, d_to.day) + timedelta(days=1))

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = session.exec(count_stmt).one()
    rows = session.exec(stmt.order_by(SurveyResponse.submitted_at.desc()).offset((page-1)*PAGE_SIZE).limit(PAGE_SIZE)).all()
    total_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE

    return templates.TemplateResponse("admin/responses.html", {
        "request": request, "rows": rows, "page": page,
        "total": total, "total_pages": total_pages,
        "q": q, "min_score": min_score, "from": from_, "to": to,
    })

@app.get("/admin/responses.csv")
def admin_responses_csv(
    q: Optional[str] = None,
    min_score: Optional[int] = None,
    from_: Optional[str] = None,
    to: Optional[str] = None,
    session: Session = Depends(get_session),
    _auth: None = Depends(admin_required),
):
    def parse_date(s: Optional[str]):
        try: return datetime.strptime(s, "%Y-%m-%d").date()
        except: return None
    d_from, d_to = parse_date(from_), parse_date(to)

    def generate():
        yield "\ufeff"
        s = StringIO()
        w = csv.writer(s)
        w.writerow(["id","respondent_id","user_id","score","submitted_at","phone_hash","name_enc"])
        yield s.getvalue(); s.seek(0); s.truncate(0)
        stmt = (
            select(SurveyResponse, Respondent, User)
            .join(Respondent, Respondent.id == SurveyResponse.respondent_id)
            .join(User, User.id == Respondent.user_id)
        )
        if q:
            like = f"%{q}%"
            stmt = stmt.where(
                (User.name_enc.ilike(like)) |
                (User.phone_hash.ilike(like)) |
                (SurveyResponse.answers_json.ilike(like))
            )
        if min_score:
            stmt = stmt.where(SurveyResponse.score >= min_score)
        if d_from:
            stmt = stmt.where(SurveyResponse.submitted_at >= datetime(d_from.year, d_from.month, d_from.day))
        if d_to:
            stmt = stmt.where(SurveyResponse.submitted_at < datetime(d_to.year, d_to.month, d_to.day) + timedelta(days=1))
        for sr, resp, user in session.exec(stmt.order_by(SurveyResponse.id)):
            w.writerow([sr.id, sr.respondent_id, user.id, sr.score or "", sr.submitted_at, user.phone_hash, user.name_enc])
            yield s.getvalue(); s.seek(0); s.truncate(0)
    return StreamingResponse(generate(), media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="responses.csv"'})


#---- 문진 가져오기 ----#

@app.get("/survey")
def survey_root(auth: str | None = Cookie(default=None, alias=AUTH_COOKIE_NAME),
                session: Session = Depends(get_session)):
    # 로그인 확인
    user_id = verify_user(auth) if auth else -1
    if user_id < 0:
        return RedirectResponse(url="/login", status_code=302)

    # 응답자 초안 생성 (기존 legacy와 동일한 개념)
    user = session.get(User, user_id)
    resp = Respondent(user_id=user.id, campaign_id="demo", status="draft")
    session.add(resp)
    session.commit()
    session.refresh(resp)

    # 응답자 토큰 (서명) 생성 — 이후 단계들에서 계속 들고 다님
    rtoken = signer.sign(str(resp.id)).decode("utf-8")
    return RedirectResponse(url=f"/survey/1?rtoken={rtoken}", status_code=303)

@app.get("/survey/{step}", response_class=HTMLResponse)
def survey_step_get(request: Request, step: int, rtoken: str, acc: str | None = None):
    # 단계 범위 체크
    if step < 1 or step > 3:
        return RedirectResponse(url="/survey/1", status_code=303)

    # rtoken 검증(응답자 ID 복원)
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

@app.post("/survey/{step}")
async def survey_step_post(request: Request, step: int,
                           acc: str = Form("{}"),
                           rtoken: str = Form(...)):
    # 응답자 토큰 검증
    respondent_id = verify_token(rtoken)
    if respondent_id < 0:
        return RedirectResponse(url="/login", status_code=302)

    form = await request.form()

    # 누적 응답 dict
    try:
        acc_data = json.loads(acc) if acc else {}
    except Exception:
        acc_data = {}

    # 현재 단계 질문만 반영
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

    # 다음 단계 or 제출
    acc_q = json.dumps(acc_data, ensure_ascii=False)
    if step < 3:
        return RedirectResponse(
            url=f"/survey/{step+1}?acc={acc_q}&rtoken={rtoken}",
            status_code=303
        )
    else:
        return RedirectResponse(
            url=f"/survey/submit?acc={acc_q}&rtoken={rtoken}",
            status_code=303
        )

@app.get("/survey/submit")
def survey_submit_get(request: Request,
                      acc: str,
                      rtoken: str,
                      session: Session = Depends(get_session)):
    # 응답자 ID 복원
    respondent_id = verify_token(rtoken)
    if respondent_id < 0:
        return templates.TemplateResponse("error.html", {"request": request, "message": "세션이 만료되었습니다. 다시 시작해주세요."}, status_code=401)

    # SurveyResponse 저장
    sr = SurveyResponse(
        respondent_id=respondent_id,
        answers_json=acc,
        score=None  # 점수 규칙 확정되면 계산해서 넣기
    )
    session.add(sr)

    # Respondent 상태 변경
    resp = session.get(Respondent, respondent_id)
    if resp:
        resp.status = "submitted"
        session.add(resp)

    session.commit()
    session.refresh(sr)

    # 보고서 준비 URL로 이동 (기존 로직 재사용)
    rtoken_report = signer.sign(f"{sr.id}").decode("utf-8")  # report용 토큰
    return RedirectResponse(url=f"/report/ready?rtoken={rtoken_report}", status_code=303)


from fastapi.routing import APIRoute

@app.get("/_routes")
def _routes():
    return [{"path": r.path, "methods": list(r.methods)} for r in app.routes if isinstance(r, APIRoute)]
