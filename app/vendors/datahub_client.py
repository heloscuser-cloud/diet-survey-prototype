import os, json, base64, hashlib
from typing import Any, Dict, Optional, Tuple, List
import requests, re
from Crypto.Cipher import AES

print("[MOD] datahub_client loaded from", __file__, flush=True)

class DatahubError(Exception):
    pass

def _pkcs7_pad(b: bytes, block=16) -> bytes:
    n = block - (len(b) % block)
    return b + bytes([n])*n

def _pkcs7_unpad(b: bytes) -> bytes:
    n = b[-1]
    if n < 1 or n > 16: return b
    return b[:-n]

def _parse_encspec(spec: str) -> Tuple[str, str, str]:
    """
    예시: 'AES256/CBC/PKCS7/IV=00000000000000000000000000000000'
    리턴 (algo, mode, padding)
    """
    spec = (spec or "").upper()
    parts = spec.split("/")
    if len(parts) < 3:
        raise DatahubError(f"Invalid EncSpec: {spec}")
    algo, mode, padding = parts[0], parts[1], parts[2]
    return algo, mode, padding

#인코딩 헬퍼
def _get_text_encoding() -> str:
    """
    평문 인코딩: 기본 'utf-8', ENV로 오버라이드
    - DATAHUB_TEXT_ENCODING = 'utf-8' | 'cp949' | 'euc-kr'
    """
    enc = (os.getenv("DATAHUB_TEXT_ENCODING", "utf-8") or "").strip().lower()
    if enc in ("utf8", "utf-8"): return "utf-8"
    if enc in ("cp949", "euc-kr", "euckr", "ksc5601"): return "cp949"  # cp949로 통일
    return "utf-8"

def _get_key_iv() -> Tuple[bytes, Optional[bytes]]:
    """
    EncKey/IV 디코드 + 강제 옵션 반영 + (개발모드) SELFTEST로
    올바른 key/iv 후보를 '직접' 선택.
    - DATAHUB_FORCE_KEY_BITS: 128|256 (기본 EncSpec 추정)
    - DATAHUB_FORCE_IV_MODE : ENV|ZERO (기본 ENV)
    - DATAHUB_FORCE_KEY_SHAPE: right|left|sha256|md5 (기본 right)
    - DATAHUB_SELFTEST_PLAIN / DATAHUB_SELFTEST_EXPECT 가 있으면
      모든 key/iv 후보(점/urlsafe/hex/raw) 중에서 'expect'를 만들어내는
      후보를 선택하여 반환한다.
    """
    import base64, os, hashlib
    from Crypto.Cipher import AES

    enc_key = (os.getenv("DATAHUB_ENC_KEY_B64", "") or "").strip()
    iv_env  = (os.getenv("DATAHUB_ENC_IV_B64", "")  or "").strip()

    def pad4(s: str) -> str:
        return s + ("=" * ((4 - len(s) % 4) % 4))

    def b64_variants(s: str) -> list[bytes]:
        outs, seen = [], set()
        for v in [
            s,
            s.replace("-", "+").replace("_", "/"),
            s.replace(".", ""),
            s.replace(".", "+"),
            s.replace(".", "/"),
        ]:
            vv = pad4(v)
            if vv in seen:
                continue
            seen.add(vv)
            try:
                outs.append(base64.b64decode(vv))
            except Exception:
                pass
        return outs

    def hex_try(s: str) -> list[bytes]:
        try:
            return [bytes.fromhex(s)]
        except Exception:
            return []

    def raw_try(s: str) -> list[bytes]:
        return [s.encode("utf-8")] if s else []

    key_cands = b64_variants(enc_key) + hex_try(enc_key) + raw_try(enc_key)
    iv_cands  = b64_variants(iv_env)  + hex_try(iv_env)  + raw_try(iv_env)
    if not iv_cands:
        iv_cands = [b"\x00"*16]

    # EncSpec 기반 기본 비트수
    spec = (os.getenv("DATAHUB_ENC_SPEC", "") or "").strip().upper()
    default_bits = 256 if ("256" in spec or "AES256" in spec) else 128

    kb     = int(os.getenv("DATAHUB_FORCE_KEY_BITS", str(default_bits)) or str(default_bits))
    ivmode = (os.getenv("DATAHUB_FORCE_IV_MODE", "ENV") or "ENV").upper()
    kshape = (os.getenv("DATAHUB_FORCE_KEY_SHAPE", "right") or "right").lower()

    # 키 shaping
    def shape_key(k: bytes, bits: int, mode: str) -> bytes:
        need = 32 if bits == 256 else 16
        if mode == "right":
            return (k[:need]).ljust(need, b"\x00")
        if mode == "left":
            return (k[-need:]).rjust(need, b"\x00")
        if mode == "sha256":
            d = hashlib.sha256(k).digest()
            return d[:need]
        if mode == "md5":
            d = hashlib.md5(k).digest()
            return (d if need == 16 else (d + hashlib.md5(d).digest()))[:need]
        return (k[:need]).ljust(need, b"\x00")

    # IV shaping
    def shape_iv(v: bytes, mode: str) -> bytes:
        if mode == "ZERO":
            return b"\x00"*16
        vv = (v[:16]).ljust(16, b"\x00")
        return vv

    # (개발) SELFTEST로 올바른 key/iv 후보를 직접 선택
    plain  = (os.getenv("DATAHUB_SELFTEST_PLAIN", "") or "").strip()
    expect = (os.getenv("DATAHUB_SELFTEST_EXPECT", "") or "").strip()
    app_env = (os.getenv("APP_ENV", "dev") or "").strip().lower()
    text_enc = (os.getenv("DATAHUB_TEXT_ENCODING", "utf-8") or "").lower()
    text_enc = "cp949" if text_enc in ("cp949","euc-kr","euckr","ksc5601") else "utf-8"

    if app_env != "prod" and plain and expect:
        p = plain.encode(text_enc, errors="strict")
        data = _pkcs7_pad(p, 16)
        for i, kc in enumerate(key_cands):
            key = shape_key(kc, kb, kshape)
            for j, ic in enumerate(iv_cands):
                iv  = shape_iv(ic, ivmode)
                try:
                    ct = AES.new(key, AES.MODE_CBC, iv).encrypt(data)
                    b64 = base64.b64encode(ct).decode("ascii")
                    if b64 == expect:
                        print("[ENC][KIV][PICKED]", f"key_idx={i}", f"iv_idx={j}",
                              "key_bits=", kb, "key_len=", len(key),
                              "iv_len=", len(iv), "iv_src=", ivmode, "key_shape=", kshape)
                        return key, iv
                except Exception:
                    continue
        # 못 찾으면 아래 일반 경로로 진행

    # 일반 경로: 첫 후보 사용
    key = shape_key(key_cands[0] if key_cands else b"", kb, kshape)
    iv  = shape_iv(iv_cands[0] if iv_cands else b"\x00"*16, ivmode)

    print("[ENC][KIV]", "key_bits=", kb, "key_len=", len(key),
          "iv_len=", len(iv), "iv_src=", ivmode, "key_shape=", kshape)
    return key, iv


def encrypt_field(plain: str) -> str:
    """
    AES-CBC + PKCS7 → Base64
    - DATAHUB_TEXT_ENCODING (utf-8/cp949)
    - 평소엔 현재 설정값으로 1회 암호화
    - 다만 개발모드에서 DATAHUB_SELFTEST_EXPECT가 설정되어 있으면
      아래 조합을 자동탐색 후 '일치하는' 조합을 로그로 출력:
        * encoding: [현재설정, utf-8, cp949]
        * key_bits: [256, 128]
        * iv_mode : [ENV, ZERO]
    - 찾은 조합을 이후에도 동일하게 재사용할 수 있도록 로그 확인 후 ENV 고정 권장
    """
    expect = (os.getenv("DATAHUB_SELFTEST_EXPECT", "") or "").strip()
    app_env = (os.getenv("APP_ENV", "dev") or "").strip().lower()

    # 기본 인코딩(설정)
    def _norm_enc(name: str) -> str:
        n = (name or "utf-8").lower()
        return "cp949" if n in ("cp949","euc-kr","euckr","ksc5601") else "utf-8"

    enc_pref = _norm_enc(os.getenv("DATAHUB_TEXT_ENCODING", "utf-8"))
    enc_candidates = [enc_pref]
    for e in ("utf-8","cp949"):
        if e not in enc_candidates:
            enc_candidates.append(e)

    # key/iv 재구성 헬퍼 (key_bits/iv_mode에 따라)
    def _build_kiv(key_bits: int, iv_mode: str) -> Tuple[bytes, bytes]:
        # 원본 키/IV 디코드
        full_key, full_iv = _get_key_iv()  # full_key는 최대 32, full_iv는 16 보장됨
        if key_bits == 256:
            key = (full_key[:32]).ljust(32, b"\x00")
        else:
            key = (full_key[:16]).ljust(16, b"\x00")
        iv = (full_iv if iv_mode == "ENV" else b"\x00"*16)
        return key, iv

    # 단일 조합으로 실제 암호화
    def _enc_once(s: str, enc_name: str, key_bits: int, iv_mode: str) -> str:
        key, iv = _build_kiv(key_bits, iv_mode)
        data = s.encode(enc_name, errors="strict")
        data = _pkcs7_pad(data, 16)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        return base64.b64encode(cipher.encrypt(data)).decode("ascii")

    # 개발환경 + expect 설정 → 자동탐색
    if app_env != "prod" and expect:
        for enc_name in enc_candidates:
            for key_bits in (256, 128):
                for iv_mode in ("ENV", "ZERO"):
                    try:
                        ct = _enc_once(plain, enc_name, key_bits, iv_mode)
                    except Exception:
                        continue
                    if ct == expect:
                        # 🔍 정답 조합 로그 (꼭 확인해서 ENV로 고정해줘)
                        print("[ENC][SELFTEST][FINDER]",
                              f"encoding={enc_name} key_bits={key_bits} iv_mode={iv_mode}")
                        # 이후 동일 방식으로 암호화 결과 반환
                        return ct
        # 탐색 실패 시, 아래 일반 경로로 진행

    # 일반 경로: ENV 강제 설정값 사용
    chosen_enc = _get_text_encoding()  # utf-8 / cp949
    kb  = int(os.getenv("DATAHUB_FORCE_KEY_BITS", "256") or "256")
    ivm = (os.getenv("DATAHUB_FORCE_IV_MODE", "ENV") or "ENV").upper()
    print("[ENC][PLAINTEXT-ENCODING]", chosen_enc, "| key_bits=", kb, "| iv_mode=", ivm)
    # 실제 키/IV는 _get_key_iv()에서 강제 옵션 반영되어 생성됨
    key, iv = _get_key_iv()
    data = plain.encode(chosen_enc, errors="strict")
    data = _pkcs7_pad(data, 16)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return base64.b64encode(cipher.encrypt(data)).decode("ascii")




def _crypto_selftest():
    """
    포털에서 준 Plain/EncData 쌍으로 '정답'을 맞출 수 있는지 테스트.
    OK가 되어야 암호화 레이어 일치가 확정.
    """
    import os
    plain  = os.getenv("DATAHUB_SELFTEST_PLAIN", "").strip()
    expect = os.getenv("DATAHUB_SELFTEST_EXPECT", "").strip()

    if not plain or not expect:
        print("[ENC][SELFTEST] skipped (set DATAHUB_SELFTEST_PLAIN & DATAHUB_SELFTEST_EXPECT)")
        return

    try:
        got = encrypt_field(plain)
        ok  = (got == expect)
        print("[ENC][SELFTEST]", "OK" if ok else "FAIL", "| got=", got, "| expect=", expect)
    except Exception as e:
        print("[ENC][SELFTEST][ERR]", repr(e))



class DatahubClient:
    def __init__(self, base: Optional[str] = None, token: Optional[str] = None):
        raw_base = base or os.getenv("DATAHUB_API_BASE", "https://datahub-dev.scraping.co.kr")
        self.base = (raw_base or "").strip().rstrip("/")
        self.token = (token or os.getenv("DATAHUB_TOKEN", "")).strip()

        # 🔍 init 진입 로그 (selftest 실행 조건/ENV 상태 확인)
        app_env   = (os.getenv("APP_ENV", "dev") or "").strip().lower()
        st_flag   = (os.getenv("DATAHUB_SELFTEST", "1") or "").strip()
        st_plain  = os.getenv("DATAHUB_SELFTEST_PLAIN", "")
        st_expect = os.getenv("DATAHUB_SELFTEST_EXPECT", "")
        print("[ENC][INIT]",
              "base=", repr(self.base),
              "app_env=", app_env,
              "selftest_flag=", st_flag,
              "plain_set=", bool(st_plain),
              "expect_set=", bool(st_expect))

        # ✅ 토큰 유무와 관계없이 selftest 먼저 실행
        if app_env != "prod" and st_flag == "1":
            _crypto_selftest()

        # 이후 토큰 검증
        if not self.token:
            raise DatahubError("DATAHUB_TOKEN is missing")
            
    def _post(self, path: str, body: Dict[str, Any], timeout=25) -> Dict[str, Any]:
        url = f"{self.base}{path}"
        headers = {
            "Authorization": f"Token {self.token}",
            "Content-Type": "application/json;charset=UTF-8",
        }
        try:
            r = requests.post(url, headers=headers, json=body, timeout=timeout)
        except Exception as e:
            print("[DATAHUB][ERR-REQ]", path, repr(e))
            raise DatahubError(f"REQUEST_ERROR: {e}")

        try:
            data = r.json()
        except Exception:
            data = {"errCode": "HTTP", "errMsg": r.text, "result": "FAIL"}

        # 🔍 요청/응답 요약을 '항상' 먼저 기록
        try:
            print("[DATAHUB][BASE]", repr(self.base))
            print("[DATAHUB][URL ]", url)
            print("[DATAHUB][REQ ]", path, list(body.keys()))
            print("[DATAHUB][RSP-STATUS]", r.status_code)
            print("[DATAHUB][RSP-SHORT]", data.get("errCode"), data.get("result"), (data.get("errMsg") or "")[:200])
        except Exception:
            pass

        if r.status_code != 200:
            raise DatahubError(f"HTTP {r.status_code}: {data}")

        return data
    
    def medical_checkup_simple(
        self,
        login_option: str,
        user_name: str,
        hp_number: str,
        jumin_or_birth: str,
        telecom_gubun: str | None = None,
        callback_id: Optional[str] = None,
    ) -> dict:
        body = {
            "LOGINOPTION": login_option,
            "USERNAME": user_name,
            "HPNUMBER": hp_number,
            "JUMIN": encrypt_field(jumin_or_birth),
        }
        if login_option == "3" and telecom_gubun:
            body["TELECOMGUBUN"] = telecom_gubun
        
        if body["LOGINOPTION"] == "3":
            if telecom_gubun not in {"1","2","3"}:
                raise DatahubError("통신사 간편인증은 TELECOMGUBUN(1/2/3)이 필요합니다.")
            body["TELECOMGUBUN"] = telecom_gubun
        if callback_id:
            body["CALLBACKID"] = callback_id  # ★ 기존 인증 세션을 이어받아 결과 조회

        return self._post("/scrap/common/nhis/MedicalCheckupGlanceSimple", body)


    # === 1) 간편인증 Step1: 시작/즉시조회 ===
    def simple_auth_start(
        self,
        login_option: str,
        user_name: str,
        hp_number: str,
        jumin_or_birth: str,
        telecom_gubun: str | None = None,
    ) -> Dict[str, Any]:
        body = {
            "LOGINOPTION": login_option,
            "USERNAME": user_name,
            "HPNUMBER": hp_number,
            "JUMIN": encrypt_field(jumin_or_birth),
        }
        if login_option == "3" and telecom_gubun:
            body["TELECOMGUBUN"] = telecom_gubun
        return self._post("/scrap/common/nhis/MedicalCheckupGlanceSimple", body)



    # === 1-2) 콜백ID로만 결과 재조회 (새 인증 X) ===
    def post_medical_glance_simple_with_callbackid(self, callbackId: str) -> Dict[str, Any]:
        return self._post("/scrap/common/nhis/MedicalCheckupGlanceSimple", {"CALLBACKID": callbackId})



    # --- 2) 간편인증 Step2: captcha(최종 완료 콜)
    def simple_auth_complete(self, callback_id: str, callback_type: str = "SIMPLE") -> Dict[str, Any]:
        body = {
            "callbackId": callback_id,
            "callbackType": callback_type or "SIMPLE",
        }
        return self._post("/scrap/captcha", body)


    # --- 3) 인증서 방식: 건강검진 결과 조회
    def nhis_medical_checkup(self, jumin: str, cert_name: str, cert_pwd: str, der_b64: str, key_b64: str) -> Dict[str, Any]:
        body = {
            "JUMIN": encrypt_field(jumin),         # 13자리
            "P_CERTNAME": cert_name,               # cn=... 문자열
            "P_CERTPWD": encrypt_field(cert_pwd),  # 암호화 TRUE
            "P_SIGNCERT_DER": der_b64,             # BASE64
            "P_SIGNPRI_KEY": key_b64,              # BASE64
        }
        return self._post("/scrap/common/nhis/MedicalCheckupResult", body)


def pick_latest_general(resp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    DataHub '건강검진결과 한눈에 보기' 응답에서 '최근 10년 중 가장 최신 1건'만 골라
    서비스 표준 키로 변환해 반환.
    기대: resp["data"]["INCOMELIST"] = [ {...}, ... ]  (키 대소/케이스 다양성 방어)
    - 연도(GUNYEAR): "YYYY" 또는 "YYYY년" 등 문자열
    - 일자(GUNDATE): "MM/DD" 또는 "YYYYMMDD" 혹은 "YYYY.MM.DD" / "YYYY-MM-DD"
    - 일부 응답엔 CheckUpDate: "YYYYMMDD"
    """

    def _s(x) -> str:
        return str(x or "").strip()

    def _norm_year(y: str) -> str:
        y = _s(y)
        if y.endswith("년"):
            y = y[:-1].strip()
        # 숫자만 추출(문자 섞여 있어도 첫 숫자그룹 사용)
        nums = re.findall(r"\d{4}", y)
        return nums[0] if nums else ""

    def _norm_gundate(gy: str, gd: str) -> str:
        """
        gy: GUNYEAR ("YYYY" 혹은 "YYYY년")
        gd: GUNDATE ("MM/DD" or "YYYYMMDD" or "YYYY.MM.DD" or "YYYY-MM-DD")
        -> "YYYY-MM-DD" 반환(실패 시 빈 문자열)
        """
        y = _norm_year(gy)
        s = _s(gd).replace(".", "/").replace("-", "/")

        # 케이스1: YYYYMMDD 순수 숫자 8자리
        only_digits = re.fullmatch(r"\d{8}", s)
        if only_digits:
            yy, mm, dd = s[:4], s[4:6], s[6:8]
            return f"{yy}-{mm}-{dd}"

        # 케이스2: 슬래시 구분
        parts = [p for p in s.split("/") if p]
        if len(parts) == 2:
            # MM/DD + 연도는 gy에서
            mm = parts[0].zfill(2)
            dd = parts[1].zfill(2)
            if y:
                return f"{y}-{mm}-{dd}"
            return ""  # 연도가 없으면 불가
        if len(parts) == 3:
            # YYYY/MM/DD 또는 YY/MM/DD 등 → 앞은 연도로 취급
            yy = parts[0].zfill(4)
            mm = parts[1].zfill(2)
            dd = parts[2].zfill(2)
            return f"{yy}-{mm}-{dd}"

        return ""

    # 1) 리스트 위치/이름 방어
    data = (resp or {}).get("data") or resp.get("Data") or {}
    items: List[Dict[str, Any]] = (
        data.get("INCOMELIST")
        or data.get("incomeList")
        or []
    )
    if not isinstance(items, list) or not items:
        return None

    # 2) 날짜 파싱 → 최신 1건 선별
    def to_iso_date(it: Dict[str, Any]) -> str:
        gy = _s(it.get("GUNYEAR"))
        gd = _s(it.get("GUNDATE"))
        iso = _norm_gundate(gy, gd)
        if not iso:
            # 백업: CheckUpDate가 "YYYYMMDD"로 올 수 있음
            cud = _s(it.get("CheckUpDate"))
            if re.fullmatch(r"\d{8}", cud):
                iso = f"{cud[:4]}-{cud[4:6]}-{cud[6:8]}"
        if iso:
            return iso
        # 정말 아무것도 없으면 연말 fallback(연도만 있을 때도 UI 정렬을 위해)
        y = _norm_year(gy)
        return f"{y}-12-31" if y else "0000-01-01"

    ranked = sorted(items, key=to_iso_date, reverse=True)
    top = ranked[0]
    iso_date = to_iso_date(top)

    # 3) 표준 키 매핑 (모두 문자열 타입 가정)
    def _sv(key: str) -> str:
        return _s(top.get(key))

    mapped = {
        "exam_date": iso_date,              # YYYY-MM-DD
        "exam_year": _s(top.get("GUNYEAR")),
        "exam_place": _sv("GUNPLACE"),
        "height": _sv("HEIGHT"),
        "weight": _sv("WEIGHT"),
        "bmi": _sv("BODYMASS"),
        "bp": _sv("BLOODPRESS"),            # "120/80"
        "vision": _sv("SIGHT"),
        "hearing": _sv("HEARING"),
        "hemoglobin": _sv("HEMOGLOBIN"),
        "fbs": _sv("BLOODSUGAR"),           # 공복혈당
        "tc": _sv("TOTCHOLESTEROL"),
        "hdl": _sv("HDLCHOLESTEROL"),
        "ldl": _sv("LDLCHOLESTEROL"),
        "tg": _sv("TRIGLYCERIDE"),
        "gfr": _sv("GFR"),
        "creatinine": _sv("SERUMCREATININE"),
        "ast": _sv("SGOT"),
        "alt": _sv("SGPT"),
        "ggt": _sv("YGPT"),
        "urine_protein": _sv("YODANBAK"),
        "chest": _sv("CHESTTROUBLE"),
        "judgment": _sv("JUDGMENT"),
        "_raw": top,  # 디버깅용(필요 없으면 제거 가능)
    }
    return mapped
