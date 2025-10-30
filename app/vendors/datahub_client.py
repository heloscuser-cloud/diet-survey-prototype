import os, json, base64, hashlib
from typing import Any, Dict, Optional, Tuple
import requests

from Crypto.Cipher import AES

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

def _get_key_iv() -> Tuple[bytes, Optional[bytes]]:
    """
    EncKey / EncIV 다양한 포맷을 모두 허용:
    - 점(.) 섞인 형태 (예: 'Qt.P5OVv/DQDHbvAo.zelQ99tsPKzhJ4')
    - Base64 / base64url(-,_) / Hex / Plain
    - IV는 최종 16바이트로 보정
    우선순위:
      1) EncSpec에 IV=... 가 있으면 그 값을 최우선 사용
      2) 아니면 ENV(DATAHUB_ENC_IV_B64)
      3) 없으면 0x00 * 16
    """
    enc_key = (os.getenv("DATAHUB_ENC_KEY_B64", "") or "").strip()
    iv_env  = (os.getenv("DATAHUB_ENC_IV_B64", "") or "").strip()

    def _normalize_b64(s: str) -> str:
        # 공백 제거 + 점(.) 제거 + base64url 호환 변환
        t = (s or "").strip().replace(" ", "").replace(".", "")
        t = t.replace("-", "+").replace("_", "/")
        # 길이 4의 배수 패딩
        pad = (-len(t)) % 4
        if pad:
            t = t + ("=" * pad)
        return t

    def _try_many_formats(s: str, want_len: int | None = None) -> Optional[bytes]:
        if not s:
            return None
        # 1) normalized base64
        try:
            t = _normalize_b64(s)
            b = base64.b64decode(t)
            if (want_len is None) or (len(b) == want_len) or (want_len in (16, 24, 32) and len(b) in (16, 24, 32)):
                return b
        except Exception:
            pass
        # 2) hex
        try:
            b = bytes.fromhex(s)
            if (want_len is None) or len(b) == want_len:
                return b
        except Exception:
            pass
        # 3) plain utf-8
        try:
            b = s.encode("utf-8")
            if (want_len is None) or len(b) == want_len:
                return b
        except Exception:
            pass
        return None

    # --- KEY ---
    key_bytes = _try_many_formats(enc_key)
    if key_bytes is None:
        key_bytes = b""

    # EncSpec 파싱
    spec = (os.getenv("DATAHUB_ENC_SPEC", "") or "").strip().upper()
    algo = spec.split("/")[0] if "/" in spec else spec
    # '.../256' 같이 뒤에 비트수가 오는 형태도 지원
    if ("256" in spec) or ("AES256" in algo):
        key_bytes = (key_bytes[:32]).ljust(32, b"\x00")
    elif ("128" in spec) or ("AES128" in algo) or ("AES" in algo):
        key_bytes = (key_bytes[:16]).ljust(16, b"\x00")
    else:
        # 기본 32바이트로 보정
        key_bytes = (key_bytes[:32]).ljust(32, b"\x00")


    # --- IV ---
    iv: Optional[bytes] = None
    if "IV=" in spec:
        iv_str = spec.split("IV=")[1].strip()
        iv = _try_many_formats(iv_str, want_len=16)
    else:
        iv = _try_many_formats(iv_env, want_len=16)

    if iv is None:
        iv = b"\x00" * 16
    elif len(iv) != 16:
        iv = (iv[:16]).ljust(16, b"\x00")

    return key_bytes, iv


def encrypt_field(plain: str) -> str:
    """
    EncSpec/EncKey/IV에 따라 AES-CBC(+PKCS7/PKCS5)로 암호화 후 Base64 리턴.
    - EncSpec: DATAHUB_ENC_SPEC (예: 'AES/CBC/PKCS5PADDING/256' 또는 'AES256/CBC/PKCS7')
    """
    spec = (os.getenv("DATAHUB_ENC_SPEC", "") or "").upper()
    # PKCS5PADDING을 PKCS7로 동일 취급
    spec_normalized = spec.replace("PKCS5PADDING", "PKCS7")
    # 키/IV 획득(IV는 _get_key_iv 내부에서 EncSpec의 IV=... 또는 ENV를 우선 처리)
    key, iv = _get_key_iv()  # ← ★ iv_env 같은 이름 쓰지 않음!

    print("[ENC][SPEC]", spec, "| key_len=", len(key), "| iv_len=", len(iv))  # 로그

    # 블록/패딩
    block = 16
    data = plain.encode("utf-8")
    if "PKCS7" in spec_normalized:
        data = _pkcs7_pad(data, block)

    # 암호화
    cipher = AES.new(key, AES.MODE_CBC, iv)
    enc = cipher.encrypt(data)
    return base64.b64encode(enc).decode("ascii")


def _crypto_selftest():
    """
    공급사 포털에서 제공한 Plain/EncData 쌍으로 즉시 판정.
    - DATAHUB_SELFTEST_PLAIN : 포털 PlainData (예: !Helo999어드민)
    - DATAHUB_SELFTEST_EXPECT: 포털 EncData   (예: oXCcQ5Z0iINu+9Oi0u5/... )
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
        if not self.token:
            # __init__ 끝나기 직전(마지막 return/raise 전에) 추가
            if os.getenv("APP_ENV", "dev") != "prod" and os.getenv("DATAHUB_SELFTEST", "1") == "1":
                _crypto_selftest()
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
    

    def simple_auth_start(
        self,
        login_option: str,      # "0"=카카오, "1"=삼성, "2"=페이코, "3"=통신사, "4"=KB, "5"=네이버, "6"=신한, "7"=토스
        user_name: str,
        hp_number: str,         # "01012341234" 또는 "010-1234-1234" 모두 허용
        jumin_or_birth: str,    # yyyyMMdd (가이드 문서에서 JUMIN이 '생년월일'로 정의)
        telecom: str = ""       # "1"(SKT) / "2"(KT) / "3"(LGU+) - 통신사 인증 선택時 필수
    ) -> Dict[str, Any]:
        """
        건강보험_건강검진결과 한눈에보기(간편인증)
        POST /scrap/common/nhis/MedicalCheckupGlanceSimple
        필드: LOGINOPTION, JUMIN(암호화), USERNAME, HPNUMBER, TELECOMGUBUN
        """
        # 하이픈 허용
        hp = hp_number.strip()

        # 통신사 코드 보정: 영문 입력이 들어왔을 때 숫자코드로 치환
        tel = (telecom or "").strip().upper()
        if tel in ("SKT", "S", "SK"): tel = "1"
        elif tel in ("KT",):           tel = "2"
        elif tel in ("LGU", "LGU+", "L"): tel = "3"


        # 디버그: JUMIN 암호문 일부만 노출(앞 6글자만) 로그 꼭 지우기 ################
        try:
            _tmp_ct = encrypt_field(jumin_or_birth)
            print("[ENC][JUMIN][LEN]", len(_tmp_ct), "| head=", _tmp_ct[:6], "***")
        except Exception as _e:
            print("[ENC][JUMIN][ERR]", repr(_e))


        payload = {
            "LOGINOPTION": str(login_option).strip(),
            "JUMIN":       encrypt_field(jumin_or_birth.strip()),  # ★ 가이드상 암호화 필수
            "USERNAME":    user_name.strip(),
            "HPNUMBER":    hp,
            # TELECOMGUBUN은 통신사(=LOGINOPTION "3")일 때만 포함
        }
        if payload["LOGINOPTION"] == "3":
            if tel not in {"1","2","3"}:
                raise DatahubError("통신사 간편인증은 TELECOMGUBUN(1/2/3)이 필요합니다.")
            payload["TELECOMGUBUN"] = tel

        # ★ 가이드에 나온 ‘정식’ 경로로 호출
        return self._post("/scrap/common/nhis/MedicalCheckupGlanceSimple", payload)


    # --- 2) 간편인증 Step2: captcha(최종 완료 콜)
    def simple_auth_complete(self, callback_id: str, callback_type: str = "SIMPLE", **kwargs) -> Dict[str, Any]:
        body = {
            "callbackId": callback_id,
            "callbackType": callback_type,
            # callbackResponse / 1 / 2 / retry 등은 필요시에만
        }
        body.update({k:v for k,v in kwargs.items() if v is not None})
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

def pick_latest_general(datahub_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    간편인증 응답(data.INCOMELIST[])에서 가장 최신 1건만 { exam_date, items, raw }로 정규화
    - GUNYEAR: "2022"
    - GUNDATE: "11/02"
    - 기타 가공은 필요 시 확장
    """
    data = (datahub_response or {}).get("data") or {}
    rows = data.get("INCOMELIST") or []

    # 날짜 정렬 키 만들기 (YYYYMMDD)
    def yyyymmdd(r):
        y = str(r.get("GUNYEAR") or "").strip()
        md = str(r.get("GUNDATE") or "").strip()  # "MM/DD"
        mm, dd = "01", "01"
        if "/" in md:
            parts = md.split("/")
            if len(parts) == 2:
                mm = parts[0].zfill(2)
                dd = parts[1].zfill(2)
        return (y + mm + dd) if len(y) == 4 else ""

    rows.sort(key=yyyymmdd, reverse=True)
    latest = rows[0] if rows else None
    if not latest:
        return {"exam_date": "", "items": [], "raw": datahub_response}

    exam_date = latest.get("GUNYEAR","") + "-" + (latest.get("GUNDATE","").replace("/", "-"))
    return {"exam_date": exam_date, "items": [], "raw": latest}

