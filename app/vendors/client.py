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
    EncKey는 Base64/Hex/Plain 중 하나라고 가정.
    IV는 EncSpec에 있거나, ENV(DATAHUB_ENC_IV_B64)로 대체.
    """
    enc_key = os.getenv("DATAHUB_ENC_KEY_B64", "")
    key_bytes = None
    # Base64 시도
    try:
        key_bytes = base64.b64decode(enc_key)
    except Exception:
        key_bytes = None
    # Hex 시도
    if key_bytes is None:
        try:
            key_bytes = bytes.fromhex(enc_key)
        except Exception:
            key_bytes = None
    # Plain
    if key_bytes is None:
        key_bytes = enc_key.encode("utf-8")

    iv_b64 = os.getenv("DATAHUB_ENC_IV_B64")
    iv = base64.b64decode(iv_b64) if iv_b64 else None
    return key_bytes, iv

def encrypt_field(plain: str) -> str:
    """
    EncSpec/EncKey에 따라 AES-CBC(+PKCS7)로 암호화 후 Base64 리턴.
    - EncSpec: DATAHUB_ENC_SPEC
    - EncKey : DATAHUB_ENC_KEY_B64
    - IV     : EncSpec에 명시되어 있지 않으면 DATAHUB_ENC_IV_B64 사용(없으면 0-IV)
    """
    spec = os.getenv("DATAHUB_ENC_SPEC", "")
    algo, mode, padding = _parse_encspec(spec)
    key, iv_env = _get_key_iv()

    if "AES256" in algo:
        key = (key or b"")[:32].ljust(32, b"\x00")
        block = 16
    elif "AES128" in algo or "AES" in algo:
        key = (key or b"")[:16].ljust(16, b"\x00")
        block = 16
    else:
        raise DatahubError(f"Unsupported algo in EncSpec: {algo}")

    # IV 결정: EncSpec에 IV=... 가 포함되어 있으면 우선
    iv = iv_env
    if "IV=" in spec:
        try:
            iv_str = spec.split("IV=")[1]
            # hex or 0-filled
            if len(iv_str) in (16, 32):
                # ASCII 그대로일 수 있으므로 zero-pad to 16
                iv = iv_str.encode("utf-8").ljust(16, b"\x00")
            else:
                # 32/64 hex일 수 있음
                try:
                    iv = bytes.fromhex(iv_str)
                except Exception:
                    pass
        except Exception:
            pass
    if iv is None:
        iv = b"\x00"*16

    data = plain.encode("utf-8")
    if "PKCS7" in padding:
        data = _pkcs7_pad(data, block)

    cipher = AES.new(key, AES.MODE_CBC, iv)
    enc = cipher.encrypt(data)
    return base64.b64encode(enc).decode("ascii")

class DatahubClient:
    def __init__(self, base: Optional[str] = None, token: Optional[str] = None):
        self.base = (base or os.getenv("DATAHUB_API_BASE", "https://datahub-dev.scraping.co.kr")).rstrip("/")
        self.token = token or os.getenv("DATAHUB_TOKEN", "")
        if not self.token:
            raise DatahubError("DATAHUB_TOKEN is missing")

    def _post(self, path: str, body: Dict[str, Any], timeout=25) -> Dict[str, Any]:
        url = f"{self.base}{path}"
        headers = {
            "Authorization": f"Token {self.token}",
            "Content-Type": "application/json;charset=UTF-8",
        }
        r = requests.post(url, headers=headers, json=body, timeout=timeout)
        try:
            data = r.json()
        except Exception:
            data = {"errCode": "HTTP", "errMsg": r.text, "result": "FAIL"}
        if r.status_code != 200:
            raise DatahubError(f"HTTP {r.status_code}: {data}")
        return data

    # --- 1) 간편인증 Step1: 로그인 옵션에 따라 요청
    def simple_auth_start(self, login_option: str, user_name: str, hp_number: str, jumin_or_birth: str, telecom: str = "") -> Dict[str, Any]:
        """
        /scrap/${*Simple} (문서 placeholder) → 실제 경로는 공급사 안내 값 사용
        필드(문서 예): LOGINOPTION(0=카카오), TELECOMGUBUN(PASS時), HPNUMBER, USERNAME, JUMINNUM(암호화)
        """
        payload = {
            "LOGINOPTION": login_option,
            "TELECOMGUBUN": telecom or "",
            "HPNUMBER": hp_number,
            "USERNAME": user_name,
            "JUMINNUM": encrypt_field(jumin_or_birth),  # 생년월일 8자리 혹은 주민번호 13자리
        }
        return self._post("/scrap/simple", payload)  # 실제 경로는 공급사에서 안내한 값으로 변경 필요

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
    데이터허브 응답에서 '일반' 검진 중 가장 최근 1건만 추려 { exam_date, items[], raw } 형태로 반환.
    (문서: data.CHECKUPLIST[].CHECKUPKIND/DATE/YEAR/OPINION/ORGAN ...)
    """
    data = datahub_response or {}
    data_part = data.get("data") or {}
    rows = data_part.get("CHECKUPLIST") or []
    rows = [r for r in rows if (str(r.get("CHECKUPKIND","")).strip() == "일반")]
    # 최신 정렬: CHECKUPDATE(YYYYMMDD) or CHECKUPYEAR
    def row_date(r):
        d = str(r.get("CHECKUPDATE") or "")
        if len(d) == 8:
            return d
        y = str(r.get("CHECKUPYEAR") or "")
        return (y + "0101") if len(y) == 4 else ""
    rows.sort(key=lambda r: row_date(r), reverse=True)
    latest = rows[0] if rows else None
    if not latest:
        return {"exam_date": "", "items": [], "raw": datahub_response}
    # DataHub 기본 응답엔 세부 항목 리스트가 없으니 원문 그대로 raw 로 보관
    exam_date = latest.get("CHECKUPDATE") or ""
    items = []  # 필요 시 추가 API/확장 항목 붙일 수 있음
    return {"exam_date": exam_date, "items": items, "raw": latest}
