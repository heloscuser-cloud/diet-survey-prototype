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
        try:
            r = requests.post(url, headers=headers, json=body, timeout=timeout)
        except Exception as e:
            # 네트워크 예외 자체도 남겨두자
            print("[DATAHUB][ERR-REQ]", path, repr(e))
            raise DatahubError(f"REQUEST_ERROR: {e}")

        # 응답 본문 파싱 시도
        try:
            data = r.json()
        except Exception:
            data = {"errCode": "HTTP", "errMsg": r.text, "result": "FAIL"}

        # 🔍 요청/응답 로그를 '무조건' 먼저 찍는다.
        try:
            # body는 민감값(암호화 후)이긴 하지만 키만 남기자
            print("[DATAHUB][REQ]", path, list(body.keys()))
            print("[DATAHUB][RSP-STATUS]", r.status_code)
            # errCode / result / errMsg 요약
            print("[DATAHUB][RSP-SHORT]", data.get("errCode"), data.get("result"), (data.get("errMsg") or "")[:200])
        except Exception:
            pass

        # 여기서 비정상 상태코드면 그 다음 raise
        if r.status_code != 200:
            # 본문도 같이 남겨 원인 추적
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

