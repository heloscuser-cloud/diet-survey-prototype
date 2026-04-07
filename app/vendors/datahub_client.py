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

    # 환경 + expect 설정 → 자동탐색
    if app_env != "dev" and expect:
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


def pick_latest_general(resp: dict, mode: str = "latest"):
    """
    mode:
    - "latest": 기존 동작 (가장 최근 1건)
    - "all":    필터 없이 INCOMELIST 전체 반환 (연도 내림차순 정렬 시도)
    """
    data = (resp or {}).get("data") or {}
    income = data.get("INCOMELIST") or data.get("INCOME_LIST") or []

    # === 전체 그대로 반환 (테스트/진단용) ===
    if mode == "all":
        def _year_of(row):
            for k in ("EXAMYEAR","GUNYEAR","YEAR","YY"):
                v = row.get(k)
                if isinstance(v, str) and v.isdigit(): return int(v)
                if isinstance(v, int): return v
            for k in ("EXAMDATE","EXAM_DATE","검진일자","exam_date"):
                v = row.get(k)
                if isinstance(v, str) and len(v) >= 4 and v[:4].isdigit():
                    return int(v[:4])
            return -1
        items = list(income) if isinstance(income, list) else []
        try:
            items.sort(key=_year_of, reverse=True)
        except Exception:
            pass
        return {"items": items, "count": len(items), "keys": list(data.keys())}

    # === 최근 1건 선택 ===
    latest = None
    best_year = -1
    def _year_of(row):
        for k in ("EXAMYEAR","GUNYEAR","YEAR","YY"):
            v = row.get(k)
            if isinstance(v, str) and v.isdigit(): return int(v)
            if isinstance(v, int): return v
        for k in ("EXAMDATE","EXAM_DATE","검진일자","exam_date"):
            v = row.get(k)
            if isinstance(v, str) and len(v) >= 4 and v[:4].isdigit():
                return int(v[:4])
        return -1

    if isinstance(income, list):
        for row in income:
            yr = _year_of(row)
            if yr > best_year:
                best_year = yr
                latest = row

    return latest or {}


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
 
    def _post(self, path: str, body: Dict[str, Any], timeout: Tuple[int, int] = (5, 25)) -> Dict[str, Any]:
        """
        DataHub 공통 POST 함수
        - timeout: (connect, read)
          기본 (5, 25) / Step2(captcha)는 (5, 5) 권장
        """
        url = f"{self.base.rstrip('/')}{path}"
        headers = {
            "Authorization": f"Token {self.token}",
            "Content-Type": "application/json;charset=UTF-8",
        }

        try:
            # 요청 로그
            print("[DATAHUB][BASE]", repr(self.base))
            print("[DATAHUB][URL ]", url)
            print("[DATAHUB][REQ ]", path, list(body.keys()))

            # 요청 실행 (timeout 튜플 적용)
            r = requests.post(url, headers=headers, json=body, timeout=timeout)
        except Exception as e:
            print("[DATAHUB][ERR-REQ]", path, repr(e))
            # timeout 등 예외는 DatahubError로 감싸서 상위에서 처리
            raise DatahubError(f"REQUEST_ERROR: {e}")

        try:
            data = r.json()
        except Exception:
            data = {"errCode": "HTTP", "errMsg": r.text, "result": "FAIL"}

        # 응답 요약 로그
        try:
            print("[DATAHUB][RSP-STATUS]", r.status_code)
            print("[DATAHUB][RSP-SHORT]",
                  data.get("errCode"), data.get("result"), (data.get("errMsg") or "")[:200])
        except Exception:
            pass

        if r.status_code != 200:
            raise DatahubError(f"HTTP {r.status_code}: {data}")

        return data

 
 
    def medical_checkup_simple(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # payload 예: {"CALLBACKID": "...", "CALLBACKTYPE": "SIMPLE"}
        return self._post("/scrap/common/nhis/MedicalCheckupGlanceSimple", payload, timeout=(5,25))


    # === 1) 간편인증 Step1: 시작/즉시조회 ===
    def simple_auth_start(self, login_option, user_name, hp_number, jumin_or_birth, telecom_gubun=None):
        body = {
            "LOGINOPTION":  str(login_option or "0"),
            "USERNAME":     user_name,
            "HPNUMBER":     hp_number,
            "JUMIN":        encrypt_field(jumin_or_birth),
        }
        if str(login_option) == "3" and telecom_gubun:
            body["TELECOMGUBUN"] = telecom_gubun  # 1~6 숫자코드
        return self._post("/scrap/common/nhis/MedicalCheckupGlanceSimple", body, timeout=(5,25))



    # --- 2) 간편인증 Step2: captcha(최종 완료 콜)
    def simple_auth_complete(
        self,
        callback_id: str,
        callback_type: str = "SIMPLE",
        callbackResponse: str = "",
        callbackResponse1: str = "",
        callbackResponse2: str = "",
        retry: str = "",
    ) -> Dict[str, Any]:
        """
        Step2 - 간편인증 완료 확인 (/scrap/captcha)
        ※ callbackResponse* 키들은 비어 있어도 반드시 포함해야 함.
        ※ timeout=(5,5): 5초 내 미응답이면 상위에서 폴링으로 진행.
        """
        body = {
            "callbackId": callback_id,
            "callbackType": callback_type or "SIMPLE",
            "callbackResponse": callbackResponse or "",
            "callbackResponse1": callbackResponse1 or "",
            "callbackResponse2": callbackResponse2 or "",
            "retry": retry or "",
        }

        # Step2는 짧은 읽기 타임아웃
        return self._post("/scrap/captcha", body, timeout=(5, 5))



    # --- 3) 인증서 방식: 건강검진 결과 조회
    def nhis_medical_checkup(self, jumin: str, cert_name: str, cert_pwd: str, der_b64: str, key_b64: str) -> Dict[str, Any]:
        body = {
            "JUMIN": encrypt_field(jumin),         # 8자리
            "P_CERTNAME": cert_name,               # cn=... 문자열
            "P_CERTPWD": encrypt_field(cert_pwd),  # 암호화 TRUE
            "P_SIGNCERT_DER": der_b64,             # BASE64
            "P_SIGNPRI_KEY": key_b64,              # BASE64
        }
        return self._post("/scrap/common/nhis/MedicalCheckupResult", body)


    # 보강 재조회
    def medical_checkup_simple_with_identity(
        self,
        callback_id: str,
        callback_type: str,             # "SIMPLE"
        login_option: str,
        user_name: str,
        hp_number: str,
        jumin_or_birth: str,            # 8자리 YYYYMMDD
        telecom_gubun: Optional[str] = None,  # PASS(3)일 때만 "1"~"6"
    ) -> Dict[str, Any]:
        body = {
            "CALLBACKID":   callback_id,
            "CALLBACKTYPE": (callback_type or "SIMPLE"),
            "LOGINOPTION":  str(login_option),
            "USERNAME":     user_name,
            "HPNUMBER":     hp_number,
            "JUMIN":        encrypt_field(jumin_or_birth),  # ★ 암호화 필수
        }
        if str(login_option) == "3" and telecom_gubun:
            body["TELECOMGUBUN"] = telecom_gubun
        return self._post("/scrap/common/nhis/MedicalCheckupGlanceSimple", body, timeout=(5,25))
