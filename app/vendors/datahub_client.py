import os, json, base64, hashlib
from typing import Any, Dict, Optional, Tuple
import requests

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
    EncKey / EncIV 다양한 포맷 허용 & 자동탐색:
    - '.'(dot) 포함 커스텀 b64 → (제거|'.'→'+'|'.'→'/') 모두 시도
    - base64url(-,_) 보정
    - Base64 → Hex → Plain 순으로 시도
    - IV는 최종 16바이트로 보정
    우선순위: EncSpec의 IV=... > ENV(DATAHUB_ENC_IV_B64) > 0x00*16
    """
    enc_key = (os.getenv("DATAHUB_ENC_KEY_B64", "") or "").strip()
    iv_env  = (os.getenv("DATAHUB_ENC_IV_B64", "") or "").strip()

    def _pad4(t: str) -> str:
        pad = (-len(t)) % 4
        return t + ("=" * pad)

    def _try_b64_variants(s: str) -> Optional[bytes]:
        if not s:
            return None
        raw = (s or "").strip().replace(" ", "")
        # base64url 정규화 1차
        bases = [raw,
                 raw.replace("-", "+").replace("_", "/"),
                 raw.replace(".", ""),                # dot 제거
                 raw.replace(".", "+"),               # dot→'+'
                 raw.replace(".", "/"),               # dot→'/'
                 raw.replace(".", "").replace("-", "+").replace("_", "/"),
                 raw.replace(".", "+").replace("-", "+").replace("_", "/"),
                 raw.replace(".", "/").replace("-", "+").replace("_", "/"),
                ]
        seen = set()
        for b in bases:
            if b in seen: 
                continue
            seen.add(b)
            try:
                candidate = base64.b64decode(_pad4(b))
                return candidate
            except Exception:
                continue
        return None

    def _decode_any(s: str) -> Optional[bytes]:
        # 1) 다양한 b64 변형 시도
        b = _try_b64_variants(s)
        if b is not None:
            return b
        # 2) hex
        try:
            return bytes.fromhex(s)
        except Exception:
            pass
        # 3) plain
        try:
            return s.encode("utf-8")
        except Exception:
            return None

    # --- KEY ---
    key_bytes = _decode_any(enc_key) or b""

    # EncSpec 파싱 & 키 길이 보정
    spec = (os.getenv("DATAHUB_ENC_SPEC", "") or "").strip().upper()
    algo = spec.split("/")[0] if "/" in spec else spec
    # 256/128 표기를 모두 인식
    if ("256" in spec) or ("AES256" in algo):
        key_bytes = (key_bytes[:32]).ljust(32, b"\x00")
        key_bits = 256
    elif ("128" in spec) or ("AES128" in algo) or ("AES" in algo):
        key_bytes = (key_bytes[:16]).ljust(16, b"\x00")
        key_bits = 128
    else:
        key_bytes = (key_bytes[:32]).ljust(32, b"\x00")
        key_bits = 256

    # --- IV ---
    iv: Optional[bytes] = None
    iv_source = "ZERO"
    if "IV=" in spec:
        iv_str = spec.split("IV=")[1].strip()
        iv = _decode_any(iv_str)
        iv_source = "SPEC"
    elif iv_env:
        iv = _decode_any(iv_env)
        iv_source = "ENV"

    if not iv:
        iv = b"\x00" * 16
    elif len(iv) != 16:
        iv = (iv[:16]).ljust(16, b"\x00")

    # 🔍 어떤 방식으로 잡혔는지 요약 로그
    try:
        print("[ENC][KIV]",
              "key_bits=", key_bits,
              "key_len=", len(key_bytes),
              "iv_len=", len(iv),
              "iv_src=", iv_source)
    except Exception:
        pass

    return key_bytes, iv

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

    # 일반 경로: 현재 설정값 또는 강제 설정값 사용
    chosen_enc = enc_pref
    kb = int(os.getenv("DATAHUB_FORCE_KEY_BITS", "256") or "256")
    ivm = os.getenv("DATAHUB_FORCE_IV_MODE", "ENV").upper()
    print("[ENC][PLAINTEXT-ENCODING]", chosen_enc, "| key_bits=", kb, "| iv_mode=", ivm)
    return _enc_once(plain, chosen_enc, kb, ivm)


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

