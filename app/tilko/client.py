# app/tilko/client.py
import os, base64, json, requests
from typing import Dict, Any, Optional
from Crypto.Cipher import AES, PKCS1_v1_5
from Crypto.PublicKey import RSA

class TilkoError(RuntimeError):
    pass

class TilkoClient:
    """
    Tilko 간편인증(NHIS) + 건강검진 조회 전용 클라이언트.
    - AES-128-CBC(IV=0) 본문 암호화
    - RSA 공개키로 AES 키 암호화(ENC-KEY 헤더)
    - API-KEY 헤더
    """
    def __init__(self, api_key: str, host: Optional[str] = None):
        self.api_key = api_key
        self.host = host or os.getenv("TILKO_API_HOST", "https://dev.tilko.net").rstrip("/")
        self._public_key_b64: Optional[str] = None

    # ---------- 내부 유틸 ----------
    def _ensure_public_key(self) -> str:
        if self._public_key_b64:
            return self._public_key_b64
        url = f"{self.host}/api/Auth/GetPublicKey"
        r = requests.get(url, params={"ApiKey": self.api_key}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("Status") != "OK" or not data.get("PublicKey"):
            raise TilkoError(f"GetPublicKey failed: {data}")
        self._public_key_b64 = data["PublicKey"]
        return self._public_key_b64

    @staticmethod
    def _pkcs7_pad(b: bytes) -> bytes:
        pad_len = 16 - (len(b) % 16)
        return b + bytes([pad_len]) * pad_len

    @staticmethod
    def _aes_encrypt_cbc_zero_iv(aes_key_16b: bytes, plain: bytes | str) -> str:
        iv = b"\x00" * 16
        if isinstance(plain, str):
            plain = plain.encode("utf-8")
        from Crypto.Cipher import AES
        cipher = AES.new(aes_key_16b, AES.MODE_CBC, iv)
        enc = cipher.encrypt(TilkoClient._pkcs7_pad(plain))
        return base64.b64encode(enc).decode("utf-8")

    def _rsa_encrypt_aes_key(self, aes_key_16b: bytes) -> str:
        pub_der = base64.b64decode(self._ensure_public_key())
        rsa_key = RSA.import_key(pub_der)
        cipher = PKCS1_v1_5.new(rsa_key)
        enc = cipher.encrypt(aes_key_16b)
        return base64.b64encode(enc).decode("utf-8")

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        import traceback
        aes_key = os.urandom(16)
        enc_key = self._rsa_encrypt_aes_key(aes_key)

        headers = {
            "Content-Type": "application/json",
            "API-KEY": self.api_key,
            "ENC-KEY": enc_key,
        }

        enc_fields = payload.get("__encrypt__", [])
        body = {
            k: (self._aes_encrypt_cbc_zero_iv(aes_key, v) if k in enc_fields else v)
            for k, v in payload.items() if k != "__encrypt__"
        }

        url = f"{self.host}{path}"

        # --- 디버그: 요청 요약(민감정보 마스킹) ---
        try:
            masked_headers = dict(headers)
            if "API-KEY" in masked_headers:
                masked_headers["API-KEY"] = masked_headers["API-KEY"][:6] + "****"
            print("[TILKO][REQ]", url, "hdr=", masked_headers, "enc_fields=", enc_fields, "keys=", list(body.keys()))
        except Exception:
            pass

        try:
            r = requests.post(url, headers=headers, json=body, timeout=25)
            print("[TILKO][RES]", r.status_code)
            r.raise_for_status()
            data = r.json()
            # Tilko 포맷에 따라 Status 체크(없으면 OK로 간주)
            if isinstance(data, dict) and data.get("Status") not in (None, "OK"):
                print("[TILKO][ERR] payload:", data)
                raise TilkoError(f"TILKO API error: {data}")
            return data
        except Exception as e:
            # HTTP 오류/타임아웃/JSON 파싱 문제 모두 여기로
            try:
                print("[TILKO][EXC]", repr(e))
                if 'r' in locals():
                    print("[TILKO][RES-TEXT]", getattr(r, "text", "")[:2000])
            except Exception:
                pass
            traceback.print_exc()
            raise


    # ---------- 공개 API ----------
    def nhis_simpleauth_request(self, name: str, phone: str, birth_yyyymmdd: str) -> Dict[str, Any]:
        """
        간편인증 시작: /api/v1.0/NhisSimpleAuth/SimpleAuthRequest
        - 문서 명세에 따라 [암호화] 항목만 __encrypt__에 넣을 것
        """
        payload = {
            "Name": name,
            "CellphoneNo": phone,
            "Birth": birth_yyyymmdd,
            "__encrypt__": ["Name", "CellphoneNo", "Birth"],
        }
        return self._post("/api/v1.0/NhisSimpleAuth/SimpleAuthRequest", payload)

    def nhis_healthcheck_after_auth(self, tx_id: str, from_year: str, to_year: str) -> Dict[str, Any]:
        """
        인증 후 건강검진 조회: /api/v1.0/NhisSimpleAuth/Ggpab003M0105
        """
        payload = {
            "TxId": tx_id,
            "FromYear": from_year,
            "ToYear": to_year,
            "__encrypt__": ["TxId"],
        }
        return self._post("/api/v1.0/NhisSimpleAuth/Ggpab003M0105", payload)
