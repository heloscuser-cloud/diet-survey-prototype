import os
import secrets
from pathlib import Path

from sqlmodel import Session, select

# main.py에서 engine/ReportFile/now_kst 등을 이미 쓰고 있으니 그대로 import해서 재사용
import sys
sys.path.insert(0, "/app")  # 'app' 패키지 인식용

from app.main import engine, ReportFile, now_kst, REPORTS_DIR

def main():
    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)

    # 한 번에 너무 많이 처리하지 않도록 배치
    BATCH_SIZE = int(os.getenv("MIGRATE_BATCH_SIZE", "50"))

    with Session(engine) as session:
        while True:
            # file_path 없는 것 중 content가 있는 것만 가져오기
            rows = session.exec(
                select(ReportFile)
                .where(ReportFile.file_path.is_(None))
                .where(ReportFile.content.is_not(None))
                .limit(BATCH_SIZE)
            ).all()

            if not rows:
                print("[MIGRATE] done. no more rows.")
                break

            for rf in rows:
                content = rf.content
                if content is None:
                    continue
                if isinstance(content, memoryview):
                    content = content.tobytes()
                if not content:
                    # 빈 content는 그냥 NULL 처리
                    rf.content = None
                    session.add(rf)
                    continue

                # 파일명 충돌 방지: reportfile_id + survey_response_id + 랜덤
                safe_name = f"rf_{rf.id}_sr_{rf.survey_response_id}_{now_kst().strftime('%Y%m%d%H%M%S')}_{secrets.token_hex(6)}.pdf"
                abs_path = Path(REPORTS_DIR) / safe_name

                # 원자적 저장: tmp로 쓰고 rename
                tmp_path = abs_path.with_suffix(".tmp")
                tmp_path.write_bytes(content)
                tmp_path.replace(abs_path)

                # DB 업데이트: file_path 세팅 + content NULL (DB 용량 줄이기)
                rf.file_path = safe_name
                rf.content = None
                session.add(rf)

                print(f"[MIGRATE] reportfile_id={rf.id} survey_response_id={rf.survey_response_id} -> {abs_path}")

            session.commit()

if __name__ == "__main__":
    main()