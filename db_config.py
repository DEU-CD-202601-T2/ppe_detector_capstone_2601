"""DB 접속 정보. .env 파일에서 환경변수를 로드."""
import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host":     os.environ.get("DB_HOST", "127.0.0.1"),
    "port":     int(os.environ.get("DB_PORT", "3306")),
    "user":     os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", ""),
    "database": os.environ.get("DB_NAME", "capstone_db"),
    "charset":  "utf8mb4",
}
