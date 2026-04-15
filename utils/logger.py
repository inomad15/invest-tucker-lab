"""로깅 설정 모듈.

loguru 기반 로거를 프로젝트 전역에서 사용할 수 있도록 설정한다.
"""

import sys
from pathlib import Path

from loguru import logger

# 기본 핸들러 제거 후 재설정
logger.remove()

# 콘솔 출력
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>",
    level="INFO",
)

# 파일 출력
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.add(
    LOG_DIR / "tucker_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
)
