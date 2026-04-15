"""Upbit API 데이터 수집 모듈.

pyupbit을 이용하여 BTC/KRW, ETH/KRW의 분봉 데이터를 수집한다.
백테스트용 과거 데이터 및 실시간 데이터 조회 기능을 제공한다.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyupbit

from utils.logger import logger

# Upbit API 분봉 조회 제한: 한 번에 최대 200개
MAX_CANDLES_PER_REQUEST = 200

# 타임프레임 → pyupbit interval 매핑
INTERVAL_MAP: dict[str, str] = {
    "1m": "minute1",
    "3m": "minute3",
    "5m": "minute5",
    "15m": "minute15",
    "30m": "minute30",
    "1h": "minute60",
    "4h": "minute240",
    "1d": "day",
}

# 타임프레임 → 분 단위 변환
TIMEFRAME_MINUTES: dict[str, int] = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}

DATA_DIR = Path(__file__).resolve().parent


def fetch_ohlcv(
    market: str,
    timeframe: str,
    lookback_days: int = 90,
) -> pd.DataFrame:
    """Upbit에서 OHLCV 데이터를 수집한다.

    Args:
        market: 마켓 코드 (예: "KRW-BTC")
        timeframe: 타임프레임 (예: "5m", "15m")
        lookback_days: 수집할 과거 일수

    Returns:
        OHLCV DataFrame (columns: open, high, low, close, volume, value)
        인덱스는 KST 기준 datetime
    """
    interval = INTERVAL_MAP.get(timeframe)
    if interval is None:
        raise ValueError(f"지원하지 않는 타임프레임: {timeframe}. 사용 가능: {list(INTERVAL_MAP.keys())}")

    minutes_per_candle = TIMEFRAME_MINUTES[timeframe]
    total_candles = (lookback_days * 24 * 60) // minutes_per_candle

    logger.info(f"{market} {timeframe} 데이터 수집 시작 (약 {total_candles}개 캔들, {lookback_days}일)")

    all_frames: list[pd.DataFrame] = []
    remaining = total_candles
    to_date: datetime | None = None

    while remaining > 0:
        count = min(remaining, MAX_CANDLES_PER_REQUEST)

        try:
            df = pyupbit.get_ohlcv(
                ticker=market,
                interval=interval,
                count=count,
                to=to_date,
            )
        except Exception as e:
            logger.error(f"데이터 수집 실패: {market} {timeframe} - {e}")
            raise

        if df is None or df.empty:
            logger.warning(f"더 이상 데이터가 없음. 수집된 캔들: {total_candles - remaining}개")
            break

        all_frames.append(df)
        remaining -= len(df)

        # 다음 요청의 종료 시점 = 현재 배치의 가장 오래된 캔들
        to_date = df.index[0]

        logger.debug(f"  수집 진행: {total_candles - remaining}/{total_candles} (to={to_date})")

    if not all_frames:
        raise RuntimeError(f"데이터를 수집할 수 없음: {market} {timeframe}")

    # 역순으로 쌓았으므로 시간순 정렬
    result = pd.concat(all_frames).sort_index()
    result = result[~result.index.duplicated(keep="last")]

    # 컬럼명 통일
    result.columns = ["open", "high", "low", "close", "volume", "value"]
    result.index.name = "datetime"

    logger.info(f"{market} {timeframe} 수집 완료: {len(result)}개 캔들 ({result.index[0]} ~ {result.index[-1]})")

    return result


def save_data(df: pd.DataFrame, market: str, timeframe: str) -> Path:
    """수집된 데이터를 CSV 파일로 저장한다.

    Args:
        df: OHLCV DataFrame
        market: 마켓 코드
        timeframe: 타임프레임

    Returns:
        저장된 파일 경로
    """
    filename = f"{market.lower().replace('-', '_')}_{timeframe}.csv"
    filepath = DATA_DIR / filename
    df.to_csv(filepath)
    logger.info(f"데이터 저장: {filepath}")
    return filepath


def load_data(market: str, timeframe: str) -> pd.DataFrame:
    """저장된 CSV 데이터를 로드한다.

    Args:
        market: 마켓 코드
        timeframe: 타임프레임

    Returns:
        OHLCV DataFrame
    """
    filename = f"{market.lower().replace('-', '_')}_{timeframe}.csv"
    filepath = DATA_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(f"데이터 파일 없음: {filepath}")

    df = pd.read_csv(filepath, index_col="datetime", parse_dates=True)
    logger.info(f"데이터 로드: {filepath} ({len(df)}개 캔들)")
    return df
