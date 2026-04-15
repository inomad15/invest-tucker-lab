"""Binance 공개 API 데이터 수집 모듈.

인증 없이 Binance public klines API를 이용하여
BTC/USDT, ETH/USDT의 장기 분봉 데이터를 수집한다.
Upbit API 데이터 제한을 보완하기 위한 용도.
"""

import time as time_module
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from utils.logger import logger

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

# Binance 타임프레임 문자열
INTERVAL_MAP: dict[str, str] = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}

# 한 번 요청에 최대 1000개
MAX_CANDLES_PER_REQUEST = 1000

DATA_DIR = Path(__file__).resolve().parent


def _ts_to_ms(dt: datetime) -> int:
    """datetime을 밀리초 타임스탬프로 변환한다."""
    return int(dt.timestamp() * 1000)


def fetch_binance_ohlcv(
    symbol: str,
    timeframe: str,
    lookback_days: int = 90,
) -> pd.DataFrame:
    """Binance에서 OHLCV 데이터를 수집한다.

    Args:
        symbol: 심볼 (예: "BTCUSDT", "ETHUSDT")
        timeframe: 타임프레임 (예: "1m", "5m", "15m")
        lookback_days: 수집할 과거 일수

    Returns:
        OHLCV DataFrame (columns: open, high, low, close, volume, value)
        인덱스는 UTC 기준 datetime
    """
    interval = INTERVAL_MAP.get(timeframe)
    if interval is None:
        raise ValueError(f"지원하지 않는 타임프레임: {timeframe}")

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=lookback_days)

    logger.info(
        f"Binance {symbol} {timeframe} 데이터 수집 시작 "
        f"({start_time.strftime('%Y-%m-%d')} ~ {end_time.strftime('%Y-%m-%d')}, {lookback_days}일)"
    )

    all_data: list[list] = []
    current_start_ms = _ts_to_ms(start_time)
    end_ms = _ts_to_ms(end_time)

    request_count = 0
    while current_start_ms < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start_ms,
            "endTime": end_ms,
            "limit": MAX_CANDLES_PER_REQUEST,
        }

        try:
            response = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            logger.error(f"Binance API 요청 실패: {e}")
            raise

        if not data:
            break

        all_data.extend(data)
        request_count += 1

        # 다음 요청 시작점 = 마지막 캔들의 close_time + 1ms
        last_close_time = data[-1][6]
        current_start_ms = last_close_time + 1

        if request_count % 10 == 0:
            logger.debug(f"  수집 진행: {len(all_data)}개 캔들 ({request_count}회 요청)")

        # Rate limit 방지 (Binance: 1200 req/min)
        if request_count % 5 == 0:
            time_module.sleep(0.5)

    if not all_data:
        raise RuntimeError(f"데이터를 수집할 수 없음: {symbol} {timeframe}")

    # DataFrame 변환
    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_volume",
        "taker_buy_quote_volume", "ignore",
    ])

    # 타입 변환
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = df[col].astype(float)

    # UTC datetime 인덱스
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    # KST로 변환
    df["datetime"] = df["datetime"].dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    df = df.set_index("datetime")

    # 필요한 컬럼만 유지
    result = df[["open", "high", "low", "close", "volume", "quote_volume"]].copy()
    result.columns = ["open", "high", "low", "close", "volume", "value"]

    # 중복 제거
    result = result[~result.index.duplicated(keep="last")]

    logger.info(
        f"Binance {symbol} {timeframe} 수집 완료: {len(result)}개 캔들 "
        f"({result.index[0]} ~ {result.index[-1]})"
    )

    return result


def save_binance_data(df: pd.DataFrame, symbol: str, timeframe: str) -> Path:
    """Binance 데이터를 CSV로 저장한다."""
    filename = f"binance_{symbol.lower()}_{timeframe}.csv"
    filepath = DATA_DIR / filename
    df.to_csv(filepath)
    logger.info(f"데이터 저장: {filepath}")
    return filepath


def load_binance_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """저장된 Binance CSV 데이터를 로드한다."""
    filename = f"binance_{symbol.lower()}_{timeframe}.csv"
    filepath = DATA_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(f"데이터 파일 없음: {filepath}")

    df = pd.read_csv(filepath, index_col="datetime", parse_dates=True)
    logger.info(f"데이터 로드: {filepath} ({len(df)}개 캔들)")
    return df
