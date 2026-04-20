"""기술적 지표 계산 모듈.

Tucker 전략에 사용되는 세 가지 지표를 계산한다:
1. VWAP (거래량 가중 이동 평균선) - UTC 00:00 (KST 09:00) 리셋
2. 9-EMA (지수 이동 평균선)
3. Volume Profile (가격대별 거래량 분포)
"""

from datetime import time, timezone

import numpy as np
import pandas as pd

from utils.logger import logger


def calc_vwap(df: pd.DataFrame, reset_hour_utc: int = 0) -> pd.Series:
    """VWAP을 계산한다. 매일 UTC reset_hour_utc 시에 리셋된다.

    VWAP = cumsum(typical_price * volume) / cumsum(volume)
    typical_price = (high + low + close) / 3

    Args:
        df: OHLCV DataFrame (datetime 인덱스, KST 기준)
        reset_hour_utc: VWAP 리셋 시각 (UTC). 기본값 0 = KST 09:00

    Returns:
        VWAP Series
    """
    # KST → UTC 변환하여 리셋 시점 판별
    # KST = UTC + 9, 따라서 UTC 00:00 = KST 09:00
    reset_hour_kst = reset_hour_utc + 9
    if reset_hour_kst >= 24:
        reset_hour_kst -= 24

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_volume = typical_price * df["volume"]

    # 리셋 그룹 생성: KST 기준으로 날짜 경계를 reset_hour_kst에서 나눔
    # 09:00 KST 이전은 전일 세션에 포함
    adjusted_dt = df.index - pd.Timedelta(hours=reset_hour_kst)
    session_date = adjusted_dt.date

    # 세션별 누적 계산
    cum_tp_volume = tp_volume.groupby(session_date).cumsum()
    cum_volume = df["volume"].groupby(session_date).cumsum()

    vwap = cum_tp_volume / cum_volume
    # 거래량이 0인 구간은 종가로 대체
    vwap = vwap.fillna(df["close"])

    logger.debug(f"VWAP 계산 완료 (리셋: KST {reset_hour_kst:02d}:00)")
    return vwap


def calc_ema(df: pd.DataFrame, period: int = 9) -> pd.Series:
    """지수 이동 평균선(EMA)을 계산한다.

    Args:
        df: OHLCV DataFrame
        period: EMA 기간. 기본값 9

    Returns:
        EMA Series
    """
    ema = df["close"].ewm(span=period, adjust=False).mean()
    logger.debug(f"{period}-EMA 계산 완료")
    return ema


def calc_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """RSI(Relative Strength Index)를 Wilder 방식(EMA smoothing)으로 계산한다.

    Args:
        df: OHLCV DataFrame
        period: RSI 기간. 기본값 14

    Returns:
        RSI Series (0~100 범위, 초반 워밍업 구간은 50.0으로 채움)
    """
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def calc_volume_ratio(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """거래량 비율(현재 봉 거래량 ÷ 최근 N봉 평균)을 계산한다.

    Args:
        df: OHLCV DataFrame
        lookback: 평균 산출 기간. 기본값 20

    Returns:
        volume_ratio Series (1.0 이상이면 평균 초과, 1.0 미만이면 미달)
    """
    avg_volume = df["volume"].rolling(window=lookback, min_periods=1).mean()
    ratio = df["volume"] / avg_volume.replace(0.0, np.nan)
    return ratio.fillna(1.0)


def calc_volume_profile(
    df: pd.DataFrame,
    num_bins: int = 20,
) -> pd.DataFrame:
    """세션별 Volume Profile을 계산한다.

    가격대를 num_bins 구간으로 나누고, 각 구간에 쌓인 거래량을 집계한다.
    세션 시작(KST 09:00)부터 현재 봉까지의 누적 프로파일을 계산한다.

    Args:
        df: OHLCV DataFrame
        num_bins: 가격 구간 수. 기본값 20

    Returns:
        DataFrame with columns:
            - price_low: 구간 하한
            - price_high: 구간 상한
            - price_mid: 구간 중간값
            - volume: 해당 구간의 총 거래량
            - is_thin: 얇은 매물대 여부
    """
    price_min = df["low"].min()
    price_max = df["high"].max()

    bin_edges = np.linspace(price_min, price_max, num_bins + 1)
    bin_volumes = np.zeros(num_bins)

    # 벡터화된 Volume Profile 계산 (numpy 기반)
    lows = df["low"].values
    highs = df["high"].values
    volumes = df["volume"].values
    ranges = highs - lows

    # 동일 가격 캔들 처리
    same_price_mask = ranges == 0
    if same_price_mask.any():
        same_indices = np.searchsorted(bin_edges, lows[same_price_mask], side="right") - 1
        same_indices = np.clip(same_indices, 0, num_bins - 1)
        np.add.at(bin_volumes, same_indices, volumes[same_price_mask])

    # 가격 범위가 있는 캔들 처리
    diff_mask = ~same_price_mask
    if diff_mask.any():
        d_lows = lows[diff_mask]
        d_highs = highs[diff_mask]
        d_volumes = volumes[diff_mask]
        d_ranges = ranges[diff_mask]

        for i in range(num_bins):
            overlap_low = np.maximum(d_lows, bin_edges[i])
            overlap_high = np.minimum(d_highs, bin_edges[i + 1])
            overlap = np.maximum(0, overlap_high - overlap_low)
            ratio = overlap / d_ranges
            bin_volumes[i] += np.sum(d_volumes * ratio)

    result = pd.DataFrame({
        "price_low": bin_edges[:-1],
        "price_high": bin_edges[1:],
        "price_mid": (bin_edges[:-1] + bin_edges[1:]) / 2,
        "volume": bin_volumes,
    })

    return result


def is_thin_volume_above(
    current_price: float,
    volume_profile: pd.DataFrame,
    thin_threshold_pct: float = 30.0,
) -> bool:
    """현재 가격 위쪽의 매물대가 '얇은지' 판단한다.

    현재 가격 위쪽 구간들의 평균 거래량이
    전체 구간 중위값의 thin_threshold_pct% 이하이면 '얇다'고 판단한다.

    Args:
        current_price: 현재 가격
        volume_profile: Volume Profile DataFrame
        thin_threshold_pct: 얇은 매물대 판정 기준 (중위값 대비 %)

    Returns:
        True면 위쪽 매물대가 얇음 (롱 진입에 유리)
    """
    above = volume_profile[volume_profile["price_low"] > current_price]

    if above.empty:
        # 위쪽에 구간이 없으면 (최고가 근처) → 매물대 없음 = 얇음
        return True

    median_volume = volume_profile["volume"].median()
    if median_volume == 0:
        return True

    above_avg_volume = above["volume"].mean()
    threshold = median_volume * (thin_threshold_pct / 100.0)

    return above_avg_volume <= threshold


def add_indicators(
    df: pd.DataFrame,
    ema_period: int = 9,
    reset_hour_utc: int = 0,
    rsi_period: int = 14,
    volume_ratio_lookback: int = 20,
) -> pd.DataFrame:
    """DataFrame에 모든 지표를 추가한다.

    Args:
        df: OHLCV DataFrame
        ema_period: EMA 기간
        reset_hour_utc: VWAP 리셋 시각 (UTC)
        rsi_period: RSI 기간 (Phase 1 진입 필터용)
        volume_ratio_lookback: 거래량 비율 평균 산출 기간

    Returns:
        지표가 추가된 DataFrame (vwap, ema, rsi, volume_ratio 컬럼 추가)
    """
    result = df.copy()
    result["vwap"] = calc_vwap(df, reset_hour_utc=reset_hour_utc)
    result["ema"] = calc_ema(df, period=ema_period)
    result["rsi"] = calc_rsi(df, period=rsi_period)
    result["volume_ratio"] = calc_volume_ratio(df, lookback=volume_ratio_lookback)

    logger.info(
        f"지표 추가 완료: VWAP(리셋 UTC {reset_hour_utc:02d}:00), "
        f"{ema_period}-EMA, RSI({rsi_period}), vol_ratio({volume_ratio_lookback}봉)"
    )
    return result
