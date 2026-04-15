"""Tucker Brooks 매매 전략 v2 모듈.

v1 대비 개선점:
1. 청산 확인 강화: EMA 아래 N봉 연속 마감 시에만 청산 (whipsaw 방지)
2. 쿨다운: 청산 후 일정 봉 동안 재진입 금지 (연속 손절 방지)
3. 추세 강도 필터: VWAP 대비 가격 괴리율이 일정 이상일 때만 진입
4. ATR 기반 손절: 극단적 역행 시 즉시 손절 (안전장치)

현물 전용 (롱만 가능).
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from strategy.indicators import (
    add_indicators,
    calc_volume_profile,
    is_thin_volume_above,
)
from utils.logger import logger


class Signal(Enum):
    """매매 시그널."""
    HOLD = "hold"
    BUY = "buy"
    SELL = "sell"
    WAIT = "wait"


@dataclass
class TradeResult:
    """개별 거래 결과."""
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    pnl_pct: float
    pnl_krw: float
    holding_bars: int
    exit_reason: str  # v2: 청산 사유 추가


class TuckerStrategyV2:
    """Tucker Brooks 매매 전략 v2.

    v1의 과도한 whipsaw 문제를 해결하기 위해
    청산 확인 봉 수, 쿨다운, ATR 손절을 추가.

    Attributes:
        ema_period: EMA 기간
        reset_hour_utc: VWAP 리셋 시각 (UTC)
        ema_proximity_pct: EMA 근접 판정 범위 (%)
        vwap_chop_lookback: VWAP 횡보 판정 lookback 봉 수
        vwap_chop_cross_threshold: 횡보 판정 교차 횟수 임계값
        vp_num_bins: Volume Profile 구간 수
        vp_thin_threshold_pct: 얇은 매물대 판정 기준
        vp_lookback_bars: VP 계산 시 사용할 과거 봉 수

        exit_confirm_bars: EMA 아래 연속 N봉 마감 시 청산 (v2)
        cooldown_bars: 청산 후 재진입 금지 봉 수 (v2)
        vwap_min_distance_pct: VWAP 대비 최소 괴리율 (%, v2)
        atr_period: ATR 계산 기간 (v2)
        atr_stop_multiplier: ATR 기반 손절 배수 (v2, 0이면 비활성)
    """

    def __init__(
        self,
        ema_period: int = 9,
        reset_hour_utc: int = 0,
        ema_proximity_pct: float = 0.3,
        vwap_chop_lookback: int = 10,
        vwap_chop_cross_threshold: int = 3,
        vp_num_bins: int = 20,
        vp_thin_threshold_pct: float = 30.0,
        vp_lookback_bars: int = 50,
        # v2 신규 파라미터
        exit_confirm_bars: int = 2,
        cooldown_bars: int = 3,
        vwap_min_distance_pct: float = 0.1,
        atr_period: int = 14,
        atr_stop_multiplier: float = 2.0,
    ) -> None:
        self.ema_period = ema_period
        self.reset_hour_utc = reset_hour_utc
        self.ema_proximity_pct = ema_proximity_pct
        self.vwap_chop_lookback = vwap_chop_lookback
        self.vwap_chop_cross_threshold = vwap_chop_cross_threshold
        self.vp_num_bins = vp_num_bins
        self.vp_thin_threshold_pct = vp_thin_threshold_pct
        self.vp_lookback_bars = vp_lookback_bars

        self.exit_confirm_bars = exit_confirm_bars
        self.cooldown_bars = cooldown_bars
        self.vwap_min_distance_pct = vwap_min_distance_pct
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier

    def _calc_atr(self, df: pd.DataFrame) -> pd.Series:
        """ATR (Average True Range)을 계산한다."""
        high = df["high"]
        low = df["low"]
        prev_close = df["close"].shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=self.atr_period, adjust=False).mean()
        return atr

    def _is_vwap_choppy(self, df: pd.DataFrame, idx: int) -> bool:
        """VWAP 횡보 구간인지 판별한다."""
        start = max(0, idx - self.vwap_chop_lookback)
        window = df.iloc[start:idx + 1]

        if len(window) < 3:
            return True

        position = np.sign(window["close"] - window["vwap"])
        crosses = (position.diff().abs() > 0).sum()
        return crosses >= self.vwap_chop_cross_threshold

    def _is_ema_pullback_entry(self, df: pd.DataFrame, idx: int) -> bool:
        """EMA 눌림목 매수 진입 조건을 확인한다."""
        row = df.iloc[idx]
        ema_val = row["ema"]
        vwap_val = row["vwap"]

        # 조건 1: 종가 > VWAP
        if row["close"] <= vwap_val:
            return False

        # 조건 1.5 (v2): VWAP 대비 최소 괴리율
        vwap_distance = (row["close"] - vwap_val) / vwap_val * 100
        if vwap_distance < self.vwap_min_distance_pct:
            return False

        # 조건 2: 저가가 EMA 근처까지 내려옴
        proximity = ema_val * (self.ema_proximity_pct / 100.0)
        if row["low"] > ema_val + proximity:
            return False

        # 조건 3: 종가 > EMA
        if row["close"] <= ema_val:
            return False

        # 조건 4: 양봉
        if row["close"] <= row["open"]:
            return False

        return True

    def _check_volume_profile(self, df: pd.DataFrame, idx: int) -> bool:
        """현재 가격 위쪽의 Volume Profile이 얇은지 확인한다."""
        start = max(0, idx - self.vp_lookback_bars)
        window = df.iloc[start:idx + 1]

        vp = calc_volume_profile(window, num_bins=self.vp_num_bins)
        current_price = df.iloc[idx]["close"]

        return is_thin_volume_above(
            current_price=current_price,
            volume_profile=vp,
            thin_threshold_pct=self.vp_thin_threshold_pct,
        )

    def _count_consecutive_below_ema(self, df: pd.DataFrame, idx: int) -> int:
        """현재 봉 포함, 연속으로 종가 < EMA인 봉 수를 센다."""
        count = 0
        for i in range(idx, max(idx - self.exit_confirm_bars - 1, -1), -1):
            if df.iloc[i]["close"] < df.iloc[i]["ema"]:
                count += 1
            else:
                break
        return count

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """전체 데이터에 대해 매매 시그널을 생성한다.

        Args:
            df: OHLCV DataFrame

        Returns:
            시그널이 추가된 DataFrame
        """
        if "vwap" not in df.columns or "ema" not in df.columns:
            df = add_indicators(
                df,
                ema_period=self.ema_period,
                reset_hour_utc=self.reset_hour_utc,
            )

        # ATR 계산
        df = df.copy()
        df["atr"] = self._calc_atr(df)

        signals: list[str] = []
        positions: list[int] = []
        exit_reasons: list[str] = []

        in_position = False
        entry_price = 0.0
        cooldown_remaining = 0

        warmup = max(
            self.ema_period * 3,
            self.vwap_chop_lookback,
            self.vp_lookback_bars,
            self.atr_period * 2,
        )

        for idx in range(len(df)):
            if idx < warmup:
                signals.append(Signal.WAIT.value)
                positions.append(0)
                exit_reasons.append("")
                continue

            row = df.iloc[idx]

            if in_position:
                # === 청산 조건 확인 ===
                exit_reason = ""

                # ATR 기반 긴급 손절 (극단적 역행)
                if self.atr_stop_multiplier > 0:
                    atr_stop = entry_price - (row["atr"] * self.atr_stop_multiplier)
                    if row["close"] < atr_stop:
                        exit_reason = "atr_stop"

                # EMA 아래 N봉 연속 마감 (일반 청산)
                if not exit_reason:
                    consecutive_below = self._count_consecutive_below_ema(df, idx)
                    if consecutive_below >= self.exit_confirm_bars:
                        exit_reason = "ema_exit"

                if exit_reason:
                    signals.append(Signal.SELL.value)
                    positions.append(0)
                    exit_reasons.append(exit_reason)
                    in_position = False
                    entry_price = 0.0
                    cooldown_remaining = self.cooldown_bars
                else:
                    signals.append(Signal.HOLD.value)
                    positions.append(1)
                    exit_reasons.append("")
            else:
                # === 진입 조건 확인 ===
                if cooldown_remaining > 0:
                    cooldown_remaining -= 1
                    signals.append(Signal.WAIT.value)
                    positions.append(0)
                    exit_reasons.append("cooldown")
                elif self._is_vwap_choppy(df, idx):
                    signals.append(Signal.WAIT.value)
                    positions.append(0)
                    exit_reasons.append("choppy")
                elif self._is_ema_pullback_entry(df, idx) and self._check_volume_profile(df, idx):
                    signals.append(Signal.BUY.value)
                    positions.append(1)
                    exit_reasons.append("")
                    in_position = True
                    entry_price = row["close"]
                else:
                    signals.append(Signal.WAIT.value)
                    positions.append(0)
                    exit_reasons.append("")

        df["signal"] = signals
        df["position"] = positions
        df["exit_reason"] = exit_reasons

        buy_count = signals.count(Signal.BUY.value)
        sell_count = signals.count(Signal.SELL.value)

        # 청산 사유별 통계
        ema_exits = exit_reasons.count("ema_exit")
        atr_stops = exit_reasons.count("atr_stop")
        logger.info(
            f"시그널 생성 완료: BUY={buy_count}, SELL={sell_count} "
            f"(EMA청산={ema_exits}, ATR손절={atr_stops})"
        )

        return df
