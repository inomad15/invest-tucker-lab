"""Tucker Brooks 매매 전략 v3 모듈.

v2 대비 핵심 개선: 진입 로직을 다봉 패턴 기반으로 재설계.

원본 전략의 "눌림목"을 정확히 구현:
1. 선행 상승: 최근 N봉 내 가격이 EMA 위로 충분히 벌어진 적이 있어야 함
2. 풀백: 이후 가격이 EMA 근처로 되돌아옴
3. 반등 확인: EMA 지지 + 양봉 마감

추가 개선:
- 청산 확인 봉 수 (whipsaw 방지)
- 쿨다운 (연속 손절 방지)
- ATR 기반 긴급 손절
- VWAP 추세 강도 필터

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
    exit_reason: str


class TuckerStrategyV3:
    """Tucker Brooks 매매 전략 v3.

    다봉 패턴 기반 눌림목 진입 + 확인 청산.

    Args:
        ema_period: EMA 기간
        reset_hour_utc: VWAP 리셋 시각 (UTC)
        swing_lookback: 선행 상승 확인 lookback 봉 수
        swing_min_distance_pct: 선행 상승 시 EMA 대비 최소 괴리 (%)
        ema_proximity_pct: EMA 근접 판정 범위 (%)
        vwap_chop_lookback: VWAP 횡보 판정 lookback
        vwap_chop_cross_threshold: 횡보 교차 횟수 임계값
        vp_num_bins: Volume Profile 구간 수
        vp_thin_threshold_pct: 얇은 매물대 판정 기준
        vp_lookback_bars: VP lookback
        exit_confirm_bars: 청산 확인 봉 수
        cooldown_bars: 청산 후 재진입 금지 봉 수
        atr_period: ATR 기간
        atr_stop_multiplier: ATR 손절 배수 (0이면 비활성)
    """

    def __init__(
        self,
        ema_period: int = 9,
        reset_hour_utc: int = 0,
        swing_lookback: int = 10,
        swing_min_distance_pct: float = 0.3,
        ema_proximity_pct: float = 0.5,
        vwap_chop_lookback: int = 10,
        vwap_chop_cross_threshold: int = 4,
        vp_num_bins: int = 20,
        vp_thin_threshold_pct: float = 30.0,
        vp_lookback_bars: int = 50,
        exit_confirm_bars: int = 2,
        cooldown_bars: int = 5,
        atr_period: int = 14,
        atr_stop_multiplier: float = 2.0,
    ) -> None:
        self.ema_period = ema_period
        self.reset_hour_utc = reset_hour_utc
        self.swing_lookback = swing_lookback
        self.swing_min_distance_pct = swing_min_distance_pct
        self.ema_proximity_pct = ema_proximity_pct
        self.vwap_chop_lookback = vwap_chop_lookback
        self.vwap_chop_cross_threshold = vwap_chop_cross_threshold
        self.vp_num_bins = vp_num_bins
        self.vp_thin_threshold_pct = vp_thin_threshold_pct
        self.vp_lookback_bars = vp_lookback_bars
        self.exit_confirm_bars = exit_confirm_bars
        self.cooldown_bars = cooldown_bars
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier

    def _calc_atr(self, df: pd.DataFrame) -> pd.Series:
        """ATR을 계산한다."""
        high = df["high"]
        low = df["low"]
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(span=self.atr_period, adjust=False).mean()

    def _is_vwap_choppy(self, df: pd.DataFrame, idx: int) -> bool:
        """VWAP 횡보 구간인지 판별한다."""
        start = max(0, idx - self.vwap_chop_lookback)
        window = df.iloc[start:idx + 1]
        if len(window) < 3:
            return True
        position = np.sign(window["close"].values - window["vwap"].values)
        crosses = np.sum(np.abs(np.diff(position)) > 0)
        return crosses >= self.vwap_chop_cross_threshold

    def _had_prior_swing(self, df: pd.DataFrame, idx: int) -> bool:
        """최근 N봉 내 선행 상승(EMA에서 충분히 벌어진 적)이 있는지 확인한다.

        Tucker의 눌림목은 "상승 후 되돌림"이므로,
        먼저 가격이 EMA 위로 충분히 올라간 적이 있어야 한다.
        """
        start = max(0, idx - self.swing_lookback)
        window = df.iloc[start:idx]  # 현재 봉 제외

        if len(window) < 2:
            return False

        # lookback 구간 내 high가 EMA 대비 swing_min_distance_pct% 이상 벌어진 봉이 있는가
        ema_vals = window["ema"].values
        high_vals = window["high"].values
        distances = (high_vals - ema_vals) / ema_vals * 100

        return np.any(distances >= self.swing_min_distance_pct)

    def _is_pullback_bounce(self, df: pd.DataFrame, idx: int) -> bool:
        """현재 봉이 EMA 눌림목 반등 양봉인지 확인한다.

        조건:
        1. 종가 > VWAP (상승 방향)
        2. 선행 상승이 있었음 (had_prior_swing)
        3. 저가가 EMA 근접 (풀백 확인)
        4. 종가 > EMA (지지 확인)
        5. 양봉 (close > open)
        """
        row = df.iloc[idx]
        ema_val = row["ema"]

        # 조건 1: 종가 > VWAP
        if row["close"] <= row["vwap"]:
            return False

        # 조건 2: 선행 상승 확인
        if not self._had_prior_swing(df, idx):
            return False

        # 조건 3: 저가가 EMA 근접 (눌림목)
        proximity = ema_val * (self.ema_proximity_pct / 100.0)
        if row["low"] > ema_val + proximity:
            return False

        # 조건 4: 종가 > EMA (지지 확인)
        if row["close"] <= ema_val:
            return False

        # 조건 5: 양봉
        if row["close"] <= row["open"]:
            return False

        return True

    def _check_volume_profile(self, df: pd.DataFrame, idx: int) -> bool:
        """위쪽 매물대가 얇은지 확인한다."""
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
        """연속으로 종가 < EMA인 봉 수를 센다."""
        count = 0
        for i in range(idx, max(idx - self.exit_confirm_bars - 1, -1), -1):
            if df.iloc[i]["close"] < df.iloc[i]["ema"]:
                count += 1
            else:
                break
        return count

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """전체 데이터에 대해 매매 시그널을 생성한다."""
        if "vwap" not in df.columns or "ema" not in df.columns:
            df = add_indicators(
                df,
                ema_period=self.ema_period,
                reset_hour_utc=self.reset_hour_utc,
            )

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
            self.swing_lookback + 5,
        )

        for idx in range(len(df)):
            if idx < warmup:
                signals.append(Signal.WAIT.value)
                positions.append(0)
                exit_reasons.append("")
                continue

            row = df.iloc[idx]

            if in_position:
                exit_reason = ""

                # ATR 긴급 손절
                if self.atr_stop_multiplier > 0:
                    atr_stop = entry_price - (row["atr"] * self.atr_stop_multiplier)
                    if row["close"] < atr_stop:
                        exit_reason = "atr_stop"

                # EMA 아래 N봉 연속 마감
                if not exit_reason:
                    if self._count_consecutive_below_ema(df, idx) >= self.exit_confirm_bars:
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
                if cooldown_remaining > 0:
                    cooldown_remaining -= 1
                    signals.append(Signal.WAIT.value)
                    positions.append(0)
                    exit_reasons.append("cooldown")
                elif self._is_vwap_choppy(df, idx):
                    signals.append(Signal.WAIT.value)
                    positions.append(0)
                    exit_reasons.append("choppy")
                elif self._is_pullback_bounce(df, idx) and self._check_volume_profile(df, idx):
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
        ema_exits = exit_reasons.count("ema_exit")
        atr_stops = exit_reasons.count("atr_stop")

        logger.info(
            f"v3 시그널 생성: BUY={buy_count}, SELL={sell_count} "
            f"(EMA청산={ema_exits}, ATR손절={atr_stops})"
        )

        return df
