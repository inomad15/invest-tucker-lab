"""Tucker v4 전략 — 평균회귀 + 눌림목 복합 전략.

검증된 사실:
- 추세추종 지표(VWAP, EMA, ADX, RSI, MACD)는 미래 방향을 예측 못함 (50% 미만)
- 평균회귀는 유의미한 예측력이 있음: EMA에서 멀어질수록 되돌림 확률 증가
  - 괴리 -1% 이상 → 1시간 후 상승 57%
  - 괴리 +1% 이상 → 1시간 후 하락 55%

전략 설계:
1. 평균회귀 매수: 가격이 EMA 아래로 크게 이탈 → 되돌림 기대 매수
2. 모멘텀 확인: 상위 타임프레임이 극단적 하락이 아닐 때만 진입
3. 청산: EMA 복귀 시 익절 + ATR 기반 손절

현물 전용 (롱만 가능).
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from utils.logger import logger


class Signal(Enum):
    HOLD = "hold"
    BUY = "buy"
    SELL = "sell"
    WAIT = "wait"


@dataclass
class TradeResult:
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    pnl_pct: float
    pnl_krw: float
    holding_bars: int
    exit_reason: str


class TuckerV4Strategy:
    """Tucker v4 — 평균회귀 + 눌림목 복합 전략.

    매수 조건 (모두 충족):
    1. 가격이 EMA 아래로 entry_deviation_pct% 이상 이탈 (과매도)
    2. RSI가 rsi_oversold 이하 (모멘텀 확인)
    3. 양봉 출현 (반등 시작 신호)
    4. 장기 EMA(200) 대비 extreme_drop_pct% 이상 하락이 아닐 것 (추세 붕괴 필터)

    청산 조건:
    - 익절: 가격이 EMA 위로 복귀 (평균회귀 완료)
    - 손절: ATR 기반 손절선 이탈
    - 시간 손절: max_hold_bars봉 이내 미복귀 시 청산

    Args:
        ema_period: 평균회귀 기준 EMA 기간
        entry_deviation_pct: 매수 진입 괴리율 (EMA 아래 %)
        rsi_period: RSI 기간
        rsi_oversold: RSI 과매도 기준
        atr_period: ATR 기간
        atr_stop_multiplier: ATR 손절 배수
        max_hold_bars: 최대 보유 봉 수
        cooldown_bars: 청산 후 재진입 금지 봉 수
        ema_long_period: 장기 EMA (추세 붕괴 필터)
        extreme_drop_pct: 장기 EMA 대비 극단 하락 필터 (%)
        take_profit_pct: 익절 목표 (%, 0이면 EMA 복귀 익절만)
    """

    def __init__(
        self,
        ema_period: int = 21,
        entry_deviation_pct: float = 0.5,
        rsi_period: int = 14,
        rsi_oversold: float = 40,
        atr_period: int = 14,
        atr_stop_multiplier: float = 2.0,
        max_hold_bars: int = 36,
        cooldown_bars: int = 6,
        ema_long_period: int = 200,
        extreme_drop_pct: float = 5.0,
        take_profit_pct: float = 0,
    ) -> None:
        self.ema_period = ema_period
        self.entry_deviation_pct = entry_deviation_pct
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.max_hold_bars = max_hold_bars
        self.cooldown_bars = cooldown_bars
        self.ema_long_period = ema_long_period
        self.extreme_drop_pct = extreme_drop_pct
        self.take_profit_pct = take_profit_pct

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """필요한 지표를 모두 계산한다."""
        result = df.copy()

        # EMA
        result["ema"] = result["close"].ewm(span=self.ema_period, adjust=False).mean()
        result["ema_long"] = result["close"].ewm(span=self.ema_long_period, adjust=False).mean()

        # EMA 괴리율 (%)
        result["deviation"] = (result["close"] - result["ema"]) / result["ema"] * 100

        # RSI
        delta = result["close"].diff()
        gain = delta.clip(lower=0).ewm(span=self.rsi_period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(span=self.rsi_period, adjust=False).mean()
        rs = gain / loss
        result["rsi"] = 100 - (100 / (1 + rs))

        # ATR
        tr = pd.concat([
            result["high"] - result["low"],
            (result["high"] - result["close"].shift(1)).abs(),
            (result["low"] - result["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        result["atr"] = tr.ewm(span=self.atr_period, adjust=False).mean()

        # 장기 EMA 대비 괴리율
        result["long_deviation"] = (result["close"] - result["ema_long"]) / result["ema_long"] * 100

        # 양봉 여부
        result["is_bullish"] = result["close"] > result["open"]

        return result

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """매매 시그널을 생성한다."""
        df = self._add_indicators(df)

        signals: list[str] = []
        positions: list[int] = []
        exit_reasons: list[str] = []

        in_position = False
        entry_price = 0.0
        entry_bar = 0
        cooldown_remaining = 0

        warmup = max(self.ema_long_period + 10, self.atr_period * 2, self.rsi_period * 2)

        for idx in range(len(df)):
            if idx < warmup:
                signals.append(Signal.WAIT.value)
                positions.append(0)
                exit_reasons.append("")
                continue

            row = df.iloc[idx]

            if in_position:
                exit_reason = ""
                bars_held = idx - entry_bar

                # 익절: EMA 위로 복귀
                if row["close"] >= row["ema"]:
                    exit_reason = "ema_revert"

                # 익절: 목표 수익률 도달
                if not exit_reason and self.take_profit_pct > 0:
                    current_pnl = (row["close"] / entry_price - 1) * 100
                    if current_pnl >= self.take_profit_pct:
                        exit_reason = "take_profit"

                # 손절: ATR 기반
                if not exit_reason and self.atr_stop_multiplier > 0:
                    stop_price = entry_price - (row["atr"] * self.atr_stop_multiplier)
                    if row["close"] < stop_price:
                        exit_reason = "atr_stop"

                # 시간 손절: 최대 보유 기간 초과
                if not exit_reason and bars_held >= self.max_hold_bars:
                    exit_reason = "time_stop"

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
                    continue

                # 매수 조건 확인
                buy = True

                # 조건 1: EMA 아래로 충분히 이탈
                if row["deviation"] > -self.entry_deviation_pct:
                    buy = False

                # 조건 2: RSI 과매도
                if buy and row["rsi"] > self.rsi_oversold:
                    buy = False

                # 조건 3: 양봉 (반등 시작)
                if buy and not row["is_bullish"]:
                    buy = False

                # 조건 4: 장기 추세 붕괴 필터
                if buy and row["long_deviation"] < -self.extreme_drop_pct:
                    buy = False

                if buy:
                    signals.append(Signal.BUY.value)
                    positions.append(1)
                    exit_reasons.append("")
                    in_position = True
                    entry_price = row["close"]
                    entry_bar = idx
                else:
                    signals.append(Signal.WAIT.value)
                    positions.append(0)
                    exit_reasons.append("")

        df["signal"] = signals
        df["position"] = positions
        df["exit_reason"] = exit_reasons

        buy_count = signals.count(Signal.BUY.value)
        sell_count = signals.count(Signal.SELL.value)

        # 청산 사유 통계
        revert = exit_reasons.count("ema_revert")
        tp = exit_reasons.count("take_profit")
        atr = exit_reasons.count("atr_stop")
        time_s = exit_reasons.count("time_stop")

        logger.info(
            f"v4 시그널: BUY={buy_count}, SELL={sell_count} "
            f"(EMA복귀={revert}, 익절={tp}, ATR손절={atr}, 시간손절={time_s})"
        )

        return df
