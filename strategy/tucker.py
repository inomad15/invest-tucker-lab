"""Tucker Brooks 매매 전략 모듈.

현물 전용 (롱만 가능) 전략:
- 진입: VWAP 위 + 9-EMA 눌림목 양봉 + 위쪽 매물대 얇음
- 청산: 캔들 몸통(종가)이 9-EMA 아래에서 마감
- 관망: VWAP 횡보 구간
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
    HOLD = "hold"        # 보유 유지 (포지션 있을 때)
    BUY = "buy"          # 매수 진입
    SELL = "sell"        # 매도 청산
    WAIT = "wait"        # 관망 (포지션 없을 때)


@dataclass
class TradeResult:
    """개별 거래 결과."""
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    pnl_pct: float       # 수익률 (%)
    pnl_krw: float       # 손익 (KRW)
    holding_bars: int     # 보유 봉 수


class TuckerStrategy:
    """Tucker Brooks 매매 전략.

    현물 전용으로, VWAP + 9-EMA + Volume Profile 세 가지 조건을
    결합하여 매수/매도 시그널을 생성한다.

    Attributes:
        ema_period: EMA 기간 (기본 9)
        reset_hour_utc: VWAP 리셋 시각 (UTC, 기본 0)
        ema_proximity_pct: EMA 근접 판정 범위 (%, 기본 0.3)
        vwap_chop_lookback: VWAP 횡보 판정 lookback 봉 수 (기본 10)
        vwap_chop_cross_threshold: 횡보 판정 교차 횟수 임계값 (기본 3)
        vp_num_bins: Volume Profile 구간 수 (기본 20)
        vp_thin_threshold_pct: 얇은 매물대 판정 기준 (기본 30)
        vp_lookback_bars: VP 계산 시 사용할 과거 봉 수 (기본 50)
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
    ) -> None:
        self.ema_period = ema_period
        self.reset_hour_utc = reset_hour_utc
        self.ema_proximity_pct = ema_proximity_pct
        self.vwap_chop_lookback = vwap_chop_lookback
        self.vwap_chop_cross_threshold = vwap_chop_cross_threshold
        self.vp_num_bins = vp_num_bins
        self.vp_thin_threshold_pct = vp_thin_threshold_pct
        self.vp_lookback_bars = vp_lookback_bars

    def _is_vwap_choppy(self, df: pd.DataFrame, idx: int) -> bool:
        """VWAP 횡보 구간인지 판별한다.

        최근 lookback 봉 동안 종가가 VWAP을 몇 번 교차했는지 확인한다.
        교차 횟수가 임계값 이상이면 횡보(세력 간 힘겨루기)로 판단.

        Args:
            df: 지표가 추가된 DataFrame
            idx: 현재 봉의 정수 인덱스

        Returns:
            True면 횡보 구간 (매매 금지)
        """
        start = max(0, idx - self.vwap_chop_lookback)
        window = df.iloc[start:idx + 1]

        if len(window) < 3:
            return True  # 데이터 부족 시 관망

        # 종가와 VWAP의 위치 관계 (위=1, 아래=-1)
        position = np.sign(window["close"] - window["vwap"])
        # 부호가 바뀌는 횟수 = 교차 횟수
        crosses = (position.diff().abs() > 0).sum()

        return crosses >= self.vwap_chop_cross_threshold

    def _is_ema_pullback_entry(self, df: pd.DataFrame, idx: int) -> bool:
        """EMA 눌림목 매수 진입 조건을 확인한다.

        조건:
        1. 현재 종가 > VWAP (상승 방향)
        2. 저가가 EMA 근접 (EMA ± proximity %) — 눌림목 확인
        3. 종가 > EMA — EMA 지지 확인
        4. 양봉 (종가 > 시가) — 반등 양봉

        Args:
            df: 지표가 추가된 DataFrame
            idx: 현재 봉의 정수 인덱스

        Returns:
            True면 매수 진입 조건 충족
        """
        row = df.iloc[idx]
        ema_val = row["ema"]

        # 조건 1: 종가 > VWAP
        if row["close"] <= row["vwap"]:
            return False

        # 조건 2: 저가가 EMA 근처까지 내려옴 (눌림목)
        proximity = ema_val * (self.ema_proximity_pct / 100.0)
        if row["low"] > ema_val + proximity:
            # 저가가 EMA보다 훨씬 위 → 눌림목 아님
            return False

        # 조건 3: 종가 > EMA (지지 확인, EMA 이탈하지 않음)
        if row["close"] <= ema_val:
            return False

        # 조건 4: 양봉
        if row["close"] <= row["open"]:
            return False

        return True

    def _check_volume_profile(self, df: pd.DataFrame, idx: int) -> bool:
        """현재 가격 위쪽의 Volume Profile이 얇은지 확인한다.

        Args:
            df: 지표가 추가된 DataFrame
            idx: 현재 봉의 정수 인덱스

        Returns:
            True면 위쪽 매물대가 얇음 (매수에 유리)
        """
        start = max(0, idx - self.vp_lookback_bars)
        window = df.iloc[start:idx + 1]

        vp = calc_volume_profile(window, num_bins=self.vp_num_bins)
        current_price = df.iloc[idx]["close"]

        return is_thin_volume_above(
            current_price=current_price,
            volume_profile=vp,
            thin_threshold_pct=self.vp_thin_threshold_pct,
        )

    def _should_exit(self, df: pd.DataFrame, idx: int) -> bool:
        """청산 조건을 확인한다.

        청산 조건: 캔들 몸통(종가)이 9-EMA 아래에서 마감.

        Args:
            df: 지표가 추가된 DataFrame
            idx: 현재 봉의 정수 인덱스

        Returns:
            True면 즉시 청산
        """
        row = df.iloc[idx]
        return row["close"] < row["ema"]

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """전체 데이터에 대해 매매 시그널을 생성한다.

        Args:
            df: OHLCV DataFrame (지표 미포함 가능, 내부에서 자동 계산)

        Returns:
            시그널이 추가된 DataFrame
            - signal: Signal enum 값
            - position: 포지션 상태 (0=미보유, 1=보유)
        """
        # 지표 추가
        if "vwap" not in df.columns or "ema" not in df.columns:
            df = add_indicators(
                df,
                ema_period=self.ema_period,
                reset_hour_utc=self.reset_hour_utc,
            )

        signals: list[str] = []
        positions: list[int] = []
        in_position = False

        # EMA 안정화를 위한 워밍업 기간
        warmup = max(self.ema_period * 3, self.vwap_chop_lookback, self.vp_lookback_bars)

        for idx in range(len(df)):
            if idx < warmup:
                signals.append(Signal.WAIT.value)
                positions.append(0)
                continue

            if in_position:
                # 포지션 보유 중 → 청산 조건 확인
                if self._should_exit(df, idx):
                    signals.append(Signal.SELL.value)
                    positions.append(0)
                    in_position = False
                else:
                    signals.append(Signal.HOLD.value)
                    positions.append(1)
            else:
                # 포지션 미보유 → 진입 조건 확인
                if self._is_vwap_choppy(df, idx):
                    signals.append(Signal.WAIT.value)
                    positions.append(0)
                elif self._is_ema_pullback_entry(df, idx) and self._check_volume_profile(df, idx):
                    signals.append(Signal.BUY.value)
                    positions.append(1)
                    in_position = True
                else:
                    signals.append(Signal.WAIT.value)
                    positions.append(0)

        df = df.copy()
        df["signal"] = signals
        df["position"] = positions

        buy_count = signals.count(Signal.BUY.value)
        sell_count = signals.count(Signal.SELL.value)
        logger.info(f"시그널 생성 완료: BUY={buy_count}, SELL={sell_count}")

        return df
