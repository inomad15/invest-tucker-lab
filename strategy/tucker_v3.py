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
        rsi_period: RSI 계산 기간 (Phase 1 필터)
        rsi_threshold: RSI 진입 최소값 (기본 55 — 과매도 반등 배제, 추세 확인)
        volume_ratio_lookback: 거래량 비율 산출 기간
        volume_ratio_threshold: 거래량 증가 배수 요건 (기본 1.5x)
        require_mtf_agreement: 상위 타임프레임 추세 일치 요구 여부
        atr_max_pct: ATR/가격 비율 상한 (Phase 2 — 극변동 종목 차단, 기본 5.0%)
        market_cap_weight: 시총 기반 자본 할당 가중치 (Phase 2 — 메타 속성,
            향후 실매매 연동 시 포지션 사이징에 사용). 기본 1.0.
        take_profit_pct: 즉시 전량 익절 % (Phase 3, 0 이하 시 비활성). 기본 5.0.
        time_stop_bars: 진입 후 경과 시 강제 청산 봉 수 (Phase 3, 0 이하 시
            비활성). 기본 48 (5분봉 기준 4시간).
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
        rsi_period: int = 14,
        rsi_threshold: float = 55.0,
        volume_ratio_lookback: int = 20,
        volume_ratio_threshold: float = 1.5,
        require_mtf_agreement: bool = True,
        atr_max_pct: float = 5.0,
        market_cap_weight: float = 1.0,
        take_profit_pct: float = 5.0,
        time_stop_bars: int = 48,
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
        self.rsi_period = rsi_period
        self.rsi_threshold = rsi_threshold
        self.volume_ratio_lookback = volume_ratio_lookback
        self.volume_ratio_threshold = volume_ratio_threshold
        self.require_mtf_agreement = require_mtf_agreement
        self.atr_max_pct = atr_max_pct
        self.market_cap_weight = market_cap_weight
        self.take_profit_pct = take_profit_pct
        self.time_stop_bars = time_stop_bars

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
        6. RSI ≥ rsi_threshold (Phase 1 — 추세 확인, 과매도 반등 배제)
        7. volume_ratio ≥ volume_ratio_threshold (Phase 1 — 거래량 증가 요건)
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

        # 조건 6: RSI 충분 (Phase 1 필터)
        if not self._is_rsi_sufficient(df, idx):
            return False

        # 조건 7: 거래량 충분 (Phase 1 필터)
        if not self._is_volume_sufficient(df, idx):
            return False

        # 조건 8: 변동성 허용 범위 내 (Phase 2 필터 — 극변동 종목 차단)
        if not self._is_volatility_acceptable(df, idx):
            return False

        return True

    def _is_volatility_acceptable(self, df: pd.DataFrame, idx: int) -> bool:
        """ATR이 가격 대비 허용 범위 내인지 확인한다.

        ATR/close × 100 ≤ atr_max_pct 이어야 진입 가능.
        TREE 같은 극변동 알트코인의 무분별한 진입을 차단.
        `atr_max_pct ≤ 0`이면 필터 비활성화(모두 통과).

        ATR 컬럼이 없으면 즉석 계산(tucker_v3._calc_atr)으로 fallback.
        """
        if self.atr_max_pct <= 0:
            return True  # 필터 꺼짐

        if "atr" in df.columns:
            atr_val = df.iloc[idx]["atr"]
        else:
            # backward compat: 과거 add_indicators 사용 시 atr 없을 수 있음
            atr_series = self._calc_atr(df)
            atr_val = atr_series.iloc[idx]

        close_val = df.iloc[idx]["close"]
        if pd.isna(atr_val) or pd.isna(close_val) or close_val <= 0:
            return False  # 데이터 불완전 시 보수적 차단

        atr_pct = (atr_val / close_val) * 100.0
        return atr_pct <= self.atr_max_pct

    def _is_rsi_sufficient(self, df: pd.DataFrame, idx: int) -> bool:
        """RSI가 임계값(기본 55) 이상인지 확인한다.

        RSI 55+는 상승 추세 지속을 의미. 과매도(<30) 반등 시점의
        약한 매수 시그널을 필터링하여 추세 확인된 진입만 허용.
        """
        if "rsi" not in df.columns:
            return True  # RSI 미계산 시 필터 우회 (backward compat)
        rsi_val = df.iloc[idx]["rsi"]
        if pd.isna(rsi_val):
            return False
        return rsi_val >= self.rsi_threshold

    def _is_volume_sufficient(self, df: pd.DataFrame, idx: int) -> bool:
        """거래량이 최근 평균 대비 임계 배수(기본 1.5x) 이상인지 확인한다.

        거래량 없는 눌림목 반등은 노이즈일 가능성 큼. 거래량 증가 동반
        시그널만 진입하여 헛스윙 최소화.
        """
        if "volume_ratio" not in df.columns:
            return True  # volume_ratio 미계산 시 필터 우회
        vol_ratio = df.iloc[idx]["volume_ratio"]
        if pd.isna(vol_ratio):
            return False
        return vol_ratio >= self.volume_ratio_threshold

    def _is_mtf_aligned(
        self,
        htf_dfs: dict[str, pd.DataFrame] | None,
        timestamp: pd.Timestamp | None,
    ) -> bool:
        """상위 타임프레임(15m, 1h)의 추세가 상승인지 확인한다.

        각 HTF에서 가장 최근 완성 봉의 종가 > EMA 조건을 모두 만족하면 True.
        `require_mtf_agreement=False`이면 항상 True 반환.
        HTF 데이터가 없는데 요구 옵션 켜져 있으면 False (안전 측).
        """
        if not self.require_mtf_agreement:
            return True
        if not htf_dfs:
            # MTF 요구하는데 데이터 없음 → 진입 불허 (보수적)
            return False
        for tf_name, htf_df in htf_dfs.items():
            if htf_df is None or htf_df.empty or "ema" not in htf_df.columns:
                return False
            # timestamp 이하의 가장 최근 봉 사용
            if timestamp is not None:
                relevant = htf_df[htf_df.index <= timestamp]
                if relevant.empty:
                    return False
                last = relevant.iloc[-1]
            else:
                last = htf_df.iloc[-1]
            if pd.isna(last["close"]) or pd.isna(last["ema"]):
                return False
            if last["close"] <= last["ema"]:
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

    def generate_signals(
        self,
        df: pd.DataFrame,
        htf_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        """전체 데이터에 대해 매매 시그널을 생성한다.

        Args:
            df: OHLCV DataFrame (기본 타임프레임, 예: 5m)
            htf_dfs: 상위 타임프레임 DataFrame 맵 (예: {"15m": df15, "1h": df60}).
                None이거나 비어있고 `require_mtf_agreement=True`이면 진입 차단됨.
                백테스트 시 MTF 데이터가 준비되면 전달하고, 없으면 None.
        """
        required_cols = {"vwap", "ema", "rsi", "volume_ratio", "atr"}
        if not required_cols.issubset(df.columns):
            df = add_indicators(
                df,
                ema_period=self.ema_period,
                reset_hour_utc=self.reset_hour_utc,
                rsi_period=self.rsi_period,
                volume_ratio_lookback=self.volume_ratio_lookback,
                atr_period=self.atr_period,
            )

        df = df.copy()
        # ATR 컬럼이 add_indicators로 이미 있으면 그대로, 아니면 재계산
        if "atr" not in df.columns:
            df["atr"] = self._calc_atr(df)

        # HTF DataFrame에 EMA 지표가 없으면 자동 계산 (backtest 편의)
        if htf_dfs:
            for tf_name, htf_df in htf_dfs.items():
                if htf_df is not None and "ema" not in htf_df.columns:
                    htf_df["ema"] = htf_df["close"].ewm(
                        span=self.ema_period, adjust=False
                    ).mean()

        signals: list[str] = []
        positions: list[int] = []
        exit_reasons: list[str] = []

        in_position = False
        entry_price = 0.0
        entry_bar = 0  # Phase 3: 시간 스탑용
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
                pnl_pct = ((row["close"] / entry_price) - 1.0) * 100.0 if entry_price > 0 else 0.0

                # Phase 3 우선순위 1: 즉시 전량 익절 (+take_profit_pct %)
                if self.take_profit_pct > 0 and pnl_pct >= self.take_profit_pct:
                    exit_reason = "take_profit"

                # 기존: ATR 긴급 손절
                if not exit_reason and self.atr_stop_multiplier > 0:
                    atr_stop = entry_price - (row["atr"] * self.atr_stop_multiplier)
                    if row["close"] < atr_stop:
                        exit_reason = "atr_stop"

                # Phase 3 우선순위 3: 시간 스탑 (진입 후 N봉 초과 시 강제 청산)
                if not exit_reason and self.time_stop_bars > 0:
                    if (idx - entry_bar) >= self.time_stop_bars:
                        exit_reason = "time_stop"

                # 기존: EMA 아래 N봉 연속 마감
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
                elif (
                    self._is_pullback_bounce(df, idx)
                    and self._check_volume_profile(df, idx)
                    and self._is_mtf_aligned(htf_dfs, df.index[idx])
                ):
                    signals.append(Signal.BUY.value)
                    positions.append(1)
                    exit_reasons.append("")
                    in_position = True
                    entry_price = row["close"]
                    entry_bar = idx  # Phase 3: 시간 스탑 기준점
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
        take_profits = exit_reasons.count("take_profit")
        time_stops = exit_reasons.count("time_stop")

        logger.info(
            f"v3 시그널 생성: BUY={buy_count}, SELL={sell_count} "
            f"(TP={take_profits}, ATR손절={atr_stops}, "
            f"시간스탑={time_stops}, EMA청산={ema_exits})"
        )

        return df
