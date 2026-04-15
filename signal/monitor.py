"""실시간 시그널 모니터링 모듈.

Upbit에서 5분봉 데이터를 주기적으로 폴링하고,
Tucker v3 전략으로 시그널을 생성하여 알림을 전송한다.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyupbit
import yaml

from signal.telegram_notifier import TelegramNotifier
from strategy.indicators import add_indicators, calc_vwap, calc_ema
from strategy.tucker_v3 import TuckerStrategyV3, Signal
from utils.logger import logger

# Upbit interval 매핑
INTERVAL_MAP: dict[str, str] = {
    "1m": "minute1",
    "3m": "minute3",
    "5m": "minute5",
    "15m": "minute15",
}

# 분 단위 변환
TIMEFRAME_MINUTES: dict[str, int] = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15,
}


@dataclass
class PositionState:
    """종목별 포지션 상태."""
    in_position: bool = False
    entry_price: float = 0.0
    entry_time: datetime | None = None
    cooldown_remaining: int = 0
    last_signal: str = ""
    last_candle_time: datetime | None = None  # 마지막 처리한 캔들 시각


@dataclass
class MonitorState:
    """전체 모니터링 상태."""
    positions: dict[str, PositionState] = field(default_factory=dict)
    total_signals: int = 0
    start_time: datetime = field(default_factory=datetime.now)


class SignalMonitor:
    """실시간 시그널 모니터.

    5분봉이 완성될 때마다 전략을 실행하고,
    매수/매도 시그널 발생 시 텔레그램 알림을 전송한다.

    Args:
        markets: 모니터링할 마켓 목록
        strategies: 마켓별 전략 인스턴스
        notifier: 텔레그램 알림기
        timeframe: 타임프레임
        lookback_candles: 지표 계산에 필요한 과거 캔들 수
        poll_interval_sec: 폴링 간격 (초)
    """

    def __init__(
        self,
        markets: list[str],
        strategies: dict[str, TuckerStrategyV3],
        notifier: TelegramNotifier | None,
        timeframe: str = "5m",
        lookback_candles: int = 200,
        poll_interval_sec: int = 30,
    ) -> None:
        self.markets = markets
        self.strategies = strategies
        self.notifier = notifier
        self.timeframe = timeframe
        self.lookback_candles = lookback_candles
        self.poll_interval_sec = poll_interval_sec

        self.interval = INTERVAL_MAP.get(timeframe)
        if self.interval is None:
            raise ValueError(f"지원하지 않는 타임프레임: {timeframe}")

        self.state = MonitorState()
        for market in markets:
            self.state.positions[market] = PositionState()

    def _fetch_candles(self, market: str) -> pd.DataFrame | None:
        """Upbit에서 최신 캔들 데이터를 조회한다."""
        try:
            df = pyupbit.get_ohlcv(
                ticker=market,
                interval=self.interval,
                count=self.lookback_candles,
            )
            if df is None or df.empty:
                logger.warning(f"{market} 캔들 데이터 조회 실패")
                return None

            df.columns = ["open", "high", "low", "close", "volume", "value"]
            df.index.name = "datetime"
            return df
        except Exception as e:
            logger.error(f"{market} 데이터 조회 오류: {e}")
            return None

    def _check_signal(self, market: str, df: pd.DataFrame) -> None:
        """단일 마켓의 시그널을 확인하고 알림을 전송한다."""
        pos = self.state.positions[market]
        strategy = self.strategies[market]

        # 마지막 캔들(아직 완성 안 된 것)은 제외하고 직전 완성 캔들 기준으로 판단
        # Upbit API는 현재 진행 중인 캔들도 포함하므로, 마지막에서 두 번째가 최신 완성 캔들
        latest_complete = df.iloc[-2]
        latest_time = df.index[-2]

        # 이미 처리한 캔들이면 스킵
        if pos.last_candle_time is not None and latest_time <= pos.last_candle_time:
            return

        pos.last_candle_time = latest_time

        # 지표 계산 (완성된 캔들까지만 사용)
        analysis_df = df.iloc[:-1].copy()
        analysis_df = add_indicators(
            analysis_df,
            ema_period=strategy.ema_period,
            reset_hour_utc=strategy.reset_hour_utc,
        )

        idx = len(analysis_df) - 1
        row = analysis_df.iloc[idx]
        ema_val = row["ema"]
        vwap_val = row["vwap"]
        close = row["close"]

        # 쿨다운 처리
        if pos.cooldown_remaining > 0:
            pos.cooldown_remaining -= 1
            logger.debug(f"{market} 쿨다운 잔여: {pos.cooldown_remaining}봉")
            return

        if pos.in_position:
            # 청산 조건 확인
            # ATR 손절
            analysis_df["atr"] = strategy._calc_atr(analysis_df)
            atr_val = analysis_df.iloc[idx]["atr"]

            exit_reason = ""

            if strategy.atr_stop_multiplier > 0:
                atr_stop = pos.entry_price - (atr_val * strategy.atr_stop_multiplier)
                if close < atr_stop:
                    exit_reason = "atr_stop"

            if not exit_reason:
                consecutive = strategy._count_consecutive_below_ema(analysis_df, idx)
                if consecutive >= strategy.exit_confirm_bars:
                    exit_reason = "ema_exit"

            if exit_reason:
                pnl_pct = ((close / pos.entry_price) - 1) * 100
                logger.info(
                    f"🔴 {market} 매도 시그널 | "
                    f"가격={close:,.0f} | 진입가={pos.entry_price:,.0f} | "
                    f"수익률={pnl_pct:+.2f}% | 사유={exit_reason}"
                )

                if self.notifier:
                    self.notifier.send_signal(
                        signal_type="SELL",
                        market=market,
                        price=close,
                        ema=ema_val,
                        vwap=vwap_val,
                        reason=exit_reason,
                    )

                pos.in_position = False
                pos.entry_price = 0.0
                pos.entry_time = None
                pos.cooldown_remaining = strategy.cooldown_bars
                pos.last_signal = "SELL"
                self.state.total_signals += 1
            else:
                pnl_pct = ((close / pos.entry_price) - 1) * 100
                logger.debug(
                    f"{market} 보유 중 | 가격={close:,.0f} | 수익률={pnl_pct:+.2f}%"
                )
        else:
            # 진입 조건 확인
            if strategy._is_vwap_choppy(analysis_df, idx):
                logger.debug(f"{market} VWAP 횡보 구간 — 관망")
                return

            if strategy._is_pullback_bounce(analysis_df, idx):
                if strategy._check_volume_profile(analysis_df, idx):
                    logger.info(
                        f"🟢 {market} 매수 시그널 | "
                        f"가격={close:,.0f} | EMA={ema_val:,.0f} | VWAP={vwap_val:,.0f}"
                    )

                    if self.notifier:
                        self.notifier.send_signal(
                            signal_type="BUY",
                            market=market,
                            price=close,
                            ema=ema_val,
                            vwap=vwap_val,
                        )

                    pos.in_position = True
                    pos.entry_price = close
                    pos.entry_time = datetime.now()
                    pos.last_signal = "BUY"
                    self.state.total_signals += 1

    def run(self) -> None:
        """모니터링 루프를 실행한다."""
        logger.info(f"시그널 모니터 시작: {', '.join(self.markets)} ({self.timeframe})")

        if self.notifier:
            self.notifier.send_startup(self.markets, self.timeframe)

        cycle = 0
        while True:
            try:
                cycle += 1
                now = datetime.now().strftime("%H:%M:%S")

                for market in self.markets:
                    df = self._fetch_candles(market)
                    if df is None:
                        continue

                    self._check_signal(market, df)

                # 10사이클마다 상태 로그
                if cycle % 10 == 0:
                    status_parts = []
                    for market in self.markets:
                        pos = self.state.positions[market]
                        if pos.in_position:
                            status_parts.append(f"{market}=보유")
                        else:
                            status_parts.append(f"{market}=관망")
                    logger.info(
                        f"[{now}] 상태: {', '.join(status_parts)} | "
                        f"총 시그널: {self.state.total_signals}"
                    )

                time.sleep(self.poll_interval_sec)

            except KeyboardInterrupt:
                logger.info("모니터링 종료 (Ctrl+C)")
                if self.notifier:
                    self.notifier.send("⏹ *Tucker Signal Bot 종료*")
                break
            except Exception as e:
                logger.error(f"모니터링 오류: {e}")
                time.sleep(60)  # 오류 시 1분 대기 후 재시도
