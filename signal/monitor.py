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
    "30m": "minute30",
    "60m": "minute60",
    "1h": "minute60",
}

# 분 단위 변환
TIMEFRAME_MINUTES: dict[str, int] = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "60m": 60, "1h": 60,
}

# Phase 1: MTF 확인용 상위 타임프레임
HTF_TIMEFRAMES: list[str] = ["15m", "1h"]
HTF_LOOKBACK_CANDLES: int = 50


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

    def _fetch_htf_candles(
        self,
        market: str,
        ema_period: int,
    ) -> dict[str, pd.DataFrame]:
        """상위 타임프레임(15m, 1h) 캔들을 조회하고 EMA를 계산한다.

        실패 시 빈 dict 반환 — 해당 HTF 필터는 자동으로 False가 되어 진입 차단.

        Args:
            market: 마켓 심볼 (예: "KRW-BTC")
            ema_period: HTF EMA 기간 (기본 타임프레임과 동일 기간 사용)

        Returns:
            {"15m": df, "1h": df} 형식의 dict (EMA 컬럼 포함). 조회 실패 HTF는 제외.
        """
        htf_dfs: dict[str, pd.DataFrame] = {}
        for tf in HTF_TIMEFRAMES:
            interval = INTERVAL_MAP.get(tf)
            if interval is None:
                continue
            try:
                df = pyupbit.get_ohlcv(
                    ticker=market,
                    interval=interval,
                    count=HTF_LOOKBACK_CANDLES,
                )
                if df is None or df.empty:
                    logger.debug(f"{market} HTF {tf} 캔들 없음")
                    continue
                df.columns = ["open", "high", "low", "close", "volume", "value"]
                df.index.name = "datetime"
                # 마지막(미완성) 봉 제외, EMA 계산
                df = df.iloc[:-1].copy()
                df["ema"] = df["close"].ewm(span=ema_period, adjust=False).mean()
                htf_dfs[tf] = df
            except Exception as e:
                logger.warning(f"{market} HTF {tf} 조회 오류: {e}")
                continue
        return htf_dfs

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
            rsi_period=strategy.rsi_period,
            volume_ratio_lookback=strategy.volume_ratio_lookback,
            atr_period=strategy.atr_period,
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

            # Phase 2: ATR 변동성 필터 (진입 이전에 먼저 체크 — 가볍고 명확)
            if not strategy._is_volatility_acceptable(analysis_df, idx):
                atr_val = analysis_df.iloc[idx].get("atr", float("nan"))
                atr_pct = (atr_val / close * 100.0) if close > 0 else float("nan")
                logger.debug(
                    f"{market} 변동성 과다 — 관망 "
                    f"(ATR/price={atr_pct:.2f}% > {strategy.atr_max_pct:.1f}%)"
                )
                return

            if not strategy._is_pullback_bounce(analysis_df, idx):
                return

            if not strategy._check_volume_profile(analysis_df, idx):
                logger.debug(f"{market} 위쪽 매물대 두꺼움 — 관망")
                return

            # Phase 1: MTF 추세 일치 확인 (15m, 1h)
            if strategy.require_mtf_agreement:
                htf_dfs = self._fetch_htf_candles(market, strategy.ema_period)
                current_ts = analysis_df.index[idx]
                if not strategy._is_mtf_aligned(htf_dfs, current_ts):
                    # HTF 중 하나라도 close<=EMA면 진입 차단
                    mtf_states = {
                        tf: f"close={htf.iloc[-1]['close']:.0f}/ema={htf.iloc[-1]['ema']:.0f}"
                        for tf, htf in htf_dfs.items() if not htf.empty
                    } if htf_dfs else {"status": "데이터 없음"}
                    logger.debug(f"{market} MTF 추세 불일치 — 관망 | {mtf_states}")
                    return

            # 진입 확정
            rsi_val = analysis_df.iloc[idx].get("rsi", float("nan"))
            vol_ratio = analysis_df.iloc[idx].get("volume_ratio", float("nan"))
            logger.info(
                f"🟢 {market} 매수 시그널 | "
                f"가격={close:,.0f} | EMA={ema_val:,.0f} | VWAP={vwap_val:,.0f} | "
                f"RSI={rsi_val:.1f} | vol_ratio={vol_ratio:.2f}x"
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
