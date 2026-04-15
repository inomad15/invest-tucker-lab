"""백테스트 엔진 모듈.

Tucker 전략의 과거 성과를 시뮬레이션하고,
수익률, 승률, MDD 등 핵심 지표를 산출한다.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from strategy.tucker import Signal, TradeResult, TuckerStrategy
from utils.logger import logger


@dataclass
class BacktestResult:
    """백테스트 결과 요약."""
    market: str
    timeframe: str
    period: str                          # 백테스트 기간
    initial_capital: float               # 초기 자본 (KRW)
    final_capital: float                 # 최종 자본 (KRW)
    total_return_pct: float              # 총 수익률 (%)
    total_trades: int                    # 총 거래 횟수
    winning_trades: int                  # 수익 거래 수
    losing_trades: int                   # 손실 거래 수
    win_rate_pct: float                  # 승률 (%)
    avg_profit_pct: float                # 평균 수익 거래 수익률 (%)
    avg_loss_pct: float                  # 평균 손실 거래 손실률 (%)
    profit_factor: float                 # Profit Factor (총이익/총손실)
    max_drawdown_pct: float              # 최대 낙폭 MDD (%)
    sharpe_ratio: float                  # 샤프 비율 (연환산)
    trades: list[TradeResult] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)

    def summary(self) -> str:
        """결과 요약 문자열을 반환한다."""
        return (
            f"\n{'=' * 60}\n"
            f"  백테스트 결과: {self.market} ({self.timeframe})\n"
            f"  기간: {self.period}\n"
            f"{'=' * 60}\n"
            f"  초기 자본:      {self.initial_capital:>15,.0f} KRW\n"
            f"  최종 자본:      {self.final_capital:>15,.0f} KRW\n"
            f"  총 수익률:      {self.total_return_pct:>14.2f} %\n"
            f"{'─' * 60}\n"
            f"  총 거래 횟수:   {self.total_trades:>15d}\n"
            f"  승/패:          {self.winning_trades} / {self.losing_trades}\n"
            f"  승률:           {self.win_rate_pct:>14.1f} %\n"
            f"  평균 수익:      {self.avg_profit_pct:>14.2f} %\n"
            f"  평균 손실:      {self.avg_loss_pct:>14.2f} %\n"
            f"  Profit Factor:  {self.profit_factor:>14.2f}\n"
            f"{'─' * 60}\n"
            f"  최대 낙폭(MDD): {self.max_drawdown_pct:>14.2f} %\n"
            f"  샤프 비율:      {self.sharpe_ratio:>14.2f}\n"
            f"{'=' * 60}\n"
        )


class BacktestEngine:
    """백테스트 엔진.

    Tucker 전략으로 생성된 시그널을 기반으로
    매매를 시뮬레이션하고 성과를 평가한다.

    Attributes:
        initial_capital: 초기 자본 (KRW)
        fee_rate: 거래 수수료율 (기본 0.05%)
        slippage_pct: 슬리피지 (기본 0.05%)
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        fee_rate: float = 0.0005,
        slippage_pct: float = 0.05,
    ) -> None:
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage_pct = slippage_pct

    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        """슬리피지를 적용한 실행 가격을 계산한다.

        Args:
            price: 원래 가격
            is_buy: 매수 여부

        Returns:
            슬리피지 적용된 가격
        """
        slip = price * (self.slippage_pct / 100.0)
        return price + slip if is_buy else price - slip

    def run(
        self,
        df: pd.DataFrame,
        strategy: TuckerStrategy,
        market: str = "",
        timeframe: str = "",
    ) -> BacktestResult:
        """백테스트를 실행한다.

        Args:
            df: OHLCV DataFrame
            strategy: Tucker 전략 인스턴스
            market: 마켓 코드 (결과 표시용)
            timeframe: 타임프레임 (결과 표시용)

        Returns:
            BacktestResult
        """
        logger.info(f"백테스트 시작: {market} {timeframe}")

        # 시그널 생성
        signal_df = strategy.generate_signals(df)

        capital = self.initial_capital
        position_qty = 0.0          # 보유 수량
        entry_price = 0.0
        entry_time: pd.Timestamp | None = None
        entry_bar = 0

        trades: list[TradeResult] = []
        equity_values: list[float] = []

        for idx in range(len(signal_df)):
            row = signal_df.iloc[idx]
            signal = row["signal"]
            current_time = signal_df.index[idx]

            if signal == Signal.BUY.value and position_qty == 0:
                # 매수 진입: 종가 기준 (다음 봉 시가로 하는 방법도 있지만,
                # 캔들 마감 확인 후 즉시 진입하는 전략이므로 종가 사용)
                exec_price = self._apply_slippage(row["close"], is_buy=True)
                fee = capital * self.fee_rate
                available = capital - fee
                position_qty = available / exec_price
                entry_price = exec_price
                entry_time = current_time
                entry_bar = idx
                capital = 0  # 전액 투입

            elif signal == Signal.SELL.value and position_qty > 0:
                # 매도 청산
                exec_price = self._apply_slippage(row["close"], is_buy=False)
                gross = position_qty * exec_price
                fee = gross * self.fee_rate
                capital = gross - fee

                pnl_pct = ((exec_price / entry_price) - 1) * 100
                pnl_krw = capital - self.initial_capital if not trades else (
                    capital - (trades[-1].exit_price * position_qty / (1 + pnl_pct / 100))
                )

                # 간단하게 진입 자본 대비 수익률로 계산
                trade_capital_at_entry = position_qty * entry_price
                pnl_krw = capital - trade_capital_at_entry * (1 + self.fee_rate)

                trades.append(TradeResult(
                    entry_time=entry_time,
                    entry_price=entry_price,
                    exit_time=current_time,
                    exit_price=exec_price,
                    pnl_pct=pnl_pct,
                    pnl_krw=pnl_krw,
                    holding_bars=idx - entry_bar,
                ))

                position_qty = 0
                entry_price = 0.0
                entry_time = None

            # 자산 평가 (보유 중이면 현재가 기준)
            if position_qty > 0:
                current_value = position_qty * row["close"]
                equity_values.append(current_value)
            else:
                equity_values.append(capital)

        # 마지막에 포지션이 남아있으면 강제 청산
        if position_qty > 0:
            last_row = signal_df.iloc[-1]
            exec_price = last_row["close"]
            gross = position_qty * exec_price
            fee = gross * self.fee_rate
            capital = gross - fee

            pnl_pct = ((exec_price / entry_price) - 1) * 100
            trade_capital_at_entry = position_qty * entry_price
            pnl_krw = capital - trade_capital_at_entry * (1 + self.fee_rate)

            trades.append(TradeResult(
                entry_time=entry_time,
                entry_price=entry_price,
                exit_time=signal_df.index[-1],
                exit_price=exec_price,
                pnl_pct=pnl_pct,
                pnl_krw=pnl_krw,
                holding_bars=len(signal_df) - 1 - entry_bar,
            ))
            equity_values[-1] = capital
            position_qty = 0

        # 성과 지표 계산
        equity_curve = pd.Series(equity_values, index=signal_df.index)
        final_capital = equity_values[-1] if equity_values else self.initial_capital

        total_return_pct = ((final_capital / self.initial_capital) - 1) * 100

        winning = [t for t in trades if t.pnl_pct > 0]
        losing = [t for t in trades if t.pnl_pct <= 0]

        win_rate = (len(winning) / len(trades) * 100) if trades else 0.0
        avg_profit = np.mean([t.pnl_pct for t in winning]) if winning else 0.0
        avg_loss = np.mean([t.pnl_pct for t in losing]) if losing else 0.0

        total_profit = sum(t.pnl_pct for t in winning) if winning else 0.0
        total_loss = abs(sum(t.pnl_pct for t in losing)) if losing else 0.0
        profit_factor = (total_profit / total_loss) if total_loss > 0 else float("inf")

        # MDD 계산
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak * 100
        max_drawdown = drawdown.min()

        # 샤프 비율 (연환산, 무위험이자율 0% 가정)
        returns = equity_curve.pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            # 타임프레임에 따른 연환산 계수
            bars_per_year = 365 * 24 * 60  # 1분 기준
            if timeframe == "5m":
                bars_per_year = 365 * 24 * 12
            elif timeframe == "15m":
                bars_per_year = 365 * 24 * 4
            elif timeframe == "1h":
                bars_per_year = 365 * 24

            sharpe = returns.mean() / returns.std() * np.sqrt(bars_per_year)
        else:
            sharpe = 0.0

        period_str = f"{signal_df.index[0].strftime('%Y-%m-%d')} ~ {signal_df.index[-1].strftime('%Y-%m-%d')}"

        result = BacktestResult(
            market=market,
            timeframe=timeframe,
            period=period_str,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return_pct=total_return_pct,
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate_pct=win_rate,
            avg_profit_pct=avg_profit,
            avg_loss_pct=avg_loss,
            profit_factor=profit_factor,
            max_drawdown_pct=max_drawdown,
            sharpe_ratio=sharpe,
            trades=trades,
            equity_curve=equity_curve,
        )

        logger.info(f"백테스트 완료: 수익률={total_return_pct:.2f}%, 거래={len(trades)}회, 승률={win_rate:.1f}%")
        return result
